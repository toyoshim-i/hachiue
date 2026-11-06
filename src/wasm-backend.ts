import { fp16ToFp32, getBlockSpec } from './gguf.js'
import { streamModelChunks } from './storage.js'
import {
  type EngineOptions,
  GgmlType,
  type GgufTensorInfo,
  type ModelArch,
  type ParsedGgufHeader,
} from './types.js'
import { getWasmKernelModuleBytes, type WasmKernelExports } from './wasm-kernels.js'

export interface WasmTensorSpec {
  name: string
  type: GgmlType
  N: number
  K: number
  nb: number
  hadamardFolded: boolean
  byteLength: number
}

export interface WasmCoreInitPayload {
  arch: ModelArch
  maxSeqLen: number
  wasmBytes: Uint8Array
  fp16Table: Float32Array
  cos: Float32Array
  sin: Float32Array
  swaCos: Float32Array | null
  swaSin: Float32Array | null
  hadamardSigns: Array<[number, Float32Array]>
  tensors: WasmTensorSpec[]
}

export interface WasmBackend {
  readonly isWorker: boolean
  getTensorRowSpec(name: string): { type: GgmlType; rows: number; cols: number; rowBytes: number }
  readTensorRows(name: string, rowIndices: ArrayLike<number>): Promise<Uint8Array>
  writeTensorRows(name: string, rowIndices: ArrayLike<number>, data: Uint8Array): void
  forwardSegment(
    tokenIds: number[],
    computeLogits: boolean,
    greedyOnly: boolean,
    posBase?: number,
  ): Promise<{ logits?: Float32Array; greedyId?: number }>
  resetCache(): void
  flush(): Promise<void>
  dispose(): void
}

/**
 * Self-contained Wasm SIMD inference core.
 * Must not reference any module-scoped variables outside its body so it can execute both
 * directly in-process (Node.js / main thread) and inside a serialized Blob Web Worker.
 */
export function createWasmCore(init: WasmCoreInitPayload) {
  const MEM_FP16_TABLE = 0x000000
  const MEM_Q1_LUT = 0x040000
  const MEM_CONSTS = 0x040800
  const MEM_X_F32 = 0x041000
  const MEM_X_Q16 = 0x081000
  const MEM_X_PQ2_Q16 = 0x0a1000
  const MEM_X_SCALE = 0x0c1000
  const MEM_X_F32_SUM = 0x0c5000
  const MEM_Y_F32 = 0x100000
  const MEM_WEIGHTS_BASE = 0x500000
  const PREFILL_CHUNK = 8

  const arch = init.arch
  const maxSeqLen = init.maxSeqLen
  const fp16Table = init.fp16Table
  const cosTable = init.cos
  const sinTable = init.sin
  const swaCosTable = init.swaCos
  const swaSinTable = init.swaSin
  const hadamardSigns = new Map<number, Float32Array>(init.hadamardSigns)

  // Calculate total memory required for direct in-Wasm weight storage
  let totalWeightBytes = 0
  let maxSingleTensorBytes = 0
  for (const t of init.tensors) {
    let bytes = 0
    if (t.type === 0 || t.type === 1) {
      // F32 (0) or F16 (1) -> stored as F32
      bytes = ((t.N * t.K * 4 + 15) & ~15)
    } else if (t.type === 41 || t.type === 142 || t.type === 143) {
      // Q1_0 (41), PQ2_0 (142), PTQ1_0 (143) -> planar quants + scales16
      const qBytesPerBlock = t.type === 41 ? 16 : t.type === 142 ? 32 : 26
      const qBytes = ((t.N * t.nb * qBytesPerBlock + 15) & ~15)
      const sBytes = ((t.N * t.nb * 2 + 15) & ~15)
      bytes = qBytes + sBytes
    } else {
      bytes = ((t.byteLength + 15) & ~15)
    }
    totalWeightBytes += bytes
    if (bytes > maxSingleTensorBytes) maxSingleTensorBytes = bytes
  }

  let directMemory = true
  let wasmMemory: WebAssembly.Memory
  try {
    const totalPages = Math.ceil((MEM_WEIGHTS_BASE + totalWeightBytes + 65536) / 65536)
    wasmMemory = new WebAssembly.Memory({ initial: totalPages })
  } catch {
    // Fallback for 32-bit / fragmented address spaces: allocate staging Wasm memory
    directMemory = false
    const stagePages = Math.ceil((MEM_WEIGHTS_BASE + maxSingleTensorBytes + 65536) / 65536)
    wasmMemory = new WebAssembly.Memory({ initial: stagePages })
  }

  const wasmModule = new WebAssembly.Module(init.wasmBytes as BufferSource)
  const wasmInstance = new WebAssembly.Instance(wasmModule, { env: { memory: wasmMemory } })
  const kernels = wasmInstance.exports as unknown as WasmKernelExports

  const memBuf = wasmMemory.buffer
  // Populate FP16 lookup table at 0x000000
  new Float32Array(memBuf, MEM_FP16_TABLE, 65536).set(fp16Table)

  // Populate Q1_0 256-entry 8-bit LUT at 0x040000
  const q1LutView = new Int8Array(memBuf, MEM_Q1_LUT, 2048)
  for (let b = 0; b < 256; b++) {
    for (let bit = 0; bit < 8; bit++) {
      q1LutView[b * 8 + bit] = (b >> bit) & 1 ? 1 : -1
    }
  }

  // Populate PTQ1_0 Stage 3 tail constants [1, 1, 3, 3, 9, 9, 27, 27] at 0x040800
  new Int16Array(memBuf, MEM_CONSTS, 8).set([1, 1, 3, 3, 9, 9, 27, 27])

  const xF32View = new Float32Array(memBuf, MEM_X_F32, 65536)
  const xQ16View = new Int16Array(memBuf, MEM_X_Q16, 65536)
  const xPq2Q16View = new Int16Array(memBuf, MEM_X_PQ2_Q16, 65536)
  const xScaleView = new Float32Array(memBuf, MEM_X_SCALE, 4096)
  const xF32SumView = new Float32Array(memBuf, MEM_X_F32_SUM, 4096)
  const yF32View = new Float32Array(memBuf, MEM_Y_F32, 1048576)

  interface WasmTensor {
    type: GgmlType
    N: number
    K: number
    nb: number
    rowBytes: number
    hadamardFolded: boolean
    quantsPtr: number
    scalesPtr: number
    quantsView: Uint8Array
    scalesView: Uint8Array
    f32View?: Float32Array
    rawF16?: Uint8Array
    writeCursor: number
    leftover: Uint8Array
    leftoverLen: number
    blockCursor: number
  }

  const weights = new Map<string, WasmTensor>()
  let allocPtr = MEM_WEIGHTS_BASE

  for (const t of init.tensors) {
    let rowBytes = 0
    let qBytes = 0
    let sBytes = 0
    let storedType = t.type

    if (t.type === 0 || t.type === 1) {
      storedType = 0 as GgmlType // F32
      rowBytes = t.K * 4
      qBytes = (t.N * rowBytes + 15) & ~15
    } else if (t.type === 41 || t.type === 142 || t.type === 143) {
      const qBytesPerBlock = t.type === 41 ? 16 : t.type === 142 ? 32 : 26
      rowBytes = t.nb * qBytesPerBlock
      qBytes = (t.N * rowBytes + 15) & ~15
      sBytes = (t.N * t.nb * 2 + 15) & ~15
    } else {
      rowBytes = t.byteLength / t.N
      qBytes = (t.byteLength + 15) & ~15
    }

    let quantsPtr = MEM_WEIGHTS_BASE
    let scalesPtr = MEM_WEIGHTS_BASE + qBytes
    let quantsView: Uint8Array
    let scalesView: Uint8Array
    let f32View: Float32Array | undefined

    if (directMemory) {
      quantsPtr = allocPtr
      allocPtr += qBytes
      scalesPtr = allocPtr
      allocPtr += sBytes
      quantsView = new Uint8Array(memBuf, quantsPtr, t.N * rowBytes)
      scalesView = new Uint8Array(memBuf, scalesPtr, sBytes > 0 ? t.N * t.nb * 2 : 0)
      if (storedType === 0) {
        f32View = new Float32Array(memBuf, quantsPtr, t.N * t.K)
      }
    } else {
      quantsView = new Uint8Array(t.N * rowBytes)
      scalesView = new Uint8Array(sBytes > 0 ? t.N * t.nb * 2 : 0)
      if (storedType === 0) {
        f32View = new Float32Array(quantsView.buffer, quantsView.byteOffset, t.N * t.K)
      }
    }

    const blkSize = t.type === 41 ? 18 : t.type === 142 ? 34 : t.type === 143 ? 28 : 0
    weights.set(t.name, {
      type: storedType,
      N: t.N,
      K: t.K,
      nb: t.nb,
      rowBytes,
      hadamardFolded: t.hadamardFolded,
      quantsPtr,
      scalesPtr,
      quantsView,
      scalesView,
      f32View,
      rawF16: t.type === 1 ? new Uint8Array(t.byteLength) : undefined,
      writeCursor: 0,
      leftover: new Uint8Array(blkSize),
      leftoverLen: 0,
      blockCursor: 0,
    })
  }

  function pushTensorBytes(name: string, bytes: Uint8Array): void {
    const w = weights.get(name)
    if (!w) throw new Error(`hachiue: unknown tensor ${name}`)

    if (w.rawF16) {
      w.rawF16.set(bytes, w.writeCursor)
      w.writeCursor += bytes.byteLength
      return
    }

    if (w.type === 41 || w.type === 142 || w.type === 143) {
      const blockSize = w.type === 41 ? 18 : w.type === 142 ? 34 : 28
      const qBytesPerBlock = blockSize - 2
      const qView = w.quantsView
      const sView = w.scalesView

      const processBlock = (src: Uint8Array, off: number) => {
        const bIdx = w.blockCursor++
        const qOff = bIdx * qBytesPerBlock
        const sOff = bIdx * 2
        if (w.type === 143) {
          // PTQ1_0: [qs: 24B][qh: 2B][d: 2B]
          qView.set(src.subarray(off, off + 26), qOff)
          sView[sOff] = src[off + 26]
          sView[sOff + 1] = src[off + 27]
        } else {
          // Q1_0 (18B) or PQ2_0 (34B): [d: 2B][qs: 16B or 32B]
          sView[sOff] = src[off]
          sView[sOff + 1] = src[off + 1]
          qView.set(src.subarray(off + 2, off + blockSize), qOff)
        }
      }

      let pos = 0
      if (w.leftoverLen > 0) {
        const take = Math.min(blockSize - w.leftoverLen, bytes.byteLength)
        w.leftover.set(bytes.subarray(0, take), w.leftoverLen)
        w.leftoverLen += take
        pos += take
        if (w.leftoverLen === blockSize) {
          processBlock(w.leftover, 0)
          w.leftoverLen = 0
        }
      }
      while (pos + blockSize <= bytes.byteLength) {
        processBlock(bytes, pos)
        pos += blockSize
      }
      if (pos < bytes.byteLength) {
        w.leftover.set(bytes.subarray(pos), 0)
        w.leftoverLen = bytes.byteLength - pos
      }
      return
    }

    w.quantsView.set(bytes, w.writeCursor)
    w.writeCursor += bytes.byteLength
  }

  function finishTensor(name: string): void {
    const w = weights.get(name)
    if (!w) return
    if (w.rawF16) {
      const dv = new DataView(w.rawF16.buffer, w.rawF16.byteOffset, w.rawF16.byteLength)
      const numElems = w.N * w.K
      const outF32 = w.f32View!
      for (let i = 0; i < numElems; i++) {
        outF32[i] = fp16Table[dv.getUint16(i * 2, true)]
      }
      w.rawF16 = undefined
    }
    if (w.leftoverLen !== 0) {
      throw new Error(`hachiue: incomplete quantized block at end of tensor ${name} (${w.leftoverLen} leftover bytes)`)
    }
  }

  function finalizeWeights(): void {
    if (arch.tieWordEmbeddings) {
      const embd = weights.get('token_embd.weight')
      if (!embd) throw new Error('hachiue: missing token_embd.weight')
      weights.set('output.weight', {
        ...embd,
        hadamardFolded: Boolean(arch.hadamard?.foldedWeights.has('output.weight') || arch.hadamard?.tiedOutput),
      })
    }
  }

  // Pre-allocate scratch activation buffers & KV / SSM state
  const Hd = arch.hidden
  const F = arch.intermediate
  const H = arch.heads
  const KV = arch.kvHeads
  const Dh = arch.headDim
  const qDim = H * Dh
  const kvDim = KV * Dh
  const hy = arch.hybrid
  const NV = hy ? hy.linearValueHeads : 1
  const DKV = hy ? hy.linearHeadDim : 1
  const valDim = hy ? NV * DKV : Hd
  const convDim = hy ? (hy.linearKeyHeads * 2 + NV) * DKV : Math.max(Hd, qDim + 2 * kvDim)
  const maxRotDim = Math.max(Hd, F, qDim, valDim)

  const scratch = {
    curH: new Float32Array(PREFILL_CHUNK * Hd),
    nextH: new Float32Array(PREFILL_CHUNK * Hd),
    h2: new Float32Array(PREFILL_CHUNK * Hd),
    n1: new Float32Array(PREFILL_CHUNK * Hd),
    n2: new Float32Array(PREFILL_CHUNK * Hd),
    rotBuf: new Float32Array(PREFILL_CHUNK * maxRotDim),
    permBuf: new Float32Array(PREFILL_CHUNK * valDim),
    qkvMixed: new Float32Array(PREFILL_CHUNK * convDim),
    qkvConv: new Float32Array(PREFILL_CHUNK * convDim),
    aProj: new Float32Array(PREFILL_CHUNK * NV),
    bProj: new Float32Array(PREFILL_CHUNK * NV),
    gbeta: new Float32Array(2 * PREFILL_CHUNK * NV),
    core: new Float32Array(PREFILL_CHUNK * valDim),
    zGate: new Float32Array(PREFILL_CHUNK * valDim),
    normed: new Float32Array(PREFILL_CHUNK * valDim),
    qgBuf: new Float32Array(PREFILL_CHUNK * qDim * (arch.archName === 'qwen35' ? 2 : 1)),
    qBuf: new Float32Array(PREFILL_CHUNK * qDim),
    gateBuf: new Float32Array(PREFILL_CHUNK * qDim),
    kBuf: new Float32Array(PREFILL_CHUNK * kvDim),
    vBuf: new Float32Array(PREFILL_CHUNK * kvDim),
    qNormed: new Float32Array(PREFILL_CHUNK * qDim),
    kNormed: new Float32Array(PREFILL_CHUNK * kvDim),
    attOut: new Float32Array(PREFILL_CHUNK * qDim),
    mlpGate: new Float32Array(PREFILL_CHUNK * F * (arch.archName === 'plamo3' ? 2 : 1)),
    mlpUp: new Float32Array(PREFILL_CHUNK * F),
    swigluOut: new Float32Array(PREFILL_CHUNK * F),
    lastHidden: new Float32Array(Hd),
    finalNorm: new Float32Array(Hd),
    logits: new Float32Array(arch.vocab),
  }

  const kCacheQ = new Map<number, Int8Array>()
  const kCacheS = new Map<number, Float32Array>()
  const vCacheQ = new Map<number, Int8Array>()
  const vCacheS = new Map<number, Float32Array>()
  const ssmConvState = new Map<number, Float32Array>()
  const ssmRecState = new Map<number, Float32Array>()

  const qCacheElems = maxSeqLen * kvDim
  const sCacheElems = maxSeqLen * (kvDim >> 5)

  for (let li = 0; li < arch.layers; li++) {
    const isLinear = arch.hybrid && arch.hybrid.layerTypes[li] === 'linear'
    if (!isLinear) {
      kCacheQ.set(li, new Int8Array(qCacheElems))
      kCacheS.set(li, new Float32Array(sCacheElems))
      vCacheQ.set(li, new Int8Array(qCacheElems))
      vCacheS.set(li, new Float32Array(sCacheElems))
    } else if (arch.hybrid) {
      const lConvDim = (arch.hybrid.linearKeyHeads * 2 + arch.hybrid.linearValueHeads) * arch.hybrid.linearHeadDim
      const convStateLen = (arch.hybrid.convKernel - 1) * lConvDim
      const recStateLen = arch.hybrid.linearValueHeads * arch.hybrid.linearHeadDim * arch.hybrid.linearHeadDim
      ssmConvState.set(li, new Float32Array(convStateLen))
      ssmRecState.set(li, new Float32Array(recStateLen))
    }
  }

  let cachePos = 0

  function resetCache(): void {
    cachePos = 0
    for (const st of ssmConvState.values()) st.fill(0)
    for (const st of ssmRecState.values()) st.fill(0)
  }

  function runHadamard(src: Float32Array, dst: Float32Array, rows: number, width: number, mode: 0 | 1): void {
    if (!arch.hadamard) return
    const signs = hadamardSigns.get(width)
    const norm = 0.04419417382415922 // 1 / sqrt(512)
    for (let r = 0; r < rows; r++) {
      const rowBase = r * width
      for (let colBase = 0; colBase < width; colBase += 512) {
        const base = rowBase + colBase
        if (mode === 0 && signs) {
          for (let i = 0; i < 512; i++) dst[base + i] = src[base + i] * signs[colBase + i]
        } else {
          for (let i = 0; i < 512; i++) dst[base + i] = src[base + i]
        }

        for (let h = 1; h < 512; h <<= 1) {
          const step = h << 1
          for (let i = 0; i < 512; i += step) {
            for (let j = base + i; j < base + i + h; j++) {
              const a = dst[j]
              const b = dst[j + h]
              dst[j] = a + b
              dst[j + h] = a - b
            }
          }
        }

        if (mode === 1 && signs) {
          for (let i = 0; i < 512; i++) dst[base + i] = dst[base + i] * norm * signs[colBase + i]
        } else {
          for (let i = 0; i < 512; i++) dst[base + i] *= norm
        }
      }
    }
  }

  function runRmsNorm(x: Float32Array, weightName: string, rows: number, dim: number, out: Float32Array): void {
    const w = weights.get(weightName)
    if (!w?.f32View) throw new Error(`hachiue: missing norm tensor ${weightName}`)
    const gamma = w.f32View
    const eps = arch.rmsEps
    for (let r = 0; r < rows; r++) {
      const base = r * dim
      let sumSq = 0
      for (let i = 0; i < dim; i++) {
        const v = x[base + i]
        sumSq += v * v
      }
      const inv = 1.0 / Math.sqrt(sumSq / dim + eps)
      for (let i = 0; i < dim; i++) {
        out[base + i] = x[base + i] * inv * gamma[i]
      }
    }
  }

  function quantizeInput128(x: Float32Array, M: number, K: number, needPq2Deinterleave: boolean): void {
    const nb = K >> 7
    const totalBlocks = M * nb
    for (let b = 0; b < totalBlocks; b++) {
      const base = b << 7
      let amax = 0
      for (let i = 0; i < 128; i++) {
        const av = Math.abs(x[base + i])
        if (av > amax) amax = av
      }
      const scale = amax / 16384.0
      const inv = amax > 0 ? 16384.0 / amax : 0
      xScaleView[b] = scale
      for (let i = 0; i < 128; i++) {
        xQ16View[base + i] = Math.round(x[base + i] * inv)
      }
      if (needPq2Deinterleave) {
        for (let sub = 0; sub < 4; sub++) {
          const subBase = base + (sub << 5)
          for (let m = 0; m < 8; m++) {
            xPq2Q16View[subBase + m] = xQ16View[subBase + m * 4 + 0]
            xPq2Q16View[subBase + 8 + m] = xQ16View[subBase + m * 4 + 1]
            xPq2Q16View[subBase + 16 + m] = xQ16View[subBase + m * 4 + 2]
            xPq2Q16View[subBase + 24 + m] = xQ16View[subBase + m * 4 + 3]
          }
        }
      }
    }
  }

  function quantizeInput32(x: Float32Array, M: number, K: number): void {
    const nb32 = K >> 5
    const totalBlocks = M * nb32
    for (let b = 0; b < totalBlocks; b++) {
      const base = b << 5
      let amax = 0
      let fsum = 0
      for (let i = 0; i < 32; i++) {
        const v = x[base + i]
        fsum += v
        const av = Math.abs(v)
        if (av > amax) amax = av
      }
      const scale = amax / 16384.0
      const inv = amax > 0 ? 16384.0 / amax : 0
      xScaleView[b] = scale
      xF32SumView[b] = fsum
      for (let i = 0; i < 32; i++) {
        xQ16View[base + i] = Math.round(x[base + i] * inv)
      }
    }
  }

  // Track which input buffer is currently quantized in Wasm scratch memory to avoid redundant quantization
  let lastQuantizedInput: Float32Array | null = null
  let lastQuantizedBlockSize = 0
  let lastQuantizedHasPq2 = false

  function runMatmul(
    weightName: string,
    x: Float32Array,
    M: number,
    out: Float32Array,
    resid: Float32Array | null,
    rotMemo: Map<Float32Array, Float32Array>,
  ): void {
    const w = weights.get(weightName)
    if (!w) throw new Error(`hachiue: missing weight ${weightName}`)

    let input = x
    if (w.hadamardFolded && arch.hadamard) {
      let rot = rotMemo.get(x)
      if (!rot) {
        let srcForRot = x
        if (
          arch.hadamard.gdnVGrouped &&
          weightName.includes('.ssm_out.') &&
          arch.hybrid &&
          arch.hybrid.linearValueHeads > arch.hybrid.linearKeyHeads
        ) {
          const hd = arch.hybrid.linearHeadDim
          const nk = arch.hybrid.linearKeyHeads
          const rep = arch.hybrid.linearValueHeads / nk
          const width = hd * nk * rep
          const dst = scratch.permBuf
          for (let s = 0; s < M; s++) {
            const sBase = s * width
            for (let k = 0; k < nk; k++) {
              for (let r = 0; r < rep; r++) {
                const dstOff = sBase + (k * rep + r) * hd
                const srcOff = sBase + (r * nk + k) * hd
                for (let d = 0; d < hd; d++) dst[dstOff + d] = x[srcOff + d]
              }
            }
          }
          srcForRot = dst
        }
        rot = scratch.rotBuf
        runHadamard(srcForRot, rot, M, w.K, 0)
        rotMemo.set(x, rot)
      }
      input = rot
    }

    const totalOut = M * w.N
    if (resid) {
      yF32View.set(resid.subarray(0, totalOut), 0)
    }

    if (!directMemory) {
      new Uint8Array(wasmMemory.buffer, w.quantsPtr, w.quantsView.byteLength).set(w.quantsView)
      if (w.scalesView.byteLength > 0) {
        new Uint8Array(wasmMemory.buffer, w.scalesPtr, w.scalesView.byteLength).set(w.scalesView)
      }
    }

    const hasResid = resid ? 1 : 0
    if (w.type === 0) {
      xF32View.set(input.subarray(0, M * w.K), 0)
      kernels.gemv_f32(M, w.N, w.K, w.quantsPtr, hasResid, 0, w.N)
    } else if (w.type === 41 || w.type === 142 || w.type === 143) {
      const needPq2 = w.type === 142
      if (lastQuantizedInput !== input || lastQuantizedBlockSize !== 128 || (needPq2 && !lastQuantizedHasPq2)) {
        quantizeInput128(input, M, w.K, needPq2)
        lastQuantizedInput = input
        lastQuantizedBlockSize = 128
        lastQuantizedHasPq2 = needPq2
      }
      if (w.type === 41) {
        kernels.gemv_q1_0(M, w.N, w.K, w.nb, w.quantsPtr, w.scalesPtr, hasResid, 0, w.N)
      } else if (w.type === 142) {
        kernels.gemv_pq2_0(M, w.N, w.K, w.nb, w.quantsPtr, w.scalesPtr, hasResid, 0, w.N)
      } else {
        kernels.gemv_ptq1_0(M, w.N, w.K, w.nb, w.quantsPtr, w.scalesPtr, hasResid, 0, w.N)
      }
    } else {
      if (lastQuantizedInput !== input || lastQuantizedBlockSize !== 32) {
        quantizeInput32(input, M, w.K)
        lastQuantizedInput = input
        lastQuantizedBlockSize = 32
        lastQuantizedHasPq2 = false
      }
      if (w.type === 2) {
        kernels.gemv_q4_0(M, w.N, w.K, w.nb, w.quantsPtr, hasResid, 0, w.N)
      } else if (w.type === 3) {
        kernels.gemv_q4_1(M, w.N, w.K, w.nb, w.quantsPtr, hasResid, 0, w.N)
      } else if (w.type === 12) {
        kernels.gemv_q4_k(M, w.N, w.K, w.nb, w.quantsPtr, hasResid, 0, w.N)
      } else if (w.type === 14) {
        kernels.gemv_q6_k(M, w.N, w.K, w.nb, w.quantsPtr, hasResid, 0, w.N)
      } else {
        throw new Error(`hachiue: unsupported Wasm GEMV tensor type ${w.type}`)
      }
    }

    out.set(yF32View.subarray(0, totalOut), 0)
  }

  function embedGather(tokenIds: number[], out: Float32Array): void {
    const embd = weights.get('token_embd.weight')!
    const S = tokenIds.length
    const qView = embd.quantsView
    const sView = embd.scalesView
    const sDv = sView.byteLength > 0 ? new DataView(sView.buffer, sView.byteOffset, sView.byteLength) : null
    const qDv = new DataView(qView.buffer, qView.byteOffset, qView.byteLength)
    const pow3 = [1, 3, 9, 27, 81]

    for (let s = 0; s < S; s++) {
      const tok = tokenIds[s]
      const outBase = s * Hd
      if (embd.type === 0 && embd.f32View) {
        out.set(embd.f32View.subarray(tok * Hd, (tok + 1) * Hd), outBase)
      } else if (embd.type === 41) {
        // Q1_0
        const nb = Hd >> 7
        for (let b = 0; b < nb; b++) {
          const blkIdx = tok * nb + b
          const scale = fp16Table[sDv!.getUint16(blkIdx * 2, true)]
          const qOff = blkIdx * 16
          const dstOff = outBase + (b << 7)
          for (let j = 0; j < 128; j++) {
            const bit = (qView[qOff + (j >> 3)] >> (j & 7)) & 1
            out[dstOff + j] = bit ? scale : -scale
          }
        }
      } else if (embd.type === 142) {
        // PQ2_0
        const nb = Hd >> 7
        for (let b = 0; b < nb; b++) {
          const blkIdx = tok * nb + b
          const scale = fp16Table[sDv!.getUint16(blkIdx * 2, true)]
          const qOff = blkIdx * 32
          const dstOff = outBase + (b << 7)
          for (let j = 0; j < 128; j++) {
            const q = (qView[qOff + (j >> 2)] >> ((j & 3) * 2)) & 3
            out[dstOff + j] = (q - 1) * scale
          }
        }
      } else if (embd.type === 143) {
        // PTQ1_0
        const nb = Hd >> 7
        for (let b = 0; b < nb; b++) {
          const blkIdx = tok * nb + b
          const scale = fp16Table[sDv!.getUint16(blkIdx * 2, true)]
          const qOff = blkIdx * 26
          let wIdx = outBase + (b << 7)
          for (let n = 0; n < 5; n++) {
            const p = pow3[n]
            for (let m = 0; m < 16; m++) {
              const q = (qView[qOff + m] * p) & 0xff
              out[wIdx++] = (((q * 3) >> 8) - 1) * scale
            }
          }
          for (let n = 0; n < 5; n++) {
            const p = pow3[n]
            for (let m = 0; m < 8; m++) {
              const q = (qView[qOff + 16 + m] * p) & 0xff
              out[wIdx++] = (((q * 3) >> 8) - 1) * scale
            }
          }
          for (let n = 0; n < 4; n++) {
            const p = pow3[n]
            for (let h = 0; h < 2; h++) {
              const q = (qView[qOff + 24 + h] * p) & 0xff
              out[wIdx++] = (((q * 3) >> 8) - 1) * scale
            }
          }
        }
      } else if (embd.type === 2) {
        // Q4_0
        const nb = Hd >> 5
        for (let b = 0; b < nb; b++) {
          const blkOff = (tok * nb + b) * 18
          const d = fp16Table[qDv.getUint16(blkOff, true)]
          const dstOff = outBase + (b << 5)
          for (let j = 0; j < 16; j++) {
            const byte = qView[blkOff + 2 + j]
            out[dstOff + j] = ((byte & 0x0f) - 8) * d
            out[dstOff + 16 + j] = ((byte >> 4) - 8) * d
          }
        }
      } else if (embd.type === 3) {
        // Q4_1
        const nb = Hd >> 5
        for (let b = 0; b < nb; b++) {
          const blkOff = (tok * nb + b) * 20
          const d = fp16Table[qDv.getUint16(blkOff, true)]
          const m = fp16Table[qDv.getUint16(blkOff + 2, true)]
          const dstOff = outBase + (b << 5)
          for (let j = 0; j < 16; j++) {
            const byte = qView[blkOff + 4 + j]
            out[dstOff + j] = (byte & 0x0f) * d + m
            out[dstOff + 16 + j] = (byte >> 4) * d + m
          }
        }
      } else if (embd.type === 12) {
        // Q4_K
        const nb = Hd >> 8
        const sc = new Uint8Array(8)
        const mn = new Uint8Array(8)
        for (let b = 0; b < nb; b++) {
          const blkOff = (tok * nb + b) * 144
          const d = fp16Table[qDv.getUint16(blkOff, true)]
          const dmin = fp16Table[qDv.getUint16(blkOff + 2, true)]
          for (let j = 0; j < 4; j++) {
            sc[j] = qView[blkOff + 4 + j] & 63
            mn[j] = qView[blkOff + 8 + j] & 63
          }
          for (let j = 4; j < 8; j++) {
            sc[j] = (qView[blkOff + 8 + j] & 0x0f) | ((qView[blkOff + j] >> 6) << 4)
            mn[j] = (qView[blkOff + 8 + j] >> 4) | ((qView[blkOff + 4 + j] >> 6) << 4)
          }
          let qOff = blkOff + 16
          let yOff = outBase + (b << 8)
          for (let j = 0; j < 4; j++) {
            const d1 = d * sc[2 * j]
            const m1 = dmin * mn[2 * j]
            const d2 = d * sc[2 * j + 1]
            const m2 = dmin * mn[2 * j + 1]
            for (let l = 0; l < 32; l++) {
              const byte = qView[qOff + l]
              out[yOff + l] = d1 * (byte & 0x0f) - m1
              out[yOff + 32 + l] = d2 * (byte >> 4) - m2
            }
            yOff += 64
            qOff += 32
          }
        }
      } else if (embd.type === 14) {
        // Q6_K
        const nb = Hd >> 8
        for (let b = 0; b < nb; b++) {
          const blkOff = (tok * nb + b) * 210
          const d = fp16Table[qDv.getUint16(blkOff + 208, true)]
          for (let n = 0; n < 2; n++) {
            const qlBase = blkOff + n * 64
            const qhBase = blkOff + 128 + n * 32
            const scBase = blkOff + 192 + n * 8
            const yBase = outBase + (b << 8) + n * 128
            for (let l = 0; l < 32; l++) {
              const is = l >> 4
              const ql0 = qView[qlBase + l]
              const ql32 = qView[qlBase + l + 32]
              const qh = qView[qhBase + l]
              const q1 = ((ql0 & 0x0f) | (((qh >> 0) & 3) << 4)) - 32
              const q2 = ((ql32 & 0x0f) | (((qh >> 2) & 3) << 4)) - 32
              const q3 = ((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32
              const q4 = ((ql32 >> 4) | (((qh >> 6) & 3) << 4)) - 32
              out[yBase + l + 0] = d * qDv.getInt8(scBase + is + 0) * q1
              out[yBase + l + 32] = d * qDv.getInt8(scBase + is + 2) * q2
              out[yBase + l + 64] = d * qDv.getInt8(scBase + is + 4) * q3
              out[yBase + l + 96] = d * qDv.getInt8(scBase + is + 6) * q4
            }
          }
        }
      }
    }
  }

  function applyRope(
    x: Float32Array,
    S: number,
    numHeads: number,
    headDim: number,
    rotaryDim: number,
    posBase: number,
    cosTab: Float32Array,
    sinTab: Float32Array,
  ): void {
    const halfRot = rotaryDim >> 1
    for (let s = 0; s < S; s++) {
      const pos = posBase + s
      const csBase = pos * halfRot
      for (let h = 0; h < numHeads; h++) {
        const base = (s * numHeads + h) * headDim
        for (let pair = 0; pair < halfRot; pair++) {
          const c = cosTab[csBase + pair]
          const sn = sinTab[csBase + pair]
          const v0 = x[base + pair]
          const v1 = x[base + pair + halfRot]
          x[base + pair] = v0 * c - v1 * sn
          x[base + pair + halfRot] = v1 * c + v0 * sn
        }
      }
    }
  }

  function appendKvQ8(
    src: Float32Array,
    cacheQ: Int8Array,
    cacheS: Float32Array,
    S: number,
    kvHeads: number,
    headDim: number,
    posBase: number,
  ): void {
    const blocksPerToken = kvHeads * (headDim >> 5)
    const totalBlocks = S * blocksPerToken
    for (let blk = 0; blk < totalBlocks; blk++) {
      const s = Math.floor(blk / blocksPerToken)
      const rem = blk % blocksPerToken
      const srcBase = (s * blocksPerToken + rem) * 32
      const dstBlk = (posBase + s) * blocksPerToken + rem
      const dstBase = dstBlk * 32

      let amax = 0
      for (let i = 0; i < 32; i++) {
        const av = Math.abs(src[srcBase + i])
        if (av > amax) amax = av
      }
      const scale = amax / 127.0
      const inv = amax > 0 ? 127.0 / amax : 0
      cacheS[dstBlk] = scale
      for (let i = 0; i < 32; i++) {
        const q = Math.round(src[srcBase + i] * inv)
        cacheQ[dstBase + i] = q < -127 ? -127 : q > 127 ? 127 : q
      }
    }
  }

  function runAttentionQ8(
    q: Float32Array,
    kQ: Int8Array,
    kS: Float32Array,
    vQ: Int8Array,
    vS: Float32Array,
    out: Float32Array,
    S: number,
    heads: number,
    kvHeads: number,
    headDim: number,
    posBase: number,
    windowSize: number,
  ): void {
    const groupSize = heads / kvHeads
    const b32 = headDim >> 5
    const scale = 1.0 / Math.sqrt(headDim)
    const acc = new Float32Array(headDim)

    for (let qi = 0; qi < S; qi++) {
      const pos = posBase + qi
      const jStart = windowSize > 0 && pos + 1 > windowSize ? pos + 1 - windowSize : 0

      for (let h = 0; h < heads; h++) {
        const kvh = Math.floor(h / groupSize)
        const qBase = (qi * heads + h) * headDim
        acc.fill(0)
        let m = -1e30
        let l = 0.0

        for (let j = jStart; j <= pos; j++) {
          const kvRow = j * kvHeads + kvh
          const sBase = kvRow * b32
          const qOffBase = kvRow * headDim

          let dot = 0.0
          for (let b = 0; b < b32; b++) {
            let subDot = 0.0
            const qb = qBase + (b << 5)
            const kb = qOffBase + (b << 5)
            for (let d = 0; d < 32; d++) {
              subDot += q[qb + d] * kQ[kb + d]
            }
            dot += subDot * kS[sBase + b]
          }

          const score = dot * scale
          const mNew = score > m ? score : m
          const corr = Math.exp(m - mNew)
          const wgt = Math.exp(score - mNew)
          l = l * corr + wgt

          for (let b = 0; b < b32; b++) {
            const vs = wgt * vS[sBase + b]
            const vb = qOffBase + (b << 5)
            const ab = b << 5
            for (let d = 0; d < 32; d++) {
              acc[ab + d] = acc[ab + d] * corr + vs * vQ[vb + d]
            }
          }
          m = mNew
        }

        const invL = 1.0 / l
        for (let d = 0; d < headDim; d++) {
          out[qBase + d] = acc[d] * invL
        }
      }
    }
  }

  function forwardSegment(
    tokenIds: number[],
    computeLogits = true,
    greedyOnly = false,
    posBase?: number,
  ): { logits?: Float32Array; greedyId?: number } {
    if (posBase !== undefined) cachePos = posBase
    const S = tokenIds.length
    if (S === 0) throw new Error('hachiue: empty tokenIds')
    if (cachePos + S > maxSeqLen) {
      throw new Error(`hachiue: context length ${cachePos + S} exceeds maxSeqLen ${maxSeqLen}`)
    }

    const rotDim = arch.rope.rotaryDim
    const sc = scratch
    const rotMemo = new Map<Float32Array, Float32Array>()

    let curH = sc.curH
    let nextH = sc.nextH

    // 1. Embedding Lookup
    embedGather(tokenIds, curH)
    if (arch.hadamard?.inverseWeights.has('token_embd.weight')) {
      runHadamard(curH, nextH, S, Hd, 1)
      ;[curH, nextH] = [nextH, curH]
    }

    // 2. Transformer Layers
    for (let li = 0; li < arch.layers; li++) {
      runRmsNorm(curH, `blk.${li}.attn_norm.weight`, S, Hd, sc.n1)
      rotMemo.clear()
      lastQuantizedInput = null

      if (arch.archName === 'plamo3') {
        runMatmul(`blk.${li}.attn_qkv.weight`, sc.n1, S, sc.qkvMixed, null, rotMemo)
        const rowStride = qDim + 2 * kvDim
        for (let s = 0; s < S; s++) {
          const srcBase = s * rowStride
          sc.qBuf.set(sc.qkvMixed.subarray(srcBase, srcBase + qDim), s * qDim)
          sc.kBuf.set(sc.qkvMixed.subarray(srcBase + qDim, srcBase + qDim + kvDim), s * kvDim)
          sc.vBuf.set(sc.qkvMixed.subarray(srcBase + qDim + kvDim, srcBase + rowStride), s * kvDim)
        }

        runRmsNorm(sc.qBuf, `blk.${li}.attn_q_norm.weight`, S * H, Dh, sc.qNormed)
        runRmsNorm(sc.kBuf, `blk.${li}.attn_k_norm.weight`, S * KV, Dh, sc.kNormed)

        const isSwa = Boolean(arch.swa && arch.swa.layerTypes[li] === 'swa')
        const lCos = isSwa && swaCosTable ? swaCosTable : cosTable
        const lSin = isSwa && swaSinTable ? swaSinTable : sinTable
        const winSize = isSwa && arch.swa ? arch.swa.windowSize : 0

        applyRope(sc.qNormed, S, H, Dh, rotDim, cachePos, lCos, lSin)
        applyRope(sc.kNormed, S, KV, Dh, rotDim, cachePos, lCos, lSin)

        const kQ = kCacheQ.get(li)!
        const kS = kCacheS.get(li)!
        const vQ = vCacheQ.get(li)!
        const vS = vCacheS.get(li)!
        appendKvQ8(sc.kNormed, kQ, kS, S, KV, Dh, cachePos)
        appendKvQ8(sc.vBuf, vQ, vS, S, KV, Dh, cachePos)

        runAttentionQ8(sc.qNormed, kQ, kS, vQ, vS, sc.attOut, S, H, KV, Dh, cachePos, winSize)

        rotMemo.clear()
        lastQuantizedInput = null
        runMatmul(`blk.${li}.attn_output.weight`, sc.attOut, S, sc.n1, null, rotMemo)
        runRmsNorm(sc.n1, `blk.${li}.post_attention_norm`, S, Hd, sc.n2)
        for (let i = 0; i < S * Hd; i++) sc.h2[i] = curH[i] + sc.n2[i]

        runRmsNorm(sc.h2, `blk.${li}.ffn_norm.weight`, S, Hd, sc.n1)
        rotMemo.clear()
        lastQuantizedInput = null
        runMatmul(`blk.${li}.ffn_up.weight`, sc.n1, S, sc.mlpGate, null, rotMemo)
        for (let s = 0; s < S; s++) {
          const base = s * F * 2
          const outBase = s * F
          for (let c = 0; c < F; c++) {
            const g = sc.mlpGate[base + c]
            const u = sc.mlpGate[base + F + c]
            sc.swigluOut[outBase + c] = (g / (1.0 + Math.exp(-g))) * u
          }
        }

        rotMemo.clear()
        lastQuantizedInput = null
        runMatmul(`blk.${li}.ffn_down.weight`, sc.swigluOut, S, sc.n1, null, rotMemo)
        runRmsNorm(sc.n1, `blk.${li}.post_ffw_norm`, S, Hd, sc.n2)
        for (let i = 0; i < S * Hd; i++) nextH[i] = sc.h2[i] + sc.n2[i]
        ;[curH, nextH] = [nextH, curH]
        continue
      }

      const isLinear = arch.hybrid && arch.hybrid.layerTypes[li] === 'linear'
      if (isLinear && arch.hybrid) {
        const NK = arch.hybrid.linearKeyHeads
        const lNV = arch.hybrid.linearValueHeads
        const lDKV = arch.hybrid.linearHeadDim
        const keyDim = NK * lDKV
        const lValDim = lNV * lDKV
        const lConvDim = keyDim * 2 + lValDim
        const convK = arch.hybrid.convKernel
        const loadState = cachePos > 0

        runMatmul(`blk.${li}.attn_qkv.weight`, sc.n1, S, sc.qkvMixed, null, rotMemo)
        runMatmul(`blk.${li}.attn_gate.weight`, sc.n1, S, sc.zGate, null, rotMemo)

        const convW = weights.get(`blk.${li}.ssm_conv1d.weight`)!.f32View!
        const convState = ssmConvState.get(li)!

        const extVal = (m: number, c: number): number => {
          if (m + 1 < convK) {
            return loadState ? convState[m * lConvDim + c] : 0.0
          }
          return sc.qkvMixed[(m - (convK - 1)) * lConvDim + c]
        }

        for (let t = 0; t < S; t++) {
          const outBase = t * lConvDim
          for (let c = 0; c < lConvDim; c++) {
            let acc = 0.0
            const wBase = c * convK
            for (let j = 0; j < convK; j++) {
              acc += convW[wBase + j] * extVal(t + j, c)
            }
            sc.qkvConv[outBase + c] = acc / (1.0 + Math.exp(-acc))
          }
        }
        // Update convState in-place from tail
        const newConvState = new Float32Array((convK - 1) * lConvDim)
        for (let sk = 0; sk < convK - 1; sk++) {
          for (let scCol = 0; scCol < lConvDim; scCol++) {
            newConvState[sk * lConvDim + scCol] = extVal(S + sk, scCol)
          }
        }
        convState.set(newConvState)

        runMatmul(`blk.${li}.ssm_alpha.weight`, sc.n1, S, sc.aProj, null, rotMemo)
        runMatmul(`blk.${li}.ssm_beta.weight`, sc.n1, S, sc.bProj, null, rotMemo)

        const aNeg = weights.get(`blk.${li}.ssm_a`)!.f32View!
        const dtBias = weights.get(`blk.${li}.ssm_dt.bias`)!.f32View!
        const nGb = S * lNV
        for (let i = 0; i < nGb; i++) {
          const h = i % lNV
          const xVal = sc.aProj[i] + dtBias[h]
          const sp = (xVal > 0 ? xVal : 0) + Math.log(1.0 + Math.exp(-Math.abs(xVal)))
          sc.gbeta[i] = aNeg[h] * sp
          sc.gbeta[nGb + i] = 1.0 / (1.0 + Math.exp(-sc.bProj[i]))
        }

        // DeltaNet Recurrence
        const recState = ssmRecState.get(li)!
        if (!loadState) recState.fill(0)
        const scale = 1.0 / Math.sqrt(lDKV)
        const kNorm = new Float32Array(lDKV)
        const qNorm = new Float32Array(lDKV)

        for (let h = 0; h < lNV; h++) {
          const hk = h % NK
          const sBase = h * lDKV * lDKV
          for (let t = 0; t < S; t++) {
            const rowConv = t * lConvDim
            const qOff = rowConv + hk * lDKV
            const kOff = rowConv + keyDim + hk * lDKV
            const vOff = rowConv + keyDim * 2 + h * lDKV
            const gbIdx = t * lNV + h

            let sk = 0.0
            let sq = 0.0
            for (let dk = 0; dk < lDKV; dk++) {
              const qv = sc.qkvConv[qOff + dk]
              const kv = sc.qkvConv[kOff + dk]
              sq += qv * qv
              sk += kv * kv
            }
            const ik = 1.0 / Math.sqrt(sk + 1e-6)
            const iq = (1.0 / Math.sqrt(sq + 1e-6)) * scale
            for (let dk = 0; dk < lDKV; dk++) {
              qNorm[dk] = sc.qkvConv[qOff + dk] * iq
              kNorm[dk] = sc.qkvConv[kOff + dk] * ik
            }

            const gt = Math.exp(sc.gbeta[gbIdx])
            const bt = sc.gbeta[nGb + gbIdx]
            const coreBase = (t * lNV + h) * lDKV

            for (let dv = 0; dv < lDKV; dv++) {
              let kvDot = 0.0
              for (let dk = 0; dk < lDKV; dk++) {
                const sIdx = sBase + dk * lDKV + dv
                const sVal = recState[sIdx] * gt
                recState[sIdx] = sVal
                kvDot += sVal * kNorm[dk]
              }
              const delta = (sc.qkvConv[vOff + dv] - kvDot) * bt
              let o = 0.0
              for (let dk = 0; dk < lDKV; dk++) {
                const sIdx = sBase + dk * lDKV + dv
                const sVal = recState[sIdx] + kNorm[dk] * delta
                recState[sIdx] = sVal
                o += sVal * qNorm[dk]
              }
              sc.core[coreBase + dv] = o
            }
          }
        }

        // deltanet_norm_gate
        const ssmNorm = weights.get(`blk.${li}.ssm_norm.weight`)!.f32View!
        for (let r = 0; r < S * lNV; r++) {
          const base = r * lDKV
          let sumSq = 0.0
          for (let i = 0; i < lDKV; i++) {
            const c = sc.core[base + i]
            sumSq += c * c
          }
          const inv = 1.0 / Math.sqrt(sumSq / lDKV + arch.rmsEps)
          for (let i = 0; i < lDKV; i++) {
            const zz = sc.zGate[base + i]
            sc.normed[base + i] = ssmNorm[i] * (sc.core[base + i] * inv) * (zz / (1.0 + Math.exp(-zz)))
          }
        }

        rotMemo.clear()
        lastQuantizedInput = null
        runMatmul(`blk.${li}.ssm_out.weight`, sc.normed, S, sc.h2, curH, rotMemo)
      } else {
        const isQwen35Full = arch.archName === 'qwen35'
        if (isQwen35Full) {
          runMatmul(`blk.${li}.attn_q.weight`, sc.n1, S, sc.qgBuf, null, rotMemo)
          const total = S * H * Dh
          for (let idx = 0; idx < total; idx++) {
            const d = idx % Dh
            const sh = Math.floor(idx / Dh)
            const srcBase = sh * (Dh * 2)
            sc.qBuf[idx] = sc.qgBuf[srcBase + d]
            sc.gateBuf[idx] = sc.qgBuf[srcBase + Dh + d]
          }
        } else {
          runMatmul(`blk.${li}.attn_q.weight`, sc.n1, S, sc.qBuf, null, rotMemo)
        }
        runMatmul(`blk.${li}.attn_k.weight`, sc.n1, S, sc.kBuf, null, rotMemo)
        runMatmul(`blk.${li}.attn_v.weight`, sc.n1, S, sc.vBuf, null, rotMemo)

        runRmsNorm(sc.qBuf, `blk.${li}.attn_q_norm.weight`, S * H, Dh, sc.qNormed)
        runRmsNorm(sc.kBuf, `blk.${li}.attn_k_norm.weight`, S * KV, Dh, sc.kNormed)

        applyRope(sc.qNormed, S, H, Dh, rotDim, cachePos, cosTable, sinTable)
        applyRope(sc.kNormed, S, KV, Dh, rotDim, cachePos, cosTable, sinTable)

        const kQ = kCacheQ.get(li)!
        const kS = kCacheS.get(li)!
        const vQ = vCacheQ.get(li)!
        const vS = vCacheS.get(li)!
        appendKvQ8(sc.kNormed, kQ, kS, S, KV, Dh, cachePos)
        appendKvQ8(sc.vBuf, vQ, vS, S, KV, Dh, cachePos)

        runAttentionQ8(sc.qNormed, kQ, kS, vQ, vS, sc.attOut, S, H, KV, Dh, cachePos, 0)

        if (isQwen35Full) {
          const total = S * H * Dh
          for (let i = 0; i < total; i++) {
            sc.attOut[i] *= 1.0 / (1.0 + Math.exp(-sc.gateBuf[i]))
          }
        }

        rotMemo.clear()
        lastQuantizedInput = null
        runMatmul(`blk.${li}.attn_output.weight`, sc.attOut, S, sc.h2, curH, rotMemo)
      }

      const postNormName =
        arch.archName === 'qwen35' ? `blk.${li}.post_attention_norm.weight` : `blk.${li}.ffn_norm.weight`
      runRmsNorm(sc.h2, postNormName, S, Hd, sc.n2)

      rotMemo.clear()
      lastQuantizedInput = null
      runMatmul(`blk.${li}.ffn_gate.weight`, sc.n2, S, sc.mlpGate, null, rotMemo)
      runMatmul(`blk.${li}.ffn_up.weight`, sc.n2, S, sc.mlpUp, null, rotMemo)

      const totalF = S * F
      for (let i = 0; i < totalF; i++) {
        const g = sc.mlpGate[i]
        sc.swigluOut[i] = (g / (1.0 + Math.exp(-g))) * sc.mlpUp[i]
      }

      rotMemo.clear()
      lastQuantizedInput = null
      runMatmul(`blk.${li}.ffn_down.weight`, sc.swigluOut, S, nextH, sc.h2, rotMemo)
      ;[curH, nextH] = [nextH, curH]
    }

    cachePos += S
    if (!computeLogits) return {}

    // 3. Final RMSNorm & LM Head on the last token of the segment
    if (S > 1) {
      sc.lastHidden.set(curH.subarray((S - 1) * Hd, S * Hd))
      runRmsNorm(sc.lastHidden, 'output_norm.weight', 1, Hd, sc.finalNorm)
    } else {
      runRmsNorm(curH, 'output_norm.weight', 1, Hd, sc.finalNorm)
    }

    rotMemo.clear()
    lastQuantizedInput = null
    runMatmul('output.weight', sc.finalNorm, 1, sc.logits, null, rotMemo)

    if (greedyOnly) {
      let bestVal = -1e30
      let bestIdx = 0
      const vocab = arch.vocab
      for (let i = 0; i < vocab; i++) {
        const v = sc.logits[i]
        if (v > bestVal) {
          bestVal = v
          bestIdx = i
        }
      }
      return { greedyId: bestIdx }
    }

    return { logits: new Float32Array(sc.logits) }
  }

  function getTensorRowSpec(name: string): { type: GgmlType; rows: number; cols: number; rowBytes: number } {
    const w = weights.get(name)
    if (!w) throw new Error(`hachiue: tensor "${name}" not found`)
    return { type: w.type, rows: w.N, cols: w.K, rowBytes: w.rowBytes }
  }

  function readTensorRows(name: string, rowIndices: ArrayLike<number>): Uint8Array {
    const w = weights.get(name)
    if (!w) throw new Error(`hachiue: tensor "${name}" not found`)
    const count = rowIndices.length
    const out = new Uint8Array(count * w.rowBytes)
    for (let i = 0; i < count; i++) {
      const r = rowIndices[i]
      if (r < 0 || r >= w.N) {
        throw new Error(`hachiue: row index ${r} out of bounds for "${name}" (rows=${w.N})`)
      }
      out.set(w.quantsView.subarray(r * w.rowBytes, (r + 1) * w.rowBytes), i * w.rowBytes)
    }
    return out
  }

  function writeTensorRows(name: string, rowIndices: ArrayLike<number>, data: Uint8Array): void {
    const w = weights.get(name)
    if (!w) throw new Error(`hachiue: tensor "${name}" not found`)
    const count = rowIndices.length
    if (count === 0) return
    const broadcast = data.byteLength === w.rowBytes && count > 1
    if (!broadcast && data.byteLength !== count * w.rowBytes) {
      throw new Error(
        `hachiue: writeTensorRows("${name}") expected ${count * w.rowBytes} (or ${w.rowBytes}) bytes, got ${data.byteLength}`,
      )
    }
    for (let i = 0; i < count; i++) {
      const r = rowIndices[i]
      if (r < 0 || r >= w.N) {
        throw new Error(`hachiue: row index ${r} out of bounds for "${name}" (rows=${w.N})`)
      }
      const srcOff = broadcast ? 0 : i * w.rowBytes
      w.quantsView.set(data.subarray(srcOff, srcOff + w.rowBytes), r * w.rowBytes)
    }
  }

  return {
    pushTensorBytes,
    finishTensor,
    finalizeWeights,
    forwardSegment,
    resetCache,
    getTensorRowSpec,
    readTensorRows,
    writeTensorRows,
  }
}

/**
 * Self-contained Web Worker entry point.
 */
function wasmWorkerEntrypoint(): void {
  let core: ReturnType<typeof createWasmCore> | null = null
  const g = self as unknown as {
    onmessage: ((ev: MessageEvent) => void) | null
    postMessage: (message: unknown, transfer?: Transferable[]) => void
  }

  g.onmessage = (ev: MessageEvent) => {
    const msg = ev.data
    try {
      switch (msg.type) {
        case 'init':
          core = createWasmCore(msg.payload)
          g.postMessage({ id: msg.id, ok: true })
          break
        case 'pushBatch':
          for (const item of msg.items) {
            core!.pushTensorBytes(item.name, item.bytes)
            if (item.finished) core!.finishTensor(item.name)
          }
          g.postMessage({ id: msg.id, ok: true })
          break
        case 'finalize':
          core!.finalizeWeights()
          g.postMessage({ id: msg.id, ok: true })
          break
        case 'forward': {
          const res = core!.forwardSegment(msg.tokenIds, msg.computeLogits, msg.greedyOnly, msg.posBase)
          const transfer: Transferable[] = res.logits ? [res.logits.buffer] : []
          g.postMessage({ id: msg.id, ok: true, res }, transfer)
          break
        }
        case 'resetCache':
          core!.resetCache()
          g.postMessage({ id: msg.id, ok: true })
          break
        case 'readRows': {
          const out = core!.readTensorRows(msg.name, msg.rowIndices)
          g.postMessage({ id: msg.id, ok: true, out }, [out.buffer])
          break
        }
        case 'writeRows':
          core!.writeTensorRows(msg.name, msg.rowIndices, msg.data)
          g.postMessage({ id: msg.id, ok: true })
          break
      }
    } catch (err) {
      g.postMessage({ id: msg.id, ok: false, error: (err as Error).message || String(err) })
    }
  }
}

let cachedFp16Table: Float32Array | null = null
function getFp16Table(): Float32Array {
  if (cachedFp16Table) return cachedFp16Table
  cachedFp16Table = new Float32Array(65536)
  for (let i = 0; i < 65536; i++) cachedFp16Table[i] = fp16ToFp32(i)
  return cachedFp16Table
}

/**
 * Create and stream-initialize the Wasm SIMD backend (running inside a Web Worker when available).
 */
export async function createWasmBackend(
  header: ParsedGgufHeader,
  options: EngineOptions,
  cos: Float32Array,
  sin: Float32Array,
  swaCos: Float32Array | null,
  swaSin: Float32Array | null,
): Promise<WasmBackend> {
  const arch = header.arch
  const maxSeqLen = options.maxSeqLen ?? 4096
  options.onProgress?.({ phase: 'pipelines', message: 'Compiling Wasm SIMD kernels...' })

  const wasmBytes = getWasmKernelModuleBytes()
  const fp16Table = getFp16Table()

  const hadamardSigns: Array<[number, Float32Array]> = []
  if (arch.hadamard) {
    for (const [width, signsInt8] of arch.hadamard.signsByWidth.entries()) {
      const f32Signs = new Float32Array(width)
      for (let i = 0; i < width; i++) f32Signs[i] = signsInt8[i]
      hadamardSigns.push([width, f32Signs])
    }
  }

  const sortedTensors: GgufTensorInfo[] = Object.values(header.tensors).sort((a, b) => a.absOffset - b.absOffset)
  const tensorSpecs: WasmTensorSpec[] = []
  const specByName = new Map<string, { type: GgmlType; rows: number; cols: number; rowBytes: number }>()

  for (const t of sortedTensors) {
    const K = t.dims[0]
    const N = t.dims.length > 1 ? t.dims[1] : 1
    const [blkElems, blkBytes] = getBlockSpec(t.type)
    const nb = K / blkElems
    const hadamardFolded = Boolean(arch.hadamard?.foldedWeights.has(t.name))
    tensorSpecs.push({
      name: t.name,
      type: t.type,
      N,
      K,
      nb,
      hadamardFolded,
      byteLength: t.byteLength,
    })

    let rowBytes = 0
    const storedType = t.type === GgmlType.F16 ? GgmlType.F32 : t.type
    if (storedType === GgmlType.F32) {
      rowBytes = K * 4
    } else if (t.type === GgmlType.Q1_0 || t.type === GgmlType.PQ2_0 || t.type === GgmlType.PTQ1_0) {
      const qBytesPerBlock = t.type === GgmlType.Q1_0 ? 16 : t.type === GgmlType.PQ2_0 ? 32 : 26
      rowBytes = nb * qBytesPerBlock
    } else {
      rowBytes = nb * blkBytes
    }
    specByName.set(t.name, { type: storedType, rows: N, cols: K, rowBytes })
  }
  if (arch.tieWordEmbeddings) {
    const embSpec = specByName.get('token_embd.weight')
    if (embSpec) specByName.set('output.weight', { ...embSpec })
  }

  const initPayload: WasmCoreInitPayload = {
    arch,
    maxSeqLen,
    wasmBytes,
    fp16Table,
    cos,
    sin,
    swaCos,
    swaSin,
    hadamardSigns,
    tensors: tensorSpecs,
  }

  const canUseWorker =
    (options.useWorker ?? true) &&
    typeof Worker !== 'undefined' &&
    typeof Blob !== 'undefined' &&
    typeof URL !== 'undefined' &&
    typeof URL.createObjectURL === 'function'

  let worker: Worker | null = null
  let workerBlobUrl: string | null = null
  let nextReqId = 1
  const pending = new Map<number, { resolve: (val: any) => void; reject: (err: Error) => void }>()

  const callWorker = (msg: Record<string, unknown>, transfer: Transferable[] = []): Promise<any> => {
    const id = nextReqId++
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject })
      worker!.postMessage({ ...msg, id }, transfer)
    })
  }

  if (canUseWorker) {
    try {
      const src = `const createWasmCore = ${createWasmCore.toString()};\n(${wasmWorkerEntrypoint.toString()})();`
      const blob = new Blob([src], { type: 'text/javascript' })
      workerBlobUrl = URL.createObjectURL(blob)
      worker = new Worker(workerBlobUrl)
      worker.onmessage = (ev: MessageEvent) => {
        const { id, ok, res, out, error } = ev.data
        const entry = pending.get(id)
        if (!entry) return
        pending.delete(id)
        if (ok) entry.resolve(out ?? res ?? true)
        else entry.reject(new Error(error))
      }
      worker.onerror = (ev) => {
        for (const p of pending.values()) {
          p.reject(new Error(ev.message || 'hachiue: Wasm Worker error'))
        }
        pending.clear()
      }
      await callWorker({ type: 'init', payload: initPayload })
    } catch {
      if (worker) worker.terminate()
      if (workerBlobUrl) URL.revokeObjectURL(workerBlobUrl)
      worker = null
      workerBlobUrl = null
    }
  }

  const localCore = worker ? null : createWasmCore(initPayload)

  const isWorker = Boolean(worker)
  options.onProgress?.({
    phase: 'pipelines',
    message: isWorker ? 'Initializing Wasm SIMD Web Worker...' : 'Initializing Wasm SIMD kernels...',
    backend: 'wasm',
    isWorker,
  })

  // Stream GGUF chunks and route tensor slices to the Worker (or localCore)
  let routeIdx = 0
  const firstTensorOffset = sortedTensors.length > 0 ? sortedTensors[0].absOffset : header.dataStartOffset

  await streamModelChunks(
    options.url,
    firstTensorOffset,
    header.totalFileSize,
    async (chunk: Uint8Array, chunkFileOffset: number) => {
      const chunkEnd = chunkFileOffset + chunk.byteLength
      const batchItems: Array<{ name: string; bytes: Uint8Array; finished: boolean }> = []
      const transfers: Transferable[] = []

      while (routeIdx < sortedTensors.length) {
        const t = sortedTensors[routeIdx]
        const absStart = t.absOffset
        const absEnd = t.absOffset + t.byteLength
        if (absEnd <= chunkFileOffset) {
          routeIdx++
          continue
        }
        if (absStart >= chunkEnd) {
          break
        }
        const sliceStart = Math.max(0, absStart - chunkFileOffset)
        const sliceEnd = Math.min(chunk.byteLength, absEnd - chunkFileOffset)
        const finished = chunkFileOffset + sliceEnd === absEnd

        if (sliceEnd > sliceStart) {
          if (worker) {
            // Slice into a standalone transferable ArrayBuffer so main thread retains 0 bytes
            const copy = chunk.slice(sliceStart, sliceEnd)
            batchItems.push({ name: t.name, bytes: copy, finished })
            transfers.push(copy.buffer)
          } else {
            localCore!.pushTensorBytes(t.name, chunk.subarray(sliceStart, sliceEnd))
            if (finished) localCore!.finishTensor(t.name)
          }
        } else if (finished) {
          if (worker) {
            batchItems.push({ name: t.name, bytes: new Uint8Array(0), finished: true })
          } else {
            localCore!.finishTensor(t.name)
          }
        }

        if (finished) {
          routeIdx++
        } else {
          break
        }
      }

      if (worker && batchItems.length > 0) {
        await callWorker({ type: 'pushBatch', items: batchItems }, transfers)
      }
      await new Promise((resolve) => setTimeout(resolve, 0))
    },
    options,
  )

  if (worker) {
    await callWorker({ type: 'finalize' })
  } else {
    localCore!.finalizeWeights()
  }

  options.onProgress?.({
    phase: 'ready',
    message: worker ? 'Model loaded into Wasm SIMD Worker' : 'Model loaded into Wasm SIMD memory',
    backend: 'wasm',
    isWorker,
  })

  let lastPendingOp: Promise<unknown> = Promise.resolve()

  return {
    isWorker: Boolean(worker),
    getTensorRowSpec(name: string) {
      const spec = specByName.get(name)
      if (!spec) throw new Error(`hachiue: tensor "${name}" not found`)
      return spec
    },
    async readTensorRows(name: string, rowIndices: ArrayLike<number>): Promise<Uint8Array> {
      const spec = specByName.get(name)
      if (!spec) throw new Error(`hachiue: tensor "${name}" not found`)
      for (let i = 0; i < rowIndices.length; i++) {
        const r = rowIndices[i]
        if (r < 0 || r >= spec.rows) {
          throw new Error(`hachiue: row index ${r} out of bounds for "${name}" (rows=${spec.rows})`)
        }
      }
      if (rowIndices.length === 0) return new Uint8Array(0)
      if (worker) {
        await lastPendingOp
        return await callWorker({ type: 'readRows', name, rowIndices: Array.from(rowIndices) })
      }
      return localCore!.readTensorRows(name, rowIndices)
    },
    writeTensorRows(name: string, rowIndices: ArrayLike<number>, data: Uint8Array): void {
      const spec = specByName.get(name)
      if (!spec) throw new Error(`hachiue: tensor "${name}" not found`)
      const count = rowIndices.length
      if (count === 0) return
      const broadcast = data.byteLength === spec.rowBytes && count > 1
      if (!broadcast && data.byteLength !== count * spec.rowBytes) {
        throw new Error(
          `hachiue: writeTensorRows("${name}") expected ${count * spec.rowBytes} (or ${spec.rowBytes}) bytes, got ${data.byteLength}`,
        )
      }
      for (let i = 0; i < count; i++) {
        const r = rowIndices[i]
        if (r < 0 || r >= spec.rows) {
          throw new Error(`hachiue: row index ${r} out of bounds for "${name}" (rows=${spec.rows})`)
        }
      }
      if (worker) {
        const copy = new Uint8Array(data)
        lastPendingOp = callWorker({ type: 'writeRows', name, rowIndices: Array.from(rowIndices), data: copy }, [
          copy.buffer,
        ])
        return
      }
      localCore!.writeTensorRows(name, rowIndices, data)
    },
    async forwardSegment(tokenIds: number[], computeLogits: boolean, greedyOnly: boolean, posBase?: number) {
      if (worker) {
        await lastPendingOp
        return await callWorker({ type: 'forward', tokenIds, computeLogits, greedyOnly, posBase })
      }
      return localCore!.forwardSegment(tokenIds, computeLogits, greedyOnly, posBase)
    },
    resetCache(): void {
      if (worker) {
        lastPendingOp = callWorker({ type: 'resetCache' })
        return
      }
      localCore!.resetCache()
    },
    async flush(): Promise<void> {
      await lastPendingOp
    },
    dispose(): void {
      if (worker) {
        worker.terminate()
        worker = null
      }
      if (workerBlobUrl) {
        URL.revokeObjectURL(workerBlobUrl)
        workerBlobUrl = null
      }
    },
  }
}
