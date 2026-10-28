import { fetchGgufHeader, fp16ToFp32 } from './gguf.js'
import { SHADERS } from './kernels.js'
import { sampleLogits } from './sampler.js'
import { streamModelChunks } from './storage.js'
import { GgufTokenizer } from './tokenizer.js'
import {
  type ChatMessage,
  type EngineOptions,
  type GenerateOptions,
  type GenerateStats,
  GgmlType,
  type GgufTensorInfo,
  type ModelArch,
  type ParsedGgufHeader,
} from './types.js'

// WebGPU GPUBufferUsage numeric constants (safe to evaluate in Node.js before WebGPU exists)
const MAP_READ = 0x0001
const COPY_SRC = 0x0004
const COPY_DST = 0x0008
const UNIFORM = 0x0040
const STORAGE = 0x0080
const STORAGE_COPY = STORAGE | COPY_DST | COPY_SRC
const UNIFORM_COPY = UNIFORM | COPY_DST
const PREFILL_CHUNK = 16 // Max tokens per prefill segment to bound scratch memory on mobile

interface QuantChunkBuffer {
  rowStart: number
  rowCount: number
  quants: GPUBuffer
  scales16: GPUBuffer
}

interface GpuTensor {
  type: GgmlType
  N: number
  K: number
  nb: number
  hadamardFolded: boolean
  /** For F32 tensors */
  f32Buf?: GPUBuffer
  /** For quantized tensors (automatically split into <= maxBindingSize row chunks if needed) */
  chunks?: QuantChunkBuffer[]
}

/**
 * Create a progressive byte-streaming splitter that splits raw GGUF blocks
 * (`Q1_0`, `PQ2_0`, `PTQ1_0`) into planar `quants` and `scales16` GPU buffers
 * without ever holding the full tensor in CPU RAM.
 */
function createBlockSplitter(
  device: GPUDevice,
  type: GgmlType,
  chunks: QuantChunkBuffer[],
  K: number,
): { push: (bytes: Uint8Array) => number; finish: () => void } {
  const nb = K / 128
  const blockSize = type === GgmlType.Q1_0 ? 18 : type === GgmlType.PQ2_0 ? 34 : 28
  const quantBytesPerBlock = blockSize - 2

  let chunkIdx = 0
  let blockInChunk = 0
  let totalBlocksInCurChunk = chunks[0].rowCount * nb

  const MAX_STAGE = 4 * 1024 * 1024 // 4 MiB staging flush
  const qStage = new Uint8Array(MAX_STAGE)
  const sStage = new Uint8Array(MAX_STAGE)
  let qStageLen = 0
  let sStageLen = 0
  let qGpuOffset = 0
  let sGpuOffset = 0

  const leftover = new Uint8Array(blockSize)
  let leftoverLen = 0

  const flushStaging = (): number => {
    let written = 0
    const c = chunks[chunkIdx]
    if (qStageLen > 0) {
      const aligned = (qStageLen + 3) & ~3
      device.queue.writeBuffer(c.quants, qGpuOffset, qStage, 0, aligned)
      qGpuOffset += qStageLen
      written += aligned
      qStageLen = 0
    }
    if (sStageLen > 0) {
      const aligned = (sStageLen + 3) & ~3
      device.queue.writeBuffer(c.scales16, sGpuOffset, sStage, 0, aligned)
      sGpuOffset += sStageLen
      written += aligned
      sStageLen = 0
    }
    return written
  }

  const processOneBlock = (src: Uint8Array, off: number): number => {
    let bytesFlushed = 0
    if (type === GgmlType.PTQ1_0) {
      // PTQ1_0: [qs: 24B][qh: 2B][d: 2B]
      qStage.set(src.subarray(off, off + 26), qStageLen)
      qStageLen += 26
      sStage[sStageLen++] = src[off + 26]
      sStage[sStageLen++] = src[off + 27]
    } else {
      // Q1_0 (18B) or PQ2_0 (34B): [d: 2B][qs: 16B or 32B]
      sStage[sStageLen++] = src[off]
      sStage[sStageLen++] = src[off + 1]
      qStage.set(src.subarray(off + 2, off + blockSize), qStageLen)
      qStageLen += quantBytesPerBlock
    }

    blockInChunk++
    // Every 2 blocks: qStageLen is a multiple of 32 (Q1_0), 64 (PQ2_0), or 52 (PTQ1_0), and sStageLen is a multiple of 4
    if ((blockInChunk & 1) === 0 && qStageLen + 1024 >= MAX_STAGE) {
      bytesFlushed += flushStaging()
    }

    if (blockInChunk === totalBlocksInCurChunk) {
      bytesFlushed += flushStaging()
      chunkIdx++
      blockInChunk = 0
      qGpuOffset = 0
      sGpuOffset = 0
      if (chunkIdx < chunks.length) {
        totalBlocksInCurChunk = chunks[chunkIdx].rowCount * nb
      }
    }
    return bytesFlushed
  }

  return {
    push: (bytes: Uint8Array): number => {
      let flushed = 0
      let pos = 0
      if (leftoverLen > 0) {
        const take = Math.min(blockSize - leftoverLen, bytes.byteLength)
        leftover.set(bytes.subarray(0, take), leftoverLen)
        leftoverLen += take
        pos += take
        if (leftoverLen === blockSize) {
          flushed += processOneBlock(leftover, 0)
          leftoverLen = 0
        }
      }
      while (pos + blockSize <= bytes.byteLength) {
        flushed += processOneBlock(bytes, pos)
        pos += blockSize
      }
      if (pos < bytes.byteLength) {
        leftover.set(bytes.subarray(pos), 0)
        leftoverLen = bytes.byteLength - pos
      }
      return flushed
    },
    finish: (): void => {
      if (leftoverLen !== 0) {
        throw new Error(`hachiue: incomplete quantized block at end of tensor (${leftoverLen} leftover bytes)`)
      }
      flushStaging()
    },
  }
}

function buildRopeTables(arch: ModelArch, maxSeqLen: number): { cos: Float32Array; sin: Float32Array } {
  const halfDim = arch.rope.rotaryDim >> 1
  const cos = new Float32Array(maxSeqLen * halfDim)
  const sin = new Float32Array(maxSeqLen * halfDim)
  const theta = arch.rope.theta

  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const freq = 1.0 / Math.pow(theta, (2 * i) / arch.rope.rotaryDim)
      const angle = pos * freq
      cos[pos * halfDim + i] = Math.cos(angle)
      sin[pos * halfDim + i] = Math.sin(angle)
    }
  }
  return { cos, sin }
}

export class HachiueEngine {
  readonly header: ParsedGgufHeader
  readonly arch: ModelArch
  readonly tokenizer: GgufTokenizer | null
  readonly device: GPUDevice
  readonly maxSeqLen: number

  private readonly pipelines: Map<string, GPUComputePipeline>
  private readonly weights: Map<string, GpuTensor>
  private readonly hadamardSigns: Map<number, GPUBuffer>
  private readonly cosBuf: GPUBuffer
  private readonly sinBuf: GPUBuffer
  private readonly dummyBuf: GPUBuffer
  private readonly logitsBuf: GPUBuffer
  private readonly logitsReadBuf: GPUBuffer
  private readonly argmaxBuf: GPUBuffer
  private readonly argmaxReadBuf: GPUBuffer
  private readonly tokensBuf: GPUBuffer

  // KV Cache (Q8_0) for full-attention layers
  private readonly kCacheQ: Map<number, GPUBuffer>
  private readonly kCacheS: Map<number, GPUBuffer>
  private readonly vCacheQ: Map<number, GPUBuffer>
  private readonly vCacheS: Map<number, GPUBuffer>

  // Gated-DeltaNet persistent recurrent & conv state for linear-attention layers
  private readonly ssmConvState: Map<number, [GPUBuffer, GPUBuffer]>
  private readonly ssmRecState: Map<number, [GPUBuffer, GPUBuffer]>
  private ssmParity = 0
  private cachePos = 0
  private tokenHistory: number[] = []
  private disposed = false

  private constructor(
    header: ParsedGgufHeader,
    device: GPUDevice,
    maxSeqLen: number,
    pipelines: Map<string, GPUComputePipeline>,
    weights: Map<string, GpuTensor>,
    hadamardSigns: Map<number, GPUBuffer>,
    cosBuf: GPUBuffer,
    sinBuf: GPUBuffer,
  ) {
    this.header = header
    this.arch = header.arch
    this.tokenizer = header.tokenizer ? new GgufTokenizer(header.tokenizer) : null
    this.device = device
    this.maxSeqLen = maxSeqLen
    this.pipelines = pipelines
    this.weights = weights
    this.hadamardSigns = hadamardSigns
    this.cosBuf = cosBuf
    this.sinBuf = sinBuf

    const mkBuf = (size: number, usage = STORAGE_COPY): GPUBuffer =>
      device.createBuffer({ size: Math.max(16, (size + 3) & ~3), usage })

    this.dummyBuf = mkBuf(64)
    this.logitsBuf = mkBuf(this.arch.vocab * 4)
    this.logitsReadBuf = device.createBuffer({
      size: this.arch.vocab * 4,
      usage: MAP_READ | COPY_DST,
    })
    this.argmaxBuf = mkBuf(16)
    this.argmaxReadBuf = device.createBuffer({
      size: 16,
      usage: MAP_READ | COPY_DST,
    })
    this.tokensBuf = mkBuf(PREFILL_CHUNK * 4)

    this.kCacheQ = new Map()
    this.kCacheS = new Map()
    this.vCacheQ = new Map()
    this.vCacheS = new Map()
    this.ssmConvState = new Map()
    this.ssmRecState = new Map()

    const kvDim = this.arch.kvHeads * this.arch.headDim
    const qBytes = maxSeqLen * kvDim // 1 byte per element in Q8_0
    const sBytes = maxSeqLen * (kvDim / 32) * 4 // 1 float32 scale per 32 elements

    for (let li = 0; li < this.arch.layers; li++) {
      const isLinear = this.arch.hybrid && this.arch.hybrid.layerTypes[li] === 'linear'
      if (!isLinear) {
        this.kCacheQ.set(li, mkBuf(qBytes))
        this.kCacheS.set(li, mkBuf(sBytes))
        this.vCacheQ.set(li, mkBuf(qBytes))
        this.vCacheS.set(li, mkBuf(sBytes))
      } else if (this.arch.hybrid) {
        const hy = this.arch.hybrid
        const convDim = (hy.linearKeyHeads * 2 + hy.linearValueHeads) * hy.linearHeadDim
        const convStateBytes = (hy.convKernel - 1) * convDim * 4
        const recStateBytes = hy.linearValueHeads * hy.linearHeadDim * hy.linearHeadDim * 4
        this.ssmConvState.set(li, [mkBuf(convStateBytes), mkBuf(convStateBytes)])
        this.ssmRecState.set(li, [mkBuf(recStateBytes), mkBuf(recStateBytes)])
      }
    }
  }

  /**
   * Create and initialize a HachiueEngine from a GGUF URL.
   */
  static async create(options: EngineOptions): Promise<HachiueEngine> {
    const maxSeqLen = options.maxSeqLen ?? 4096
    const header = await fetchGgufHeader(options.url, options)
    const arch = header.arch

    let device = options.device
    if (!device) {
      if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error('hachiue: WebGPU is not supported in this browser')
      }
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      if (!adapter) {
        throw new Error('hachiue: Failed to acquire WebGPU adapter')
      }
      const maxBind = Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024)
      const maxBuf = Math.min(adapter.limits.maxBufferSize, 512 * 1024 * 1024)
      device = await adapter.requestDevice({
        requiredLimits: {
          maxStorageBufferBindingSize: maxBind,
          maxBufferSize: maxBuf,
        },
      })
    }

    device.lost.then((info) => {
      options.onDeviceLost?.({ reason: String(info.reason ?? 'unknown'), message: info.message })
    })

    options.onProgress?.({ phase: 'pipelines', message: 'Compiling WebGPU shaders...' })
    const pipelines = new Map<string, GPUComputePipeline>()
    await Promise.all(
      Object.entries(SHADERS).map(async ([name, code]) => {
        const mod = device!.createShaderModule({ code, label: name })
        if (typeof mod.getCompilationInfo === 'function') {
          const info = await mod.getCompilationInfo()
          const err = info.messages.find((m) => m.type === 'error')
          if (err) {
            throw new Error(`hachiue: WGSL error in ${name} (L${err.lineNum}:${err.linePos}): ${err.message}`)
          }
        }
        try {
          const pipe = await device!.createComputePipelineAsync({
            layout: 'auto',
            compute: { module: mod, entryPoint: 'main' },
          })
          pipelines.set(name, pipe)
        } catch (err) {
          throw new Error(`hachiue: pipeline compile failed in ${name}: ${(err as Error).message || err}`)
        }
      }),
    )

    // Upload Hadamard sign vectors if present
    const hadamardSigns = new Map<number, GPUBuffer>()
    if (arch.hadamard) {
      for (const [width, signsInt8] of arch.hadamard.signsByWidth.entries()) {
        const f32Signs = new Float32Array(width)
        for (let i = 0; i < width; i++) f32Signs[i] = signsInt8[i]
        const buf = device.createBuffer({ size: f32Signs.byteLength, usage: STORAGE_COPY })
        device.queue.writeBuffer(buf, 0, f32Signs)
        hadamardSigns.set(width, buf)
      }
    }

    // Build and upload RoPE tables
    const { cos, sin } = buildRopeTables(arch, maxSeqLen)
    const cosBuf = device.createBuffer({ size: cos.byteLength, usage: STORAGE_COPY })
    const sinBuf = device.createBuffer({ size: sin.byteLength, usage: STORAGE_COPY })
    device.queue.writeBuffer(cosBuf, 0, cos.buffer as ArrayBuffer, cos.byteOffset, cos.byteLength)
    device.queue.writeBuffer(sinBuf, 0, sin.buffer as ArrayBuffer, sin.byteOffset, sin.byteLength)

    // Allocate GPU buffers for all tensors and register stream consumers
    const maxChunkBindBytes = Math.min(device.limits.maxStorageBufferBindingSize, 128 * 1024 * 1024)
    const weights = new Map<string, GpuTensor>()

    interface StreamRoute {
      absStart: number
      absEnd: number
      push: (bytes: Uint8Array) => number
      finish: () => void
    }
    const routes: StreamRoute[] = []

    const sortedTensors: GgufTensorInfo[] = Object.values(header.tensors).sort((a, b) => a.absOffset - b.absOffset)

    for (const t of sortedTensors) {
      const K = t.dims[0]
      const N = t.dims.length > 1 ? t.dims[1] : 1
      const hadamardFolded = Boolean(arch.hadamard?.foldedWeights.has(t.name))

      if (t.type === GgmlType.F32 || t.type === GgmlType.F16) {
        const numElems = K * N
        const f32Buf = device.createBuffer({ size: Math.max(16, numElems * 4), usage: STORAGE_COPY })
        weights.set(t.name, { type: GgmlType.F32, N, K, nb: 1, hadamardFolded, f32Buf })

        if (t.type === GgmlType.F32) {
          let gpuOff = 0
          routes.push({
            absStart: t.absOffset,
            absEnd: t.absOffset + t.byteLength,
            push: (b: Uint8Array): number => {
              device!.queue.writeBuffer(f32Buf, gpuOff, b.buffer, b.byteOffset, b.byteLength)
              gpuOff += b.byteLength
              return b.byteLength
            },
            finish: () => {},
          })
        } else {
          const rawF16 = new Uint8Array(t.byteLength)
          let wOff = 0
          routes.push({
            absStart: t.absOffset,
            absEnd: t.absOffset + t.byteLength,
            push: (b: Uint8Array): number => {
              rawF16.set(b, wOff)
              wOff += b.byteLength
              return 0
            },
            finish: () => {
              const dv = new DataView(rawF16.buffer, rawF16.byteOffset, rawF16.byteLength)
              const f32Arr = new Float32Array(numElems)
              for (let i = 0; i < numElems; i++) f32Arr[i] = fp16ToFp32(dv.getUint16(i * 2, true))
              device!.queue.writeBuffer(f32Buf, 0, f32Arr)
            },
          })
        }
      } else if (t.type === GgmlType.Q1_0 || t.type === GgmlType.PQ2_0 || t.type === GgmlType.PTQ1_0) {
        const nb = K / 128
        const quantBytesPerBlock = t.type === GgmlType.Q1_0 ? 16 : t.type === GgmlType.PQ2_0 ? 32 : 26
        const quantBytesPerRow = nb * quantBytesPerBlock
        const maxRowsPerChunk = Math.max(1, Math.floor(maxChunkBindBytes / quantBytesPerRow))

        const chunks: QuantChunkBuffer[] = []
        for (let r = 0; r < N; r += maxRowsPerChunk) {
          const rowCount = Math.min(maxRowsPerChunk, N - r)
          const qBytes = Math.max(16, (rowCount * quantBytesPerRow + 3) & ~3)
          const sBytes = Math.max(16, (rowCount * nb * 2 + 3) & ~3)
          chunks.push({
            rowStart: r,
            rowCount,
            quants: device.createBuffer({ size: qBytes, usage: STORAGE_COPY }),
            scales16: device.createBuffer({ size: sBytes, usage: STORAGE_COPY }),
          })
        }

        const gpuTensor: GpuTensor = {
          type: t.type,
          N,
          K,
          nb,
          hadamardFolded,
          chunks,
        }
        weights.set(t.name, gpuTensor)

        const splitter = createBlockSplitter(device, t.type, chunks, K)
        routes.push({
          absStart: t.absOffset,
          absEnd: t.absOffset + t.byteLength,
          push: splitter.push,
          finish: splitter.finish,
        })
      } else {
        throw new Error(`hachiue: unsupported tensor type ${t.type} for ${t.name}`)
      }
    }

    // Share token_embd.weight with output.weight when tie_word_embeddings is true
    if (arch.tieWordEmbeddings) {
      const embd = weights.get('token_embd.weight')
      if (!embd) throw new Error('hachiue: missing token_embd.weight')
      weights.set('output.weight', {
        ...embd,
        hadamardFolded: Boolean(arch.hadamard?.foldedWeights.has('output.weight') || arch.hadamard?.tiedOutput),
      })
    }

    // Stream chunks from OPFS / HTTP Range with GPU staging backpressure
    let routeIdx = 0
    let stagedBytesSinceSync = 0
    const SYNC_THRESHOLD = 16 * 1024 * 1024 // Wait for GPU queue every 16 MiB to prevent staging OOM on mobile

    const firstTensorOffset = routes.length > 0 ? routes[0].absStart : header.dataStartOffset
    await streamModelChunks(
      options.url,
      firstTensorOffset,
      header.totalFileSize,
      async (chunk: Uint8Array, chunkFileOffset: number) => {
        const chunkEnd = chunkFileOffset + chunk.byteLength
        while (routeIdx < routes.length) {
          const r = routes[routeIdx]
          if (r.absEnd <= chunkFileOffset) {
            routeIdx++
            continue
          }
          if (r.absStart >= chunkEnd) {
            break
          }
          const sliceStart = Math.max(0, r.absStart - chunkFileOffset)
          const sliceEnd = Math.min(chunk.byteLength, r.absEnd - chunkFileOffset)
          if (sliceEnd > sliceStart) {
            stagedBytesSinceSync += r.push(chunk.subarray(sliceStart, sliceEnd))
          }
          if (chunkFileOffset + sliceEnd === r.absEnd) {
            r.finish()
            routeIdx++
          } else {
            break
          }
        }
        if (stagedBytesSinceSync >= SYNC_THRESHOLD) {
          stagedBytesSinceSync = 0
          await device!.queue.onSubmittedWorkDone()
        }
      },
      options,
    )

    await device.queue.onSubmittedWorkDone()
    options.onProgress?.({ phase: 'ready', message: 'Model loaded into WebGPU memory' })

    return new HachiueEngine(header, device, maxSeqLen, pipelines, weights, hadamardSigns, cosBuf, sinBuf)
  }

  /**
   * Reset conversation KV cache and recurrent states.
   */
  resetCache(): void {
    this.cachePos = 0
    this.ssmParity = 0
    this.tokenHistory = []
  }

  /**
   * Destroy all GPU buffers.
   */
  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.device.destroy()
  }

  private mkUniform(words: Array<number>, floatIndices: Set<number> = new Set()): GPUBuffer {
    const byteLen = Math.max(16, Math.ceil(words.length / 4) * 16)
    const ab = new ArrayBuffer(byteLen)
    const dv = new DataView(ab)
    for (let i = 0; i < words.length; i++) {
      if (floatIndices.has(i)) dv.setFloat32(i * 4, words[i], true)
      else dv.setUint32(i * 4, words[i] >>> 0, true)
    }
    const buf = this.device.createBuffer({ size: byteLen, usage: UNIFORM_COPY })
    this.device.queue.writeBuffer(buf, 0, ab)
    return buf
  }

  private dispatch(
    pass: GPUComputePassEncoder,
    shaderName: string,
    uniformBuf: GPUBuffer,
    storageBufs: GPUBuffer[],
    gx: number,
    gy = 1,
    gz = 1,
  ): void {
    const pipe = this.pipelines.get(shaderName)
    if (!pipe) throw new Error(`hachiue: missing pipeline ${shaderName}`)
    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: uniformBuf } }]
    for (let i = 0; i < storageBufs.length; i++) {
      entries.push({ binding: i + 1, resource: { buffer: storageBufs[i] } })
    }
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries })
    pass.setPipeline(pipe)
    pass.setBindGroup(0, bg)
    pass.dispatchWorkgroups(gx, gy, gz)
  }

  private runHadamard(
    pass: GPUComputePassEncoder,
    srcBuf: GPUBuffer,
    dstBuf: GPUBuffer,
    rows: number,
    width: number,
    mode: 0 | 1,
    tempBufs: GPUBuffer[],
  ): void {
    if (!this.arch.hadamard) return
    const signBuf = this.hadamardSigns.get(width)
    const hasSigns = signBuf ? 1 : 0
    const u = this.mkUniform([rows, width, mode, hasSigns])
    tempBufs.push(u)
    const totalBlocks = rows * (width >> 9)
    this.dispatch(pass, 'hadamard_512', u, [srcBuf, signBuf ?? this.dummyBuf, dstBuf], totalBlocks)
  }

  private runRmsNorm(
    pass: GPUComputePassEncoder,
    xBuf: GPUBuffer,
    weightName: string,
    rows: number,
    dim: number,
    outBuf: GPUBuffer,
    tempBufs: GPUBuffer[],
  ): void {
    const w = this.weights.get(weightName)
    if (!w?.f32Buf) throw new Error(`hachiue: missing norm tensor ${weightName}`)
    // Note: GGUF converter already bakes (1 + weight) into Qwen3.5 RMSNorm tensors.
    const u = this.mkUniform([rows, dim, this.arch.rmsEps, 0], new Set([2]))
    tempBufs.push(u)
    this.dispatch(pass, 'rmsnorm', u, [xBuf, w.f32Buf, outBuf], rows)
  }

  private runMatmul(
    pass: GPUComputePassEncoder,
    weightName: string,
    xBuf: GPUBuffer,
    M: number,
    outBuf: GPUBuffer,
    residBuf: GPUBuffer | null,
    tempBufs: GPUBuffer[],
    rotMemo: Map<GPUBuffer, GPUBuffer>,
  ): void {
    const w = this.weights.get(weightName)
    if (!w) throw new Error(`hachiue: missing weight ${weightName}`)

    let inputBuf = xBuf
    if (w.hadamardFolded && this.arch.hadamard) {
      let rot = rotMemo.get(xBuf)
      if (!rot) {
        rot = this.device.createBuffer({ size: Math.max(16, M * w.K * 4), usage: STORAGE_COPY })
        tempBufs.push(rot)
        this.runHadamard(pass, xBuf, rot, M, w.K, 0, tempBufs)
        rotMemo.set(xBuf, rot)
      }
      inputBuf = rot
    }

    if (w.type === GgmlType.F32 && w.f32Buf) {
      const groups = Math.ceil(w.N / 4)
      const gx = Math.min(groups, 65535)
      const gy = Math.ceil(groups / gx)
      const u = this.mkUniform([M, w.N, w.K, gx])
      tempBufs.push(u)
      this.dispatch(pass, 'gemv_f32', u, [inputBuf, w.f32Buf, outBuf], gx, gy, M)
      return
    }

    if (!w.chunks) throw new Error(`hachiue: missing quantized chunks for ${weightName}`)
    const shader =
      w.type === GgmlType.Q1_0 ? 'gemv_q1_0' : w.type === GgmlType.PQ2_0 ? 'gemv_pq2_0' : 'gemv_ptq1_0'

    for (const chunk of w.chunks) {
      const groups = Math.ceil(chunk.rowCount / 4)
      const gx = Math.min(groups, 65535)
      const gy = Math.ceil(groups / gx)
      const u = this.mkUniform([M, chunk.rowCount, w.K, w.nb, gx, residBuf ? 1 : 0, chunk.rowStart, 0])
      tempBufs.push(u)
      this.dispatch(
        pass,
        shader,
        u,
        [inputBuf, chunk.quants, chunk.scales16, residBuf ?? this.dummyBuf, outBuf],
        gx,
        gy,
        M,
      )
    }
  }

  /**
   * Run forward pass for a segment of `tokenIds` (`S <= PREFILL_CHUNK`) starting at `this.cachePos`.
   * If `computeLogits` is true, computes logits for the last token in the segment.
   */
  async forwardSegment(
    tokenIds: number[],
    computeLogits = true,
    greedyOnly = false,
  ): Promise<{ logits?: Float32Array; greedyId?: number }> {
    const S = tokenIds.length
    if (S === 0) throw new Error('hachiue: empty tokenIds')
    if (this.cachePos + S > this.maxSeqLen) {
      throw new Error(`hachiue: context length ${this.cachePos + S} exceeds maxSeqLen ${this.maxSeqLen}`)
    }

    const H = this.arch.heads
    const KV = this.arch.kvHeads
    const Dh = this.arch.headDim
    const Hd = this.arch.hidden
    const F = this.arch.intermediate
    const rotDim = this.arch.rope.rotaryDim

    const tempBufs: GPUBuffer[] = []
    const rotMemo = new Map<GPUBuffer, GPUBuffer>()
    const mkAct = (elems: number): GPUBuffer => {
      const b = this.device.createBuffer({ size: Math.max(16, elems * 4), usage: STORAGE_COPY })
      tempBufs.push(b)
      return b
    }

    const tokArr = new Uint32Array(tokenIds)
    this.device.queue.writeBuffer(this.tokensBuf, 0, tokArr)

    const enc = this.device.createCommandEncoder()
    const pass = enc.beginComputePass()

    // 1. Embedding Lookup
    const embdW = this.weights.get('token_embd.weight')!
    const embMode = embdW.type === GgmlType.Q1_0 ? 0 : embdW.type === GgmlType.PQ2_0 ? 1 : 2
    let curH = mkAct(S * Hd)
    for (const embChunk of embdW.chunks!) {
      const uEmb = this.mkUniform([Hd, embMode, S, embChunk.rowStart, embChunk.rowCount, 0, 0, 0])
      tempBufs.push(uEmb)
      this.dispatch(
        pass,
        'embed_gather',
        uEmb,
        [this.tokensBuf, embChunk.quants, embChunk.scales16, curH],
        Math.ceil(Hd / 256),
        S,
      )
    }

    if (this.arch.hadamard?.inverseWeights.has('token_embd.weight')) {
      const unrotH = mkAct(S * Hd)
      this.runHadamard(pass, curH, unrotH, S, Hd, 1, tempBufs)
      curH = unrotH
    }

    // 2. Transformer Layers
    for (let li = 0; li < this.arch.layers; li++) {
      const n1 = mkAct(S * Hd)
      this.runRmsNorm(pass, curH, `blk.${li}.attn_norm.weight`, S, Hd, n1, tempBufs)

      const isLinear = this.arch.hybrid && this.arch.hybrid.layerTypes[li] === 'linear'
      const h2 = mkAct(S * Hd)

      if (isLinear && this.arch.hybrid) {
        const hy = this.arch.hybrid
        const NK = hy.linearKeyHeads
        const NV = hy.linearValueHeads
        const DKV = hy.linearHeadDim
        const keyDim = NK * DKV
        const valDim = NV * DKV
        const convDim = keyDim * 2 + valDim
        const convK = hy.convKernel
        const loadState = this.cachePos > 0 ? 1 : 0
        const par = this.ssmParity

        const qkvMixed = mkAct(S * convDim)
        this.runMatmul(pass, `blk.${li}.attn_qkv.weight`, n1, S, qkvMixed, null, tempBufs, rotMemo)

        const qkvConv = mkAct(S * convDim)
        const convW = this.weights.get(`blk.${li}.ssm_conv1d.weight`)!.f32Buf!
        const convStates = this.ssmConvState.get(li)!
        const uConv = this.mkUniform([S, convDim, convK, loadState])
        tempBufs.push(uConv)
        const totalConvThreads = S * convDim + (convK - 1) * convDim
        const convGroups = Math.ceil(totalConvThreads / 64)
        const cgx = Math.min(convGroups, 65535)
        const cgy = Math.ceil(convGroups / cgx)
        this.dispatch(
          pass,
          'conv1d_causal',
          uConv,
          [qkvMixed, convW, convStates[par], qkvConv, convStates[par ^ 1]],
          cgx,
          cgy,
        )

        const aProj = mkAct(S * NV)
        const bProj = mkAct(S * NV)
        this.runMatmul(pass, `blk.${li}.ssm_alpha.weight`, n1, S, aProj, null, tempBufs, rotMemo)
        this.runMatmul(pass, `blk.${li}.ssm_beta.weight`, n1, S, bProj, null, tempBufs, rotMemo)

        const gbeta = mkAct(2 * S * NV)
        const aNegBuf = this.weights.get(`blk.${li}.ssm_a`)!.f32Buf!
        const dtBiasBuf = this.weights.get(`blk.${li}.ssm_dt.bias`)!.f32Buf!
        const uGb = this.mkUniform([S, NV, 0, 0])
        tempBufs.push(uGb)
        this.dispatch(pass, 'deltanet_gbeta', uGb, [aProj, bProj, aNegBuf, dtBiasBuf, gbeta], Math.ceil((S * NV) / 64))

        const core = mkAct(S * valDim)
        const recStates = this.ssmRecState.get(li)!
        const uRec = this.mkUniform([S, NV, DKV, DKV, NK, loadState, 0, 0])
        tempBufs.push(uRec)
        this.dispatch(pass, 'deltanet_recur', uRec, [qkvConv, gbeta, recStates[par], core, recStates[par ^ 1]], NV)

        const zGate = mkAct(S * valDim)
        this.runMatmul(pass, `blk.${li}.attn_gate.weight`, n1, S, zGate, null, tempBufs, rotMemo)

        const normed = mkAct(S * valDim)
        const ssmNormBuf = this.weights.get(`blk.${li}.ssm_norm.weight`)!.f32Buf!
        const uNormG = this.mkUniform([S * NV, DKV, this.arch.rmsEps, 0], new Set([2]))
        tempBufs.push(uNormG)
        this.dispatch(pass, 'deltanet_norm_gate', uNormG, [core, zGate, ssmNormBuf, normed], S * NV)

        this.runMatmul(pass, `blk.${li}.ssm_out.weight`, normed, S, h2, curH, tempBufs, rotMemo)
      } else {
        // Full Attention Layer (Qwen3 dense or Qwen3.5 full-attn interval layer)
        const isQwen35Full = this.arch.archName === 'qwen35'
        let qBuf: GPUBuffer
        let gateBuf: GPUBuffer | null = null

        if (isQwen35Full) {
          const qgBuf = mkAct(S * H * Dh * 2)
          this.runMatmul(pass, `blk.${li}.attn_q.weight`, n1, S, qgBuf, null, tempBufs, rotMemo)
          qBuf = mkAct(S * H * Dh)
          gateBuf = mkAct(S * H * Dh)
          const uSplit = this.mkUniform([S, H, Dh, 0])
          tempBufs.push(uSplit)
          this.dispatch(pass, 'split_q_gate', uSplit, [qgBuf, qBuf, gateBuf], Math.ceil((S * H * Dh) / 256))
        } else {
          qBuf = mkAct(S * H * Dh)
          this.runMatmul(pass, `blk.${li}.attn_q.weight`, n1, S, qBuf, null, tempBufs, rotMemo)
        }

        const kBuf = mkAct(S * KV * Dh)
        const vBuf = mkAct(S * KV * Dh)
        this.runMatmul(pass, `blk.${li}.attn_k.weight`, n1, S, kBuf, null, tempBufs, rotMemo)
        this.runMatmul(pass, `blk.${li}.attn_v.weight`, n1, S, vBuf, null, tempBufs, rotMemo)

        // Per-head Q/K RMSNorm
        const qNormed = mkAct(S * H * Dh)
        const kNormed = mkAct(S * KV * Dh)
        this.runRmsNorm(pass, qBuf, `blk.${li}.attn_q_norm.weight`, S * H, Dh, qNormed, tempBufs)
        this.runRmsNorm(pass, kBuf, `blk.${li}.attn_k_norm.weight`, S * KV, Dh, kNormed, tempBufs)

        // RoPE (in-place on qNormed and kNormed)
        const uRopeQ = this.mkUniform([S, H, Dh, rotDim, this.cachePos, 0, 0, 0])
        const uRopeK = this.mkUniform([S, KV, Dh, rotDim, this.cachePos, 0, 0, 0])
        tempBufs.push(uRopeQ, uRopeK)
        this.dispatch(
          pass,
          'rope',
          uRopeQ,
          [this.cosBuf, this.sinBuf, qNormed],
          Math.ceil((S * H * (rotDim >> 1)) / 64),
        )
        this.dispatch(
          pass,
          'rope',
          uRopeK,
          [this.cosBuf, this.sinBuf, kNormed],
          Math.ceil((S * KV * (rotDim >> 1)) / 64),
        )

        // Append K and V to Q8_0 KV Cache
        const uKv = this.mkUniform([S, KV, Dh, this.cachePos])
        tempBufs.push(uKv)
        const kvBlocks = S * KV * (Dh >> 5)
        const kQ = this.kCacheQ.get(li)!
        const kS = this.kCacheS.get(li)!
        const vQ = this.vCacheQ.get(li)!
        const vS = this.vCacheS.get(li)!
        this.dispatch(pass, 'kv_append_q8', uKv, [kNormed, kQ, kS], kvBlocks)
        this.dispatch(pass, 'kv_append_q8', uKv, [vBuf, vQ, vS], kvBlocks)

        // Online GQA Attention
        const attOut = mkAct(S * H * Dh)
        const uAtt = this.mkUniform([S, H, KV, Dh, this.cachePos, 0, 0, 0])
        tempBufs.push(uAtt)
        this.dispatch(pass, 'attention_q8', uAtt, [qNormed, kQ, kS, vQ, vS, attOut], S * H)

        if (gateBuf) {
          const uGate = this.mkUniform([S * H * Dh, 0, 0, 0])
          tempBufs.push(uGate)
          this.dispatch(pass, 'sigmoid_gate', uGate, [gateBuf, attOut], Math.ceil((S * H * Dh) / 256))
        }

        this.runMatmul(pass, `blk.${li}.attn_output.weight`, attOut, S, h2, curH, tempBufs, rotMemo)
      }

      // Post-Attention Norm & SwiGLU MLP
      const postNormName =
        this.arch.archName === 'qwen35' ? `blk.${li}.post_attention_norm.weight` : `blk.${li}.ffn_norm.weight`
      const n2 = mkAct(S * Hd)
      this.runRmsNorm(pass, h2, postNormName, S, Hd, n2, tempBufs)

      const gate = mkAct(S * F)
      const up = mkAct(S * F)
      this.runMatmul(pass, `blk.${li}.ffn_gate.weight`, n2, S, gate, null, tempBufs, rotMemo)
      this.runMatmul(pass, `blk.${li}.ffn_up.weight`, n2, S, up, null, tempBufs, rotMemo)

      const swigluOut = mkAct(S * F)
      const uSwi = this.mkUniform([S * F, 0, 0, 0])
      tempBufs.push(uSwi)
      this.dispatch(pass, 'swiglu', uSwi, [gate, up, swigluOut], Math.ceil((S * F) / 256))

      const nextH = mkAct(S * Hd)
      this.runMatmul(pass, `blk.${li}.ffn_down.weight`, swigluOut, S, nextH, h2, tempBufs, rotMemo)
      curH = nextH
    }

    if (this.arch.hybrid) {
      this.ssmParity ^= 1
    }
    this.cachePos += S

    if (!computeLogits) {
      pass.end()
      this.device.queue.submit([enc.finish()])
      await this.device.queue.onSubmittedWorkDone()
      for (const b of tempBufs) b.destroy()
      return {}
    }

    // 3. Final RMSNorm & LM Head on the last token of the segment
    let lastHidden = curH
    if (S > 1) {
      pass.end()
      lastHidden = mkAct(Hd)
      enc.copyBufferToBuffer(curH, (S - 1) * Hd * 4, lastHidden, 0, Hd * 4)
      const pass2 = enc.beginComputePass()
      const finalNorm = mkAct(Hd)
      this.runRmsNorm(pass2, lastHidden, 'output_norm.weight', 1, Hd, finalNorm, tempBufs)
      this.runMatmul(pass2, 'output.weight', finalNorm, 1, this.logitsBuf, null, tempBufs, rotMemo)
      if (greedyOnly) {
        const uArg = this.mkUniform([this.arch.vocab, 0, 0, 0])
        tempBufs.push(uArg)
        this.dispatch(pass2, 'argmax', uArg, [this.logitsBuf, this.argmaxBuf], 1)
      }
      pass2.end()
    } else {
      const finalNorm = mkAct(Hd)
      this.runRmsNorm(pass, lastHidden, 'output_norm.weight', 1, Hd, finalNorm, tempBufs)
      this.runMatmul(pass, 'output.weight', finalNorm, 1, this.logitsBuf, null, tempBufs, rotMemo)
      if (greedyOnly) {
        const uArg = this.mkUniform([this.arch.vocab, 0, 0, 0])
        tempBufs.push(uArg)
        this.dispatch(pass, 'argmax', uArg, [this.logitsBuf, this.argmaxBuf], 1)
      }
      pass.end()
    }

    if (greedyOnly) {
      enc.copyBufferToBuffer(this.argmaxBuf, 0, this.argmaxReadBuf, 0, 4)
      this.device.queue.submit([enc.finish()])
      await this.argmaxReadBuf.mapAsync(GPUMapMode.READ)
      const greedyId = new Uint32Array(this.argmaxReadBuf.getMappedRange())[0]
      this.argmaxReadBuf.unmap()
      for (const b of tempBufs) b.destroy()
      return { greedyId }
    }

    enc.copyBufferToBuffer(this.logitsBuf, 0, this.logitsReadBuf, 0, this.arch.vocab * 4)
    this.device.queue.submit([enc.finish()])
    await this.logitsReadBuf.mapAsync(GPUMapMode.READ)
    const logits = new Float32Array(this.logitsReadBuf.getMappedRange().slice(0))
    this.logitsReadBuf.unmap()
    for (const b of tempBufs) b.destroy()
    return { logits }
  }

  /**
   * Generate text or token IDs from a prompt string or ChatMessage array.
   */
  async generate(
    promptOrMessages: string | ChatMessage[] | number[],
    options: GenerateOptions = {},
  ): Promise<{ text: string; tokenIds: number[]; stats: GenerateStats }> {
    let promptIds: number[]
    if (Array.isArray(promptOrMessages) && typeof promptOrMessages[0] === 'number') {
      promptIds = promptOrMessages as number[]
    } else if (typeof promptOrMessages === 'string') {
      if (!this.tokenizer) throw new Error('hachiue: GGUF header does not contain a tokenizer')
      promptIds = this.tokenizer.encode(promptOrMessages)
    } else {
      if (!this.tokenizer) throw new Error('hachiue: GGUF header does not contain a tokenizer')
      const formatted = this.tokenizer.formatChat(promptOrMessages as ChatMessage[])
      promptIds = this.tokenizer.encode(formatted)
    }

    if (promptIds.length === 0) {
      throw new Error('hachiue: prompt produced 0 tokens')
    }

    // Check if we can reuse prefix in KV cache
    let prefixMatch = 0
    while (
      prefixMatch < this.tokenHistory.length &&
      prefixMatch < promptIds.length - 1 &&
      this.tokenHistory[prefixMatch] === promptIds[prefixMatch]
    ) {
      prefixMatch++
    }
    if (prefixMatch < this.tokenHistory.length) {
      // If history diverged or this is a linear-attention hybrid model without state snapshots, reset
      if (this.arch.hybrid || prefixMatch === 0) {
        this.resetCache()
        prefixMatch = 0
      } else {
        this.cachePos = prefixMatch
        this.tokenHistory = this.tokenHistory.slice(0, prefixMatch)
      }
    }

    const toPrefill = promptIds.slice(prefixMatch)
    const maxNewTokens = options.maxNewTokens ?? 512
    const stopSet = new Set<number>([this.arch.eosTokenId, ...(options.stopTokens ?? [])])
    const isGreedy =
      (options.temperature ?? 0.6) <= 0 && (options.repetitionPenalty === undefined || options.repetitionPenalty === 1.0)

    const prefillStart = performance.now()
    let nextTokenId = 0

    for (let i = 0; i < toPrefill.length; i += PREFILL_CHUNK) {
      if (options.signal?.aborted) throw new DOMException('Generation aborted', 'AbortError')
      const seg = toPrefill.slice(i, i + PREFILL_CHUNK)
      const isLastSeg = i + PREFILL_CHUNK >= toPrefill.length
      const res = await this.forwardSegment(seg, isLastSeg, isGreedy)
      this.tokenHistory.push(...seg)
      if (isLastSeg) {
        if (isGreedy && res.greedyId !== undefined) {
          nextTokenId = res.greedyId
        } else if (res.logits) {
          nextTokenId = sampleLogits(res.logits, this.arch.vocab, this.tokenHistory, options)
        }
      }
    }
    const prefillMs = performance.now() - prefillStart

    const generatedIds: number[] = []
    const stream = this.tokenizer?.createDecoderStream(true)
    const decodeStart = performance.now()

    for (let step = 0; step < maxNewTokens; step++) {
      if (options.signal?.aborted) break
      if (stopSet.has(nextTokenId)) break

      generatedIds.push(nextTokenId)
      const delta = stream ? stream.push(nextTokenId) : ''
      options.onToken?.(nextTokenId, delta)

      if (step + 1 >= maxNewTokens) break
      if (this.cachePos + 1 > this.maxSeqLen) break

      const res = await this.forwardSegment([nextTokenId], true, isGreedy)
      this.tokenHistory.push(nextTokenId)
      if (isGreedy && res.greedyId !== undefined) {
        nextTokenId = res.greedyId
      } else if (res.logits) {
        nextTokenId = sampleLogits(res.logits, this.arch.vocab, this.tokenHistory, options)
      }
    }

    const tail = stream ? stream.flush() : ''
    if (tail) options.onToken?.(-1, tail)

    const decodeMs = Math.max(1, performance.now() - decodeStart)
    const text = this.tokenizer ? this.tokenizer.decode(generatedIds, true) : ''

    return {
      text,
      tokenIds: generatedIds,
      stats: {
        promptTokens: toPrefill.length,
        generatedTokens: generatedIds.length,
        prefillMs,
        decodeMs,
        prefillTokensPerSec: (toPrefill.length / Math.max(1, prefillMs)) * 1000,
        decodeTokensPerSec: (generatedIds.length / decodeMs) * 1000,
      },
    }
  }
}
