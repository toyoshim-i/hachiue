import { fetchGgufHeader, fp16ToFp32, getBlockSpec } from './gguf.js'
import { SHADERS } from './kernels.js'
import { sampleLogits } from './sampler.js'
import { prefetchModel, streamModelChunks } from './storage.js'
import { GgufTokenizer } from './tokenizer.js'
import {
  type BackendInfo,
  type BackendType,
  type ChatMessage,
  type EngineOptions,
  type GenerateOptions,
  type GenerateStats,
  GgmlType,
  type GgufTensorInfo,
  type ModelArch,
  type ParsedGgufHeader,
  type StorageOptions,
} from './types.js'
import { createWasmBackend, type WasmBackend } from './wasm-backend.js'

// WebGPU GPUBufferUsage numeric constants (safe to evaluate in Node.js before WebGPU exists)
const MAP_READ = 0x0001
const COPY_SRC = 0x0004
const COPY_DST = 0x0008
const UNIFORM = 0x0040
const STORAGE = 0x0080
const STORAGE_COPY = STORAGE | COPY_DST | COPY_SRC
const UNIFORM_COPY = UNIFORM | COPY_DST
const PREFILL_CHUNK = 8 // Max tokens per prefill segment to bound scratch memory on mobile

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

interface ScratchBuffers {
  curH: GPUBuffer
  nextH: GPUBuffer
  h2: GPUBuffer
  n1: GPUBuffer
  n2: GPUBuffer
  rotBuf: GPUBuffer
  permBuf: GPUBuffer
  qkvMixed: GPUBuffer
  qkvConv: GPUBuffer
  aProj: GPUBuffer
  bProj: GPUBuffer
  gbeta: GPUBuffer
  core: GPUBuffer
  zGate: GPUBuffer
  normed: GPUBuffer
  qgBuf: GPUBuffer
  qBuf: GPUBuffer
  gateBuf: GPUBuffer
  kBuf: GPUBuffer
  vBuf: GPUBuffer
  qNormed: GPUBuffer
  kNormed: GPUBuffer
  attOut: GPUBuffer
  mlpGate: GPUBuffer
  mlpUp: GPUBuffer
  swigluOut: GPUBuffer
  lastHidden: GPUBuffer
  finalNorm: GPUBuffer
}

/**
 * Create a progressive byte-streaming splitter that splits raw GGUF blocks
 * (`Q1_0`, `PQ2_0`, `PTQ1_0`) into planar `quants` and `scales16` GPU buffers
 * using a single shared staging buffer pair (`qStage`, `sStage`) so CPU RAM stays < 2 MiB.
 */
function createBlockSplitter(
  device: GPUDevice,
  type: GgmlType,
  chunks: QuantChunkBuffer[],
  K: number,
  qStage: Uint8Array<ArrayBuffer>,
  sStage: Uint8Array<ArrayBuffer>,
): { push: (bytes: Uint8Array) => number; finish: () => number } {
  const nb = K / 128
  const blockSize = type === GgmlType.Q1_0 ? 18 : type === GgmlType.PQ2_0 ? 34 : 28
  const quantBytesPerBlock = blockSize - 2

  let chunkIdx = 0
  let blockInChunk = 0
  let totalBlocksInCurChunk = chunks[0].rowCount * nb

  const maxStage = qStage.byteLength
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
      if (aligned > qStageLen) qStage.fill(0, qStageLen, aligned)
      device.queue.writeBuffer(c.quants, qGpuOffset, qStage, 0, aligned)
      qGpuOffset += qStageLen
      written += aligned
      qStageLen = 0
    }
    if (sStageLen > 0) {
      const aligned = (sStageLen + 3) & ~3
      if (aligned > sStageLen) sStage.fill(0, sStageLen, aligned)
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
    if ((blockInChunk & 1) === 0 && qStageLen + 1024 >= maxStage) {
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
    finish: (): number => {
      if (leftoverLen !== 0) {
        throw new Error(`hachiue: incomplete quantized block at end of tensor (${leftoverLen} leftover bytes)`)
      }
      return flushStaging()
    },
  }
}

function createRawChunkStreamer(
  device: GPUDevice,
  chunks: QuantChunkBuffer[],
  rowBytes: number,
  qStage: Uint8Array<ArrayBuffer>,
): { push: (bytes: Uint8Array) => number; finish: () => number } {
  let chunkIdx = 0
  let bytesInChunk = 0
  let totalBytesInCurChunk = chunks[0].rowCount * rowBytes
  let qStageLen = 0
  let qGpuOffset = 0
  const maxStage = qStage.byteLength

  const flushStaging = (): number => {
    if (qStageLen === 0) return 0
    const c = chunks[chunkIdx]
    const aligned = (qStageLen + 3) & ~3
    if (aligned > qStageLen) qStage.fill(0, qStageLen, aligned)
    device.queue.writeBuffer(c.quants, qGpuOffset, qStage, 0, aligned)
    qGpuOffset += qStageLen
    const written = aligned
    qStageLen = 0
    return written
  }

  return {
    push: (bytes: Uint8Array): number => {
      let flushed = 0
      let pos = 0
      while (pos < bytes.byteLength && chunkIdx < chunks.length) {
        const remInChunk = totalBytesInCurChunk - bytesInChunk
        const remInStage = maxStage - qStageLen
        // Keep stage flushes 4-byte aligned unless completing the chunk
        const maxAlignedStage = remInStage >= 4 ? remInStage & ~3 : remInStage
        const take = Math.min(bytes.byteLength - pos, remInChunk, maxAlignedStage)
        qStage.set(bytes.subarray(pos, pos + take), qStageLen)
        qStageLen += take
        bytesInChunk += take
        pos += take

        if (bytesInChunk === totalBytesInCurChunk) {
          flushed += flushStaging()
          chunkIdx++
          bytesInChunk = 0
          qGpuOffset = 0
          if (chunkIdx < chunks.length) {
            totalBytesInCurChunk = chunks[chunkIdx].rowCount * rowBytes
          }
        } else if (qStageLen + 4096 >= maxStage && (qStageLen & 3) === 0) {
          flushed += flushStaging()
        }
      }
      return flushed
    },
    finish: (): number => {
      return flushStaging()
    },
  }
}

function buildRopeTables(
  arch: ModelArch,
  maxSeqLen: number,
  useLocalSwa = false,
): { cos: Float32Array; sin: Float32Array } {
  const rotaryDim = arch.rope.rotaryDim
  const halfDim = rotaryDim >> 1
  const cos = new Float32Array(maxSeqLen * halfDim)
  const sin = new Float32Array(maxSeqLen * halfDim)
  const theta = useLocalSwa ? (arch.rope.localTheta ?? 10000) : arch.rope.theta
  const factor = !useLocalSwa && arch.rope.factor && arch.rope.factor > 1.0 ? arch.rope.factor : 1.0
  const origMaxPos = arch.rope.originalMaxPositions ?? 4096
  const betaFast = arch.rope.yarnBetaFast ?? 32.0
  const betaSlow = arch.rope.yarnBetaSlow ?? 1.0
  const mscale = factor > 1.0 ? 0.1 * Math.log(factor) + 1.0 : 1.0

  // Precompute per-dimension inverse frequencies (supporting YaRN when factor > 1.0)
  const invFreq = new Float64Array(halfDim)
  let low = 0
  let high = halfDim - 1
  if (factor > 1.0) {
    const findCorrDim = (numRotations: number): number =>
      (rotaryDim * Math.log(origMaxPos / (numRotations * 2.0 * Math.PI))) / (2.0 * Math.log(theta))
    low = Math.max(0, findCorrDim(betaFast))
    high = Math.min(halfDim - 1, findCorrDim(betaSlow))
    if (low === high) high += 0.001
  }

  for (let i = 0; i < halfDim; i++) {
    const freq = 1.0 / Math.pow(theta, (2 * i) / rotaryDim)
    if (factor > 1.0) {
      const freqExtrap = freq
      const freqInterp = freq / factor
      const linear = (i - low) / (high - low)
      const ramp = Math.min(1.0, Math.max(0.0, linear))
      const smooth = 1.0 - ramp
      invFreq[i] = freqInterp * (1.0 - smooth) + freqExtrap * smooth
    } else {
      invFreq[i] = freq
    }
  }

  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const angle = pos * invFreq[i]
      cos[pos * halfDim + i] = Math.cos(angle) * mscale
      sin[pos * halfDim + i] = Math.sin(angle) * mscale
    }
  }
  return { cos, sin }
}

export class HachiueEngine {
  readonly header: ParsedGgufHeader
  readonly arch: ModelArch
  readonly tokenizer: GgufTokenizer | null
  readonly backend: BackendType
  readonly isWorker: boolean
  readonly backendInfo: BackendInfo
  readonly device: GPUDevice
  readonly maxSeqLen: number

  private readonly wasmBackend: WasmBackend | null
  private readonly pipelines: Map<string, GPUComputePipeline>
  private readonly weights: Map<string, GpuTensor>
  private readonly hadamardSigns: Map<number, GPUBuffer>
  private readonly cosBuf: GPUBuffer
  private readonly sinBuf: GPUBuffer
  private readonly swaCosBuf: GPUBuffer | null
  private readonly swaSinBuf: GPUBuffer | null
  private readonly dummyBuf: GPUBuffer
  private readonly logitsBuf: GPUBuffer
  private readonly logitsReadBuf: GPUBuffer
  private readonly logitsCpu: Float32Array
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
  private readonly scratch: ScratchBuffers
  private readonly uniformPool: GPUBuffer[] = []
  private uniformCursor = 0
  private readonly uniformScratch = new ArrayBuffer(32)
  private readonly uniformView = new DataView(this.uniformScratch)
  private ssmParity = 0
  private cachePos = 0
  private tokenHistory: number[] = []
  private lastMessages: ChatMessage[] = []
  private lastTurnTokens: number[] = []
  private disposed = false

  private constructor(
    header: ParsedGgufHeader,
    device: GPUDevice | null,
    maxSeqLen: number,
    pipelines: Map<string, GPUComputePipeline>,
    weights: Map<string, GpuTensor>,
    hadamardSigns: Map<number, GPUBuffer>,
    cosBuf: GPUBuffer | null,
    sinBuf: GPUBuffer | null,
    swaCosBuf: GPUBuffer | null,
    swaSinBuf: GPUBuffer | null,
    wasmBackend: WasmBackend | null = null,
    backendInfo?: BackendInfo,
  ) {
    this.header = header
    this.arch = header.arch
    this.tokenizer = header.tokenizer ? new GgufTokenizer(header.tokenizer) : null
    this.backend = wasmBackend ? 'wasm' : 'webgpu'
    this.isWorker = wasmBackend ? wasmBackend.isWorker : false
    this.backendInfo = backendInfo ?? {
      backend: this.backend,
      isWorker: this.isWorker,
      label: this.backend === 'webgpu' ? 'WebGPU' : this.isWorker ? 'Wasm SIMD (Worker)' : 'Wasm SIMD',
    }
    this.device = device as GPUDevice
    this.maxSeqLen = maxSeqLen
    this.wasmBackend = wasmBackend
    this.pipelines = pipelines
    this.weights = weights
    this.hadamardSigns = hadamardSigns
    this.cosBuf = cosBuf as GPUBuffer
    this.sinBuf = sinBuf as GPUBuffer
    this.swaCosBuf = swaCosBuf
    this.swaSinBuf = swaSinBuf

    if (wasmBackend || !device) {
      this.dummyBuf = null as unknown as GPUBuffer
      this.logitsBuf = null as unknown as GPUBuffer
      this.logitsReadBuf = null as unknown as GPUBuffer
      this.logitsCpu = new Float32Array(0)
      this.argmaxBuf = null as unknown as GPUBuffer
      this.argmaxReadBuf = null as unknown as GPUBuffer
      this.tokensBuf = null as unknown as GPUBuffer
      this.scratch = null as unknown as ScratchBuffers
      this.kCacheQ = new Map()
      this.kCacheS = new Map()
      this.vCacheQ = new Map()
      this.vCacheS = new Map()
      this.ssmConvState = new Map()
      this.ssmRecState = new Map()
      return
    }

    const mkBuf = (size: number, usage = STORAGE_COPY): GPUBuffer =>
      device.createBuffer({ size: Math.max(16, (size + 3) & ~3), usage })
    const mkAct = (elems: number): GPUBuffer => mkBuf(elems * 4)

    this.dummyBuf = mkBuf(64)
    this.logitsBuf = mkBuf(this.arch.vocab * 4)
    this.logitsReadBuf = device.createBuffer({
      size: this.arch.vocab * 4,
      usage: MAP_READ | COPY_DST,
    })
    this.logitsCpu = new Float32Array(this.arch.vocab)
    this.argmaxBuf = mkBuf(16)
    this.argmaxReadBuf = device.createBuffer({
      size: 16,
      usage: MAP_READ | COPY_DST,
    })
    this.tokensBuf = mkBuf(PREFILL_CHUNK * 4)

    const Hd = this.arch.hidden
    const F = this.arch.intermediate
    const H = this.arch.heads
    const KV = this.arch.kvHeads
    const Dh = this.arch.headDim
    const qDim = H * Dh
    const kvDim = KV * Dh
    const hy = this.arch.hybrid
    const NV = hy ? hy.linearValueHeads : 1
    const DKV = hy ? hy.linearHeadDim : 1
    const valDim = hy ? NV * DKV : Hd
    const convDim = hy ? (hy.linearKeyHeads * 2 + NV) * DKV : Math.max(Hd, qDim + 2 * kvDim)
    const maxRotDim = Math.max(Hd, F, qDim, valDim)

    this.scratch = {
      curH: mkAct(PREFILL_CHUNK * Hd),
      nextH: mkAct(PREFILL_CHUNK * Hd),
      h2: mkAct(PREFILL_CHUNK * Hd),
      n1: mkAct(PREFILL_CHUNK * Hd),
      n2: mkAct(PREFILL_CHUNK * Hd),
      rotBuf: mkAct(PREFILL_CHUNK * maxRotDim),
      permBuf: mkAct(PREFILL_CHUNK * valDim),
      qkvMixed: mkAct(PREFILL_CHUNK * convDim),
      qkvConv: mkAct(PREFILL_CHUNK * convDim),
      aProj: mkAct(PREFILL_CHUNK * NV),
      bProj: mkAct(PREFILL_CHUNK * NV),
      gbeta: mkAct(2 * PREFILL_CHUNK * NV),
      core: mkAct(PREFILL_CHUNK * valDim),
      zGate: mkAct(PREFILL_CHUNK * valDim),
      normed: mkAct(PREFILL_CHUNK * valDim),
      qgBuf: mkAct(PREFILL_CHUNK * qDim * (this.arch.archName === 'qwen35' ? 2 : 1)),
      qBuf: mkAct(PREFILL_CHUNK * qDim),
      gateBuf: mkAct(PREFILL_CHUNK * qDim),
      kBuf: mkAct(PREFILL_CHUNK * kvDim),
      vBuf: mkAct(PREFILL_CHUNK * kvDim),
      qNormed: mkAct(PREFILL_CHUNK * qDim),
      kNormed: mkAct(PREFILL_CHUNK * kvDim),
      attOut: mkAct(PREFILL_CHUNK * qDim),
      mlpGate: mkAct(PREFILL_CHUNK * F * (this.arch.archName === 'plamo3' ? 2 : 1)),
      mlpUp: mkAct(PREFILL_CHUNK * F),
      swigluOut: mkAct(PREFILL_CHUNK * F),
      lastHidden: mkAct(Hd),
      finalNorm: mkAct(Hd),
    }

    this.kCacheQ = new Map()
    this.kCacheS = new Map()
    this.vCacheQ = new Map()
    this.vCacheS = new Map()
    this.ssmConvState = new Map()
    this.ssmRecState = new Map()

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
   * Prefetch a GGUF model into the local OPFS / IndexedDB chunk cache in the background
   * without initializing WebGPU or affecting any currently active HachiueEngine instance.
   */
  static async prefetch(url: string, options: StorageOptions = {}): Promise<ParsedGgufHeader> {
    return prefetchModel(url, options)
  }

  /**
   * Create and initialize a HachiueEngine from a GGUF URL.
   * By default (`backend: 'auto'`), uses WebGPU when available and automatically falls back
   * to the Wasm SIMD Web Worker backend when WebGPU is unsupported or adapter acquisition fails.
   */
  static async create(options: EngineOptions): Promise<HachiueEngine> {
    const maxSeqLen = options.maxSeqLen ?? 4096
    const header = await fetchGgufHeader(options.url, options)
    const arch = header.arch
    const requestedBackend = options.backend ?? (options.device ? 'webgpu' : 'auto')

    const { cos, sin } = buildRopeTables(arch, maxSeqLen, false)
    let sCos: Float32Array | null = null
    let sSin: Float32Array | null = null
    if (arch.swa) {
      const swaTables = buildRopeTables(arch, maxSeqLen, true)
      sCos = swaTables.cos
      sSin = swaTables.sin
    }

    const initWasmFallback = async (fallbackReason?: string): Promise<HachiueEngine> => {
      const wasmBackend = await createWasmBackend(header, options, cos, sin, sCos, sSin)
      const info: BackendInfo = {
        backend: 'wasm',
        isWorker: wasmBackend.isWorker,
        label: wasmBackend.isWorker ? 'Wasm SIMD (Worker)' : 'Wasm SIMD',
        ...(fallbackReason ? { fallbackReason } : {}),
      }
      return new HachiueEngine(
        header,
        null,
        maxSeqLen,
        new Map(),
        new Map(),
        new Map(),
        null,
        null,
        null,
        null,
        wasmBackend,
        info,
      )
    }

    if (requestedBackend === 'wasm') {
      return initWasmFallback()
    }

    let device = options.device
    let adapterMode: BackendInfo['adapterMode'] = device ? 'custom-device' : undefined
    let pipelines: Map<string, GPUComputePipeline> | null = null

    try {
      if (!device) {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
          throw new Error('hachiue: WebGPU is not supported in this browser')
        }
        let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
        if (adapter) {
          adapterMode = 'high-performance'
        } else {
          adapter = await navigator.gpu.requestAdapter()
          if (adapter) {
            adapterMode = 'default'
          } else {
            adapter = await navigator.gpu.requestAdapter({
              featureLevel: 'compatibility',
            } as GPURequestAdapterOptions)
            if (adapter) adapterMode = 'compatibility'
          }
        }
        if (!adapter) {
          throw new Error('hachiue: Failed to acquire WebGPU adapter')
        }
        const maxBind = Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024)
        const maxBuf = Math.min(adapter.limits.maxBufferSize, 512 * 1024 * 1024)
        const maxStoragePerStage = adapter.limits.maxStorageBuffersPerShaderStage
        device = await adapter.requestDevice({
          requiredLimits: {
            maxStorageBufferBindingSize: maxBind,
            maxBufferSize: maxBuf,
            ...(maxStoragePerStage ? { maxStorageBuffersPerShaderStage: maxStoragePerStage } : {}),
          },
        })
      }

      options.onProgress?.({
        phase: 'pipelines',
        message: 'Compiling WebGPU shaders...',
        backend: 'webgpu',
        isWorker: false,
      })
      pipelines = new Map<string, GPUComputePipeline>()
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
            pipelines!.set(name, pipe)
          } catch (err) {
            throw new Error(`hachiue: pipeline compile failed in ${name}: ${(err as Error).message || err}`)
          }
        }),
      )
    } catch (err) {
      if (device && !options.device) {
        try {
          device.destroy()
        } catch {
          // ignore
        }
      }
      if (requestedBackend === 'webgpu') {
        throw err
      }
      const reason = err instanceof Error ? err.message : String(err)
      return initWasmFallback(reason)
    }

    device.lost.then((info) => {
      options.onDeviceLost?.({ reason: String(info.reason ?? 'unknown'), message: info.message })
    })

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

    // Upload RoPE tables
    const cosBuf = device.createBuffer({ size: cos.byteLength, usage: STORAGE_COPY })
    const sinBuf = device.createBuffer({ size: sin.byteLength, usage: STORAGE_COPY })
    device.queue.writeBuffer(cosBuf, 0, cos.buffer as ArrayBuffer, cos.byteOffset, cos.byteLength)
    device.queue.writeBuffer(sinBuf, 0, sin.buffer as ArrayBuffer, sin.byteOffset, sin.byteLength)

    let swaCosBuf: GPUBuffer | null = null
    let swaSinBuf: GPUBuffer | null = null
    if (sCos && sSin) {
      swaCosBuf = device.createBuffer({ size: sCos.byteLength, usage: STORAGE_COPY })
      swaSinBuf = device.createBuffer({ size: sSin.byteLength, usage: STORAGE_COPY })
      device.queue.writeBuffer(swaCosBuf, 0, sCos.buffer as ArrayBuffer, sCos.byteOffset, sCos.byteLength)
      device.queue.writeBuffer(swaSinBuf, 0, sSin.buffer as ArrayBuffer, sSin.byteOffset, sSin.byteLength)
    }

    // Allocate GPU buffers for all tensors and register stream consumers
    const maxChunkBindBytes = Math.min(device.limits.maxStorageBufferBindingSize, 64 * 1024 * 1024)
    const weights = new Map<string, GpuTensor>()
    const dummyScalesBuf = device.createBuffer({ size: 16, usage: STORAGE_COPY })

    // Single shared staging buffers across all tensors (1 MiB quants + 128 KiB scales)
    const sharedQStage = new Uint8Array(new ArrayBuffer(1024 * 1024))
    const sharedSStage = new Uint8Array(new ArrayBuffer(128 * 1024))

    interface StreamRoute {
      absStart: number
      absEnd: number
      push: (bytes: Uint8Array) => number
      finish: () => number
    }
    const routes: Array<StreamRoute | null> = []

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
            finish: () => 0,
          })
        } else {
          let rawF16: Uint8Array | null = null
          let wOff = 0
          routes.push({
            absStart: t.absOffset,
            absEnd: t.absOffset + t.byteLength,
            push: (b: Uint8Array): number => {
              if (!rawF16) rawF16 = new Uint8Array(t.byteLength)
              rawF16.set(b, wOff)
              wOff += b.byteLength
              return 0
            },
            finish: (): number => {
              if (!rawF16) return 0
              const dv = new DataView(rawF16.buffer, rawF16.byteOffset, rawF16.byteLength)
              const f32Arr = new Float32Array(numElems)
              for (let i = 0; i < numElems; i++) f32Arr[i] = fp16ToFp32(dv.getUint16(i * 2, true))
              device!.queue.writeBuffer(f32Buf, 0, f32Arr)
              rawF16 = null
              return f32Arr.byteLength
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

        const splitter = createBlockSplitter(device, t.type, chunks, K, sharedQStage, sharedSStage)
        routes.push({
          absStart: t.absOffset,
          absEnd: t.absOffset + t.byteLength,
          push: splitter.push,
          finish: splitter.finish,
        })
      } else if (
        t.type === GgmlType.Q4_0 ||
        t.type === GgmlType.Q4_1 ||
        t.type === GgmlType.Q4_K ||
        t.type === GgmlType.Q6_K
      ) {
        const [blkElems, blkBytes] = getBlockSpec(t.type)
        const nb = K / blkElems
        const rowBytes = nb * blkBytes
        const maxRowsPerChunk = Math.max(1, Math.floor(maxChunkBindBytes / rowBytes))

        const chunks: QuantChunkBuffer[] = []
        for (let r = 0; r < N; r += maxRowsPerChunk) {
          const rowCount = Math.min(maxRowsPerChunk, N - r)
          const qBytes = Math.max(16, (rowCount * rowBytes + 3) & ~3)
          chunks.push({
            rowStart: r,
            rowCount,
            quants: device.createBuffer({ size: qBytes, usage: STORAGE_COPY }),
            scales16: dummyScalesBuf,
          })
        }

        weights.set(t.name, {
          type: t.type,
          N,
          K,
          nb,
          hadamardFolded,
          chunks,
        })

        const streamer = createRawChunkStreamer(device, chunks, rowBytes, sharedQStage)
        routes.push({
          absStart: t.absOffset,
          absEnd: t.absOffset + t.byteLength,
          push: streamer.push,
          finish: streamer.finish,
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

    // Stream chunks from OPFS / IndexedDB / HTTP Range with strict per-chunk GPU sync & GC yield
    let routeIdx = 0
    const firstTensorOffset = routes.length > 0 && routes[0] ? routes[0].absStart : header.dataStartOffset
    await streamModelChunks(
      options.url,
      firstTensorOffset,
      header.totalFileSize,
      async (chunk: Uint8Array, chunkFileOffset: number) => {
        const chunkEnd = chunkFileOffset + chunk.byteLength
        while (routeIdx < routes.length) {
          const r = routes[routeIdx]
          if (!r || r.absEnd <= chunkFileOffset) {
            routes[routeIdx] = null
            routeIdx++
            continue
          }
          if (r.absStart >= chunkEnd) {
            break
          }
          const sliceStart = Math.max(0, r.absStart - chunkFileOffset)
          const sliceEnd = Math.min(chunk.byteLength, r.absEnd - chunkFileOffset)
          if (sliceEnd > sliceStart) {
            r.push(chunk.subarray(sliceStart, sliceEnd))
          }
          if (chunkFileOffset + sliceEnd === r.absEnd) {
            r.finish()
            routes[routeIdx] = null
            routeIdx++
          } else {
            break
          }
        }
        // Wait for GPU queue to retire staging copies and yield a macrotask turn so mobile browser GC
        // frees the 16 MiB chunk ArrayBuffer immediately (especially critical when reading fast from local cache).
        await device!.queue.onSubmittedWorkDone()
        await new Promise((resolve) => setTimeout(resolve, 0))
      },
      options,
    )

    await device.queue.onSubmittedWorkDone()
    options.onProgress?.({
      phase: 'ready',
      message: 'Model loaded into WebGPU memory',
      backend: 'webgpu',
      isWorker: false,
    })

    const info: BackendInfo = {
      backend: 'webgpu',
      isWorker: false,
      label: adapterMode === 'compatibility' ? 'WebGPU (Compat)' : 'WebGPU',
      adapterMode,
    }

    return new HachiueEngine(
      header,
      device,
      maxSeqLen,
      pipelines,
      weights,
      hadamardSigns,
      cosBuf,
      sinBuf,
      swaCosBuf,
      swaSinBuf,
      null,
      info,
    )
  }

  /**
   * Reset conversation KV cache and recurrent states.
   */
  resetCache(): void {
    this.cachePos = 0
    this.ssmParity = 0
    this.tokenHistory = []
    this.lastMessages = []
    this.lastTurnTokens = []
    if (this.wasmBackend) {
      this.wasmBackend.resetCache()
    }
  }

  /**
   * Destroy all GPU or Wasm Worker resources.
   */
  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    if (this.wasmBackend) {
      this.wasmBackend.dispose()
    } else if (this.device) {
      this.device.destroy()
    }
  }

  /**
   * Wait for all queued GPU buffer writes or Wasm Worker operations to finish.
   */
  async flushDeviceQueue(): Promise<void> {
    if (this.wasmBackend) {
      await this.wasmBackend.flush()
      return
    }
    await this.device.queue.onSubmittedWorkDone()
  }

  /**
   * Inspect a loaded weight tensor's dimensions and byte width per row in GPU/Wasm memory.
   */
  getTensorRowSpec(name: string): { type: GgmlType; rows: number; cols: number; rowBytes: number } {
    if (this.wasmBackend) {
      return this.wasmBackend.getTensorRowSpec(name)
    }
    const tensor = this.weights.get(name)
    if (!tensor) {
      throw new Error(`hachiue: tensor "${name}" not found`)
    }
    if (tensor.f32Buf) {
      return { type: tensor.type, rows: tensor.N, cols: tensor.K, rowBytes: tensor.K * 4 }
    }
    if (tensor.type === GgmlType.Q1_0 || tensor.type === GgmlType.PQ2_0 || tensor.type === GgmlType.PTQ1_0) {
      const qBytesPerBlock = tensor.type === GgmlType.Q1_0 ? 16 : tensor.type === GgmlType.PQ2_0 ? 32 : 26
      return { type: tensor.type, rows: tensor.N, cols: tensor.K, rowBytes: tensor.nb * qBytesPerBlock }
    }
    const [, blkBytes] = getBlockSpec(tensor.type)
    return { type: tensor.type, rows: tensor.N, cols: tensor.K, rowBytes: tensor.nb * blkBytes }
  }

  private resolveTensorRowBuffer(name: string, tensor: GpuTensor, rowIndex: number): { buf: GPUBuffer; localRow: number } {
    if (rowIndex < 0 || rowIndex >= tensor.N) {
      throw new Error(`hachiue: row index ${rowIndex} out of bounds for "${name}" (rows=${tensor.N})`)
    }
    if (tensor.f32Buf) {
      return { buf: tensor.f32Buf, localRow: rowIndex }
    }
    if (tensor.chunks) {
      for (const c of tensor.chunks) {
        if (rowIndex >= c.rowStart && rowIndex < c.rowStart + c.rowCount) {
          return { buf: c.quants, localRow: rowIndex - c.rowStart }
        }
      }
    }
    throw new Error(`hachiue: failed to resolve GPU chunk for "${name}" row ${rowIndex}`)
  }

  /**
   * Read one or more rows from a GPU/Wasm weight tensor into a contiguous `Uint8Array(rowIndices.length * rowBytes)`.
   * Automatically resolves 64 MiB storage buffer chunk boundaries.
   */
  async readTensorRows(name: string, rowIndices: ArrayLike<number>): Promise<Uint8Array> {
    if (this.wasmBackend) {
      return this.wasmBackend.readTensorRows(name, rowIndices)
    }
    const tensor = this.weights.get(name)
    if (!tensor) {
      throw new Error(`hachiue: tensor "${name}" not found`)
    }
    const { rowBytes } = this.getTensorRowSpec(name)
    const count = rowIndices.length
    if (count === 0) return new Uint8Array(0)

    const totalBytes = count * rowBytes
    const staging = this.device.createBuffer({
      size: totalBytes,
      usage: MAP_READ | COPY_DST,
    })

    const enc = this.device.createCommandEncoder()
    for (let i = 0; i < count; i++) {
      const { buf, localRow } = this.resolveTensorRowBuffer(name, tensor, rowIndices[i])
      enc.copyBufferToBuffer(buf, localRow * rowBytes, staging, i * rowBytes, rowBytes)
    }
    this.device.queue.submit([enc.finish()])
    await staging.mapAsync(MAP_READ)
    const out = new Uint8Array(staging.getMappedRange().slice(0))
    staging.unmap()
    staging.destroy()
    return out
  }

  /**
   * Overwrite one or more rows of a GPU/Wasm weight tensor in-place.
   * If `data.byteLength === rowBytes` and `rowIndices.length > 1`, the single row is broadcast to all `rowIndices`.
   */
  writeTensorRows(name: string, rowIndices: ArrayLike<number>, data: Uint8Array): void {
    if (this.wasmBackend) {
      this.wasmBackend.writeTensorRows(name, rowIndices, data)
      return
    }
    const tensor = this.weights.get(name)
    if (!tensor) {
      throw new Error(`hachiue: tensor "${name}" not found`)
    }
    const { rowBytes } = this.getTensorRowSpec(name)
    const count = rowIndices.length
    if (count === 0) return

    const broadcast = data.byteLength === rowBytes && count > 1
    if (!broadcast && data.byteLength !== count * rowBytes) {
      throw new Error(
        `hachiue: writeTensorRows("${name}") expected ${count * rowBytes} (or ${rowBytes}) bytes, got ${data.byteLength}`,
      )
    }

    for (let i = 0; i < count; i++) {
      const { buf, localRow } = this.resolveTensorRowBuffer(name, tensor, rowIndices[i])
      const srcOffset = broadcast ? data.byteOffset : data.byteOffset + i * rowBytes
      this.device.queue.writeBuffer(buf, localRow * rowBytes, data.buffer as ArrayBuffer, srcOffset, rowBytes)
    }
  }

  private mkUniform(words: Array<number>, floatIndices: Set<number> = new Set()): GPUBuffer {
    for (let i = 0; i < 8; i++) {
      const val = i < words.length ? words[i] : 0
      if (floatIndices.has(i)) this.uniformView.setFloat32(i * 4, val, true)
      else this.uniformView.setUint32(i * 4, val >>> 0, true)
    }
    const idx = this.uniformCursor++
    let buf = this.uniformPool[idx]
    if (!buf) {
      buf = this.device.createBuffer({ size: 32, usage: UNIFORM_COPY })
      this.uniformPool.push(buf)
    }
    this.device.queue.writeBuffer(buf, 0, this.uniformScratch)
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
  ): void {
    if (!this.arch.hadamard) return
    const signBuf = this.hadamardSigns.get(width)
    const hasSigns = signBuf ? 1 : 0
    const u = this.mkUniform([rows, width, mode, hasSigns])
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
  ): void {
    const w = this.weights.get(weightName)
    if (!w?.f32Buf) throw new Error(`hachiue: missing norm tensor ${weightName}`)
    // Note: GGUF converter already bakes (1 + weight) into Qwen3.5 RMSNorm tensors.
    const u = this.mkUniform([rows, dim, this.arch.rmsEps, 0], new Set([2]))
    this.dispatch(pass, 'rmsnorm', u, [xBuf, w.f32Buf, outBuf], rows)
  }

  private runMatmul(
    pass: GPUComputePassEncoder,
    weightName: string,
    xBuf: GPUBuffer,
    M: number,
    outBuf: GPUBuffer,
    residBuf: GPUBuffer | null,
    rotMemo: Map<GPUBuffer, GPUBuffer>,
  ): void {
    const w = this.weights.get(weightName)
    if (!w) throw new Error(`hachiue: missing weight ${weightName}`)

    let inputBuf = xBuf
    if (w.hadamardFolded && this.arch.hadamard) {
      let rot = rotMemo.get(xBuf)
      if (!rot) {
        let srcForRot = xBuf
        if (
          this.arch.hadamard.gdnVGrouped &&
          weightName.includes('.ssm_out.') &&
          this.arch.hybrid &&
          this.arch.hybrid.linearValueHeads > this.arch.hybrid.linearKeyHeads
        ) {
          const hd = this.arch.hybrid.linearHeadDim
          const nk = this.arch.hybrid.linearKeyHeads
          const rep = this.arch.hybrid.linearValueHeads / nk
          const uPerm = this.mkUniform([M, hd, nk, rep])
          this.dispatch(pass, 'permute_gdn', uPerm, [xBuf, this.scratch.permBuf], Math.ceil((M * w.K) / 256))
          srcForRot = this.scratch.permBuf
        }
        rot = this.scratch.rotBuf
        this.runHadamard(pass, srcForRot, rot, M, w.K, 0)
        rotMemo.set(xBuf, rot)
      }
      inputBuf = rot
    }

    if (w.type === GgmlType.F32 && w.f32Buf) {
      const groups = Math.ceil(w.N / 4)
      const gx = Math.min(groups, 65535)
      const gy = Math.ceil(groups / gx)
      const u = this.mkUniform([M, w.N, w.K, gx])
      this.dispatch(pass, 'gemv_f32', u, [inputBuf, w.f32Buf, outBuf], gx, gy, M)
      return
    }

    if (!w.chunks) throw new Error(`hachiue: missing quantized chunks for ${weightName}`)
    const shader =
      w.type === GgmlType.Q1_0
        ? 'gemv_q1_0'
        : w.type === GgmlType.PQ2_0
          ? 'gemv_pq2_0'
          : w.type === GgmlType.PTQ1_0
            ? 'gemv_ptq1_0'
            : w.type === GgmlType.Q4_0
              ? 'gemv_q4_0'
              : w.type === GgmlType.Q4_1
                ? 'gemv_q4_1'
                : w.type === GgmlType.Q4_K
                  ? 'gemv_q4_k'
                  : 'gemv_q6_k'

    for (const chunk of w.chunks) {
      const groups = Math.ceil(chunk.rowCount / 4)
      const gx = Math.min(groups, 65535)
      const gy = Math.ceil(groups / gx)
      const u = this.mkUniform([M, chunk.rowCount, w.K, w.nb, gx, residBuf ? 1 : 0, chunk.rowStart, w.N])
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
   * Reuses pre-allocated scratch activation and uniform buffers so zero GPU buffers are allocated per step.
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

    if (this.wasmBackend) {
      const posBase = this.cachePos
      this.cachePos += S
      return this.wasmBackend.forwardSegment(tokenIds, computeLogits, greedyOnly, posBase)
    }

    const H = this.arch.heads
    const KV = this.arch.kvHeads
    const Dh = this.arch.headDim
    const Hd = this.arch.hidden
    const F = this.arch.intermediate
    const rotDim = this.arch.rope.rotaryDim
    const sc = this.scratch

    this.uniformCursor = 0
    const rotMemo = new Map<GPUBuffer, GPUBuffer>()

    const tokArr = new Uint32Array(tokenIds)
    this.device.queue.writeBuffer(this.tokensBuf, 0, tokArr)

    const enc = this.device.createCommandEncoder()
    const pass = enc.beginComputePass()

    // 1. Embedding Lookup
    const embdW = this.weights.get('token_embd.weight')!
    const embMode =
      embdW.type === GgmlType.Q1_0
        ? 0
        : embdW.type === GgmlType.PQ2_0
          ? 1
          : embdW.type === GgmlType.PTQ1_0
            ? 2
            : embdW.type === GgmlType.Q4_0
              ? 3
              : embdW.type === GgmlType.Q4_1
                ? 4
                : embdW.type === GgmlType.Q4_K
                  ? 5
                  : 6
    let curH = sc.curH
    let nextH = sc.nextH
    for (const embChunk of embdW.chunks!) {
      const uEmb = this.mkUniform([Hd, embMode, S, embChunk.rowStart, embChunk.rowCount, 0, 0, 0])
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
      this.runHadamard(pass, curH, nextH, S, Hd, 1)
      ;[curH, nextH] = [nextH, curH]
    }

    // 2. Transformer Layers
    for (let li = 0; li < this.arch.layers; li++) {
      this.runRmsNorm(pass, curH, `blk.${li}.attn_norm.weight`, S, Hd, sc.n1)
      rotMemo.clear()

      if (this.arch.archName === 'plamo3') {
        const qDim = H * Dh
        const kvDim = KV * Dh
        this.runMatmul(pass, `blk.${li}.attn_qkv.weight`, sc.n1, S, sc.qkvMixed, null, rotMemo)
        const uSplitQkv = this.mkUniform([S, qDim, kvDim, 0])
        this.dispatch(
          pass,
          'split_qkv',
          uSplitQkv,
          [sc.qkvMixed, sc.qBuf, sc.kBuf, sc.vBuf],
          Math.ceil((S * (qDim + 2 * kvDim)) / 256),
        )

        // Per-head Q/K RMSNorm
        this.runRmsNorm(pass, sc.qBuf, `blk.${li}.attn_q_norm.weight`, S * H, Dh, sc.qNormed)
        this.runRmsNorm(pass, sc.kBuf, `blk.${li}.attn_k_norm.weight`, S * KV, Dh, sc.kNormed)

        // Select RoPE tables and window size depending on whether this is an SWA layer or full layer
        const isSwa = Boolean(this.arch.swa && this.arch.swa.layerTypes[li] === 'swa')
        const layerCosBuf = isSwa && this.swaCosBuf ? this.swaCosBuf : this.cosBuf
        const layerSinBuf = isSwa && this.swaSinBuf ? this.swaSinBuf : this.sinBuf
        const windowSize = isSwa && this.arch.swa ? this.arch.swa.windowSize : 0

        const uRopeQ = this.mkUniform([S, H, Dh, rotDim, this.cachePos, 0, 0, 0])
        const uRopeK = this.mkUniform([S, KV, Dh, rotDim, this.cachePos, 0, 0, 0])
        this.dispatch(
          pass,
          'rope',
          uRopeQ,
          [layerCosBuf, layerSinBuf, sc.qNormed],
          Math.ceil((S * H * (rotDim >> 1)) / 64),
        )
        this.dispatch(
          pass,
          'rope',
          uRopeK,
          [layerCosBuf, layerSinBuf, sc.kNormed],
          Math.ceil((S * KV * (rotDim >> 1)) / 64),
        )

        // Append K and V to Q8_0 KV Cache
        const uKv = this.mkUniform([S, KV, Dh, this.cachePos])
        const kvBlocks = S * KV * (Dh >> 5)
        const kQ = this.kCacheQ.get(li)!
        const kS = this.kCacheS.get(li)!
        const vQ = this.vCacheQ.get(li)!
        const vS = this.vCacheS.get(li)!
        this.dispatch(pass, 'kv_append_q8', uKv, [sc.kNormed, kQ, kS], kvBlocks)
        this.dispatch(pass, 'kv_append_q8', uKv, [sc.vBuf, vQ, vS], kvBlocks)

        // Online GQA Attention with optional Sliding Window
        const uAtt = this.mkUniform([S, H, KV, Dh, this.cachePos, windowSize, 0, 0])
        this.dispatch(pass, 'attention_q8', uAtt, [sc.qNormed, kQ, kS, vQ, vS, sc.attOut], S * H)

        // Attention output projection -> post_attention_norm -> residual add
        rotMemo.clear()
        this.runMatmul(pass, `blk.${li}.attn_output.weight`, sc.attOut, S, sc.n1, null, rotMemo)
        this.runRmsNorm(pass, sc.n1, `blk.${li}.post_attention_norm`, S, Hd, sc.n2)
        const uAdd = this.mkUniform([S * Hd, 0, 0, 0])
        this.dispatch(pass, 'add_inplace', uAdd, [curH, sc.n2, sc.h2], Math.ceil((S * Hd) / 256))

        // FFN Norm -> Fused Gate+Up -> SwiGLU -> Down -> post_ffw_norm -> residual add
        this.runRmsNorm(pass, sc.h2, `blk.${li}.ffn_norm.weight`, S, Hd, sc.n1)
        rotMemo.clear()
        this.runMatmul(pass, `blk.${li}.ffn_up.weight`, sc.n1, S, sc.mlpGate, null, rotMemo)
        const uSwiFused = this.mkUniform([S, F, 0, 0])
        this.dispatch(pass, 'swiglu_fused', uSwiFused, [sc.mlpGate, sc.swigluOut], Math.ceil((S * F) / 256))

        rotMemo.clear()
        this.runMatmul(pass, `blk.${li}.ffn_down.weight`, sc.swigluOut, S, sc.n1, null, rotMemo)
        this.runRmsNorm(pass, sc.n1, `blk.${li}.post_ffw_norm`, S, Hd, sc.n2)
        this.dispatch(pass, 'add_inplace', uAdd, [sc.h2, sc.n2, nextH], Math.ceil((S * Hd) / 256))
        ;[curH, nextH] = [nextH, curH]
        continue
      }

      const isLinear = this.arch.hybrid && this.arch.hybrid.layerTypes[li] === 'linear'

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

        this.runMatmul(pass, `blk.${li}.attn_qkv.weight`, sc.n1, S, sc.qkvMixed, null, rotMemo)
        this.runMatmul(pass, `blk.${li}.attn_gate.weight`, sc.n1, S, sc.zGate, null, rotMemo)

        const convW = this.weights.get(`blk.${li}.ssm_conv1d.weight`)!.f32Buf!
        const convStates = this.ssmConvState.get(li)!
        const uConv = this.mkUniform([S, convDim, convK, loadState])
        const totalConvThreads = S * convDim + (convK - 1) * convDim
        const convGroups = Math.ceil(totalConvThreads / 64)
        const cgx = Math.min(convGroups, 65535)
        const cgy = Math.ceil(convGroups / cgx)
        this.dispatch(
          pass,
          'conv1d_causal',
          uConv,
          [sc.qkvMixed, convW, convStates[par], sc.qkvConv, convStates[par ^ 1]],
          cgx,
          cgy,
        )

        this.runMatmul(pass, `blk.${li}.ssm_alpha.weight`, sc.n1, S, sc.aProj, null, rotMemo)
        this.runMatmul(pass, `blk.${li}.ssm_beta.weight`, sc.n1, S, sc.bProj, null, rotMemo)

        const aNegBuf = this.weights.get(`blk.${li}.ssm_a`)!.f32Buf!
        const dtBiasBuf = this.weights.get(`blk.${li}.ssm_dt.bias`)!.f32Buf!
        const uGb = this.mkUniform([S, NV, 0, 0])
        this.dispatch(
          pass,
          'deltanet_gbeta',
          uGb,
          [sc.aProj, sc.bProj, aNegBuf, dtBiasBuf, sc.gbeta],
          Math.ceil((S * NV) / 64),
        )

        const recStates = this.ssmRecState.get(li)!
        const uRec = this.mkUniform([S, NV, DKV, DKV, NK, loadState, 0, 0])
        this.dispatch(
          pass,
          'deltanet_recur',
          uRec,
          [sc.qkvConv, sc.gbeta, recStates[par], sc.core, recStates[par ^ 1]],
          NV,
        )

        const ssmNormBuf = this.weights.get(`blk.${li}.ssm_norm.weight`)!.f32Buf!
        const uNormG = this.mkUniform([S * NV, DKV, this.arch.rmsEps, 0], new Set([2]))
        this.dispatch(pass, 'deltanet_norm_gate', uNormG, [sc.core, sc.zGate, ssmNormBuf, sc.normed], S * NV)

        rotMemo.clear()
        this.runMatmul(pass, `blk.${li}.ssm_out.weight`, sc.normed, S, sc.h2, curH, rotMemo)
      } else {
        // Full Attention Layer (Qwen3 dense or Qwen3.5 full-attn interval layer)
        const isQwen35Full = this.arch.archName === 'qwen35'

        if (isQwen35Full) {
          this.runMatmul(pass, `blk.${li}.attn_q.weight`, sc.n1, S, sc.qgBuf, null, rotMemo)
          const uSplit = this.mkUniform([S, H, Dh, 0])
          this.dispatch(pass, 'split_q_gate', uSplit, [sc.qgBuf, sc.qBuf, sc.gateBuf], Math.ceil((S * H * Dh) / 256))
        } else {
          this.runMatmul(pass, `blk.${li}.attn_q.weight`, sc.n1, S, sc.qBuf, null, rotMemo)
        }

        this.runMatmul(pass, `blk.${li}.attn_k.weight`, sc.n1, S, sc.kBuf, null, rotMemo)
        this.runMatmul(pass, `blk.${li}.attn_v.weight`, sc.n1, S, sc.vBuf, null, rotMemo)

        // Per-head Q/K RMSNorm
        this.runRmsNorm(pass, sc.qBuf, `blk.${li}.attn_q_norm.weight`, S * H, Dh, sc.qNormed)
        this.runRmsNorm(pass, sc.kBuf, `blk.${li}.attn_k_norm.weight`, S * KV, Dh, sc.kNormed)

        // RoPE (in-place on qNormed and kNormed)
        const uRopeQ = this.mkUniform([S, H, Dh, rotDim, this.cachePos, 0, 0, 0])
        const uRopeK = this.mkUniform([S, KV, Dh, rotDim, this.cachePos, 0, 0, 0])
        this.dispatch(
          pass,
          'rope',
          uRopeQ,
          [this.cosBuf, this.sinBuf, sc.qNormed],
          Math.ceil((S * H * (rotDim >> 1)) / 64),
        )
        this.dispatch(
          pass,
          'rope',
          uRopeK,
          [this.cosBuf, this.sinBuf, sc.kNormed],
          Math.ceil((S * KV * (rotDim >> 1)) / 64),
        )

        // Append K and V to Q8_0 KV Cache
        const uKv = this.mkUniform([S, KV, Dh, this.cachePos])
        const kvBlocks = S * KV * (Dh >> 5)
        const kQ = this.kCacheQ.get(li)!
        const kS = this.kCacheS.get(li)!
        const vQ = this.vCacheQ.get(li)!
        const vS = this.vCacheS.get(li)!
        this.dispatch(pass, 'kv_append_q8', uKv, [sc.kNormed, kQ, kS], kvBlocks)
        this.dispatch(pass, 'kv_append_q8', uKv, [sc.vBuf, vQ, vS], kvBlocks)

        // Online GQA Attention
        const uAtt = this.mkUniform([S, H, KV, Dh, this.cachePos, 0, 0, 0])
        this.dispatch(pass, 'attention_q8', uAtt, [sc.qNormed, kQ, kS, vQ, vS, sc.attOut], S * H)

        if (isQwen35Full) {
          const uGate = this.mkUniform([S * H * Dh, 0, 0, 0])
          this.dispatch(pass, 'sigmoid_gate', uGate, [sc.gateBuf, sc.attOut], Math.ceil((S * H * Dh) / 256))
        }

        rotMemo.clear()
        this.runMatmul(pass, `blk.${li}.attn_output.weight`, sc.attOut, S, sc.h2, curH, rotMemo)
      }

      // Post-Attention Norm & SwiGLU MLP
      const postNormName =
        this.arch.archName === 'qwen35' ? `blk.${li}.post_attention_norm.weight` : `blk.${li}.ffn_norm.weight`
      this.runRmsNorm(pass, sc.h2, postNormName, S, Hd, sc.n2)

      rotMemo.clear()
      this.runMatmul(pass, `blk.${li}.ffn_gate.weight`, sc.n2, S, sc.mlpGate, null, rotMemo)
      this.runMatmul(pass, `blk.${li}.ffn_up.weight`, sc.n2, S, sc.mlpUp, null, rotMemo)

      const uSwi = this.mkUniform([S * F, 0, 0, 0])
      this.dispatch(pass, 'swiglu', uSwi, [sc.mlpGate, sc.mlpUp, sc.swigluOut], Math.ceil((S * F) / 256))

      rotMemo.clear()
      this.runMatmul(pass, `blk.${li}.ffn_down.weight`, sc.swigluOut, S, nextH, sc.h2, rotMemo)
      ;[curH, nextH] = [nextH, curH]
    }

    if (this.arch.hybrid) {
      this.ssmParity ^= 1
    }
    this.cachePos += S

    if (!computeLogits) {
      pass.end()
      this.device.queue.submit([enc.finish()])
      await this.device.queue.onSubmittedWorkDone()
      return {}
    }

    // 3. Final RMSNorm & LM Head on the last token of the segment
    if (S > 1) {
      pass.end()
      enc.copyBufferToBuffer(curH, (S - 1) * Hd * 4, sc.lastHidden, 0, Hd * 4)
      const pass2 = enc.beginComputePass()
      this.runRmsNorm(pass2, sc.lastHidden, 'output_norm.weight', 1, Hd, sc.finalNorm)
      rotMemo.clear()
      this.runMatmul(pass2, 'output.weight', sc.finalNorm, 1, this.logitsBuf, null, rotMemo)
      if (greedyOnly) {
        const uArg = this.mkUniform([this.arch.vocab, 0, 0, 0])
        this.dispatch(pass2, 'argmax', uArg, [this.logitsBuf, this.argmaxBuf], 1)
      }
      pass2.end()
    } else {
      this.runRmsNorm(pass, curH, 'output_norm.weight', 1, Hd, sc.finalNorm)
      rotMemo.clear()
      this.runMatmul(pass, 'output.weight', sc.finalNorm, 1, this.logitsBuf, null, rotMemo)
      if (greedyOnly) {
        const uArg = this.mkUniform([this.arch.vocab, 0, 0, 0])
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
      return { greedyId }
    }

    enc.copyBufferToBuffer(this.logitsBuf, 0, this.logitsReadBuf, 0, this.arch.vocab * 4)
    this.device.queue.submit([enc.finish()])
    await this.logitsReadBuf.mapAsync(GPUMapMode.READ)
    this.logitsCpu.set(new Float32Array(this.logitsReadBuf.getMappedRange()))
    this.logitsReadBuf.unmap()
    return { logits: this.logitsCpu }
  }

  /**
   * Generate text or token IDs from a prompt string or ChatMessage array.
   */
  async generate(
    promptOrMessages: string | ChatMessage[] | number[],
    options: GenerateOptions = {},
  ): Promise<{ text: string; tokenIds: number[]; stats: GenerateStats }> {
    let promptIds: number[]
    let chatInput: ChatMessage[] | null = null
    if (Array.isArray(promptOrMessages) && typeof promptOrMessages[0] === 'number') {
      promptIds = promptOrMessages as number[]
    } else if (typeof promptOrMessages === 'string') {
      if (!this.tokenizer) throw new Error('hachiue: GGUF header does not contain a tokenizer')
      promptIds = this.tokenizer.encode(promptOrMessages)
    } else {
      if (!this.tokenizer) throw new Error('hachiue: GGUF header does not contain a tokenizer')
      chatInput = promptOrMessages as ChatMessage[]
      const canExtendLastChat =
        this.lastMessages.length > 0 &&
        chatInput.length > this.lastMessages.length &&
        this.lastMessages.every(
          (m, idx) => chatInput![idx].role === m.role && chatInput![idx].content === m.content,
        )
      if (canExtendLastChat) {
        const deltaMsgs = chatInput.slice(this.lastMessages.length)
        const formattedDelta = this.tokenizer.formatChat(deltaMsgs)
        promptIds = [...this.lastTurnTokens, ...this.tokenizer.encode(formattedDelta)]
      } else {
        const formatted = this.tokenizer.formatChat(chatInput)
        promptIds = this.tokenizer.encode(formatted)
      }
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
    if (this.arch.eomTokenId !== undefined) stopSet.add(this.arch.eomTokenId)
    if (this.arch.eotTokenId !== undefined) stopSet.add(this.arch.eotTokenId)
    if (this.arch.archName === 'plamo3') {
      const eomId = this.tokenizer?.tokenToId('<|plamo:eom|>')
      if (eomId !== undefined) stopSet.add(eomId)
    }
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

    if (chatInput && this.tokenizer) {
      this.lastMessages = [...chatInput.map((m) => ({ ...m })), { role: 'assistant', content: text }]
      const turnEndStr = this.arch.archName === 'plamo3' ? '<|plamo:eom|>' : '<|im_end|>\n'
      this.lastTurnTokens = [...promptIds, ...generatedIds, ...this.tokenizer.encode(turnEndStr)]
    }

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
        reusedCache: prefixMatch > 0,
        backend: this.backend,
        isWorker: this.isWorker,
      },
    }
  }
}
