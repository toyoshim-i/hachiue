import {
  GgmlType,
  type GgufTensorInfo,
  type HadamardConfig,
  type HybridLayerConfig,
  type ModelArch,
  type ParsedGgufHeader,
  type RopeConfig,
  type StorageOptions,
  type TokenizerData,
} from './types.js'

const GGUF_MAGIC = 0x46554747 // "GGUF" little-endian
const DEFAULT_ALIGNMENT = 32
const INITIAL_HEADER_FETCH = 2 * 1024 * 1024 // 2 MiB initial probe
const MAX_HEADER_FETCH = 32 * 1024 * 1024 // 32 MiB cap

export class HeaderOutOfDataError extends Error {
  constructor() {
    super('GGUF header buffer truncated')
  }
}

/**
 * Returns [elementsPerBlock, bytesPerBlock] for a supported GGML type.
 */
export function getBlockSpec(type: GgmlType): [number, number] {
  switch (type) {
    case GgmlType.F32:
      return [1, 4]
    case GgmlType.F16:
      return [1, 2]
    case GgmlType.Q8_0:
      return [32, 34]
    case GgmlType.Q1_0:
      return [128, 18] // 2B fp16 scale + 16B sign bits
    case GgmlType.Q2_0:
      return [64, 18] // 2B fp16 scale + 16B 2-bit codes
    case GgmlType.PQ2_0:
      return [128, 34] // 2B fp16 scale + 32B 2-bit codes
    case GgmlType.PTQ1_0:
      return [128, 28] // 24B qs + 2B qh + 2B fp16 scale
    default:
      throw new Error(`hachiue: unsupported GGML tensor type ${type}`)
  }
}

export function computeTensorByteLength(dims: number[], type: GgmlType): number {
  const [blkElems, blkBytes] = getBlockSpec(type)
  const totalElems = dims.reduce((a, b) => a * b, 1)
  if (totalElems % blkElems !== 0) {
    throw new Error(`hachiue: tensor elements (${totalElems}) not divisible by block size (${blkElems})`)
  }
  return (totalElems / blkElems) * blkBytes
}

/**
 * Convert IEEE 754 binary16 (uint16) to float32 number.
 */
export function fp16ToFp32(h: number): number {
  const s = h & 0x8000 ? -1 : 1
  const e = (h >> 10) & 0x1f
  const m = h & 0x03ff
  if (e === 0) return s * m * 2 ** -24
  if (e === 31) return m ? NaN : s * Infinity
  return s * (1024 + m) * 2 ** (e - 25)
}

/**
 * Reference CPU dequantization of a single 128-element block for testing and verification.
 */
export function dequantizeBlock128Cpu(blockBytes: Uint8Array, type: GgmlType, out: Float32Array, outOffset = 0): void {
  const dv = new DataView(blockBytes.buffer, blockBytes.byteOffset, blockBytes.byteLength)
  if (type === GgmlType.Q1_0) {
    const d = fp16ToFp32(dv.getUint16(0, true))
    for (let j = 0; j < 128; j++) {
      const bit = (blockBytes[2 + (j >> 3)] >> (j & 7)) & 1
      out[outOffset + j] = bit ? d : -d
    }
    return
  }
  if (type === GgmlType.PQ2_0) {
    const d = fp16ToFp32(dv.getUint16(0, true))
    for (let j = 0; j < 128; j++) {
      const q = (blockBytes[2 + (j >> 2)] >> ((j & 3) * 2)) & 3
      out[outOffset + j] = (q - 1) * d
    }
    return
  }
  if (type === GgmlType.PTQ1_0) {
    // 24B qs + 2B qh + 2B fp16 d
    const d = fp16ToFp32(dv.getUint16(26, true))
    const pow3 = [1, 3, 9, 27, 81]
    // Stage 1: j = 0..15 (16 bytes -> 80 trits)
    let wIdx = outOffset
    for (let n = 0; n < 5; n++) {
      const p = pow3[n]
      for (let m = 0; m < 16; m++) {
        const q = (blockBytes[m] * p) & 0xff
        const xi = (q * 3) >> 8
        out[wIdx++] = (xi - 1) * d
      }
    }
    // Stage 2: j = 16..23 (8 bytes -> 40 trits)
    for (let n = 0; n < 5; n++) {
      const p = pow3[n]
      for (let m = 0; m < 8; m++) {
        const q = (blockBytes[16 + m] * p) & 0xff
        const xi = (q * 3) >> 8
        out[wIdx++] = (xi - 1) * d
      }
    }
    // Tail qh: 2 bytes -> 8 trits
    for (let n = 0; n < 4; n++) {
      const p = pow3[n]
      for (let h = 0; h < 2; h++) {
        const q = (blockBytes[24 + h] * p) & 0xff
        const xi = (q * 3) >> 8
        out[wIdx++] = (xi - 1) * d
      }
    }
    return
  }
  throw new Error(`dequantizeBlock128Cpu: unsupported type ${type}`)
}

/**
 * Repack a single 28-byte PTQ1_0 block into 32 bytes (8 x u32) of 2-bit codes (0, 1, 2) + 2-byte fp16 scale.
 * Doing this lightweight integer repack during the streaming upload (or on GPU) unifies PTQ1_0 and PQ2_0
 * or we can keep PTQ1_0 in its native 26B trit + 2B scale layout.
 * Note: Keeping PTQ1_0 in its native 26B trit layout (or 28B raw block) saves 18.75% VRAM over PQ2_0!
 */

/**
 * Reference CPU Fast Walsh-Hadamard Transform (Normalized Sylvester-Walsh-Hadamard).
 * Block size must be a power of two (e.g. 512).
 */
export function fwhtBlockCpu(vec: Float32Array, offset: number, blockSize: number): void {
  for (let h = 1; h < blockSize; h <<= 1) {
    for (let i = 0; i < blockSize; i += h << 1) {
      for (let j = i; j < i + h; j++) {
        const x = vec[offset + j]
        const y = vec[offset + j + h]
        vec[offset + j] = x + y
        vec[offset + j + h] = x - y
      }
    }
  }
  const scale = 1 / Math.sqrt(blockSize)
  for (let i = 0; i < blockSize; i++) {
    vec[offset + i] *= scale
  }
}

/**
 * Parse a GGUF v2/v3 header from an in-memory Uint8Array.
 * Throws HeaderOutOfDataError if `bytes` ends before the tensor table finishes.
 */
export function parseGgufHeaderBytes(bytes: Uint8Array): ParsedGgufHeader {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const decoder = new TextDecoder()
  let pos = 0

  const need = (n: number): void => {
    if (pos + n > bytes.byteLength) throw new HeaderOutOfDataError()
  }
  const u8 = (): number => {
    need(1)
    return dv.getUint8(pos++)
  }
  const i8 = (): number => {
    need(1)
    return dv.getInt8(pos++)
  }
  const u16 = (): number => {
    need(2)
    const v = dv.getUint16(pos, true)
    pos += 2
    return v
  }
  const i16 = (): number => {
    need(2)
    const v = dv.getInt16(pos, true)
    pos += 2
    return v
  }
  const u32 = (): number => {
    need(4)
    const v = dv.getUint32(pos, true)
    pos += 4
    return v
  }
  const i32 = (): number => {
    need(4)
    const v = dv.getInt32(pos, true)
    pos += 4
    return v
  }
  const f32 = (): number => {
    need(4)
    const v = dv.getFloat32(pos, true)
    pos += 4
    return v
  }
  const f64 = (): number => {
    need(8)
    const v = dv.getFloat64(pos, true)
    pos += 8
    return v
  }
  const u64 = (): number => {
    need(8)
    const lo = dv.getUint32(pos, true)
    const hi = dv.getUint32(pos + 4, true)
    pos += 8
    return hi * 0x100000000 + lo
  }
  const i64 = (): number => {
    need(8)
    const lo = dv.getUint32(pos, true)
    const hi = dv.getInt32(pos + 4, true)
    pos += 8
    return hi * 0x100000000 + lo
  }
  const str = (): string => {
    const len = u64()
    need(len)
    const s = decoder.decode(bytes.subarray(pos, pos + len))
    pos += len
    return s
  }
  const val = (vtype: number): unknown => {
    switch (vtype) {
      case 0:
        return u8()
      case 1:
        return i8()
      case 2:
        return u16()
      case 3:
        return i16()
      case 4:
        return u32()
      case 5:
        return i32()
      case 6:
        return f32()
      case 7:
        return u8() !== 0
      case 8:
        return str()
      case 9: {
        const elemType = u32()
        const len = u64()
        const arr = new Array(len)
        for (let i = 0; i < len; i++) arr[i] = val(elemType)
        return arr
      }
      case 10:
        return u64()
      case 11:
        return i64()
      case 12:
        return f64()
      default:
        throw new Error(`hachiue: unknown GGUF metadata value type ${vtype}`)
    }
  }

  if (u32() !== GGUF_MAGIC) {
    throw new Error('hachiue: invalid GGUF magic header')
  }
  const version = u32()
  if (version !== 2 && version !== 3) {
    throw new Error(`hachiue: unsupported GGUF version ${version}`)
  }
  const tensorCount = u64()
  const kvCount = u64()

  const meta: Record<string, unknown> = {}
  for (let i = 0; i < kvCount; i++) {
    const k = str()
    const vt = u32()
    meta[k] = val(vt)
  }

  const rawTensors: Array<{ name: string; dims: number[]; type: GgmlType; relOffset: number }> = []
  for (let i = 0; i < tensorCount; i++) {
    const name = str()
    const ndims = u32()
    const dims: number[] = []
    for (let d = 0; d < ndims; d++) dims.push(u64())
    const type = u32() as GgmlType
    const relOffset = u64()
    rawTensors.push({ name, dims, type, relOffset })
  }

  const alignment = Number(meta['general.alignment'] ?? DEFAULT_ALIGNMENT)
  const dataStartOffset = Math.ceil(pos / alignment) * alignment

  const tensors: Record<string, GgufTensorInfo> = {}
  let maxEndOffset = dataStartOffset
  for (const rt of rawTensors) {
    const byteLength = computeTensorByteLength(rt.dims, rt.type)
    const absOffset = dataStartOffset + rt.relOffset
    tensors[rt.name] = {
      name: rt.name,
      dims: rt.dims,
      type: rt.type,
      relOffset: rt.relOffset,
      absOffset,
      byteLength,
    }
    if (absOffset + byteLength > maxEndOffset) {
      maxEndOffset = absOffset + byteLength
    }
  }

  const arch = parseModelArch(meta, tensors)
  const tokenizer = parseTokenizerData(meta)

  return {
    version,
    tensorCount,
    kvCount,
    dataStartOffset,
    totalFileSize: maxEndOffset,
    meta,
    tensors,
    arch,
    tokenizer,
  }
}

function parseModelArch(meta: Record<string, unknown>, tensors: Record<string, GgufTensorInfo>): ModelArch {
  const archStr = String(meta['general.architecture'] ?? '')
  if (archStr !== 'qwen3' && archStr !== 'qwen35') {
    throw new Error(`hachiue: unsupported model architecture "${archStr}" (expected "qwen3" or "qwen35")`)
  }
  const p = (key: string): unknown => meta[`${archStr}.${key}`]
  const reqNum = (key: string): number => {
    const v = p(key)
    if (v === undefined) throw new Error(`hachiue: missing required GGUF metadata ${archStr}.${key}`)
    return Number(v)
  }

  const layers = reqNum('block_count')
  const hidden = reqNum('embedding_length')
  const intermediate = reqNum('feed_forward_length')
  const heads = reqNum('attention.head_count')
  const kvHeads = reqNum('attention.head_count_kv')
  const headDim = Number(p('attention.key_length') ?? hidden / heads)
  const rmsEps = reqNum('attention.layer_norm_rms_epsilon')
  const maxPositions = reqNum('context_length')

  const embdTensor = tensors['token_embd.weight']
  if (!embdTensor) throw new Error('hachiue: missing token_embd.weight tensor')
  const vocab = embdTensor.dims[1]
  const tieWordEmbeddings = !('output.weight' in tensors)

  const eosTokenId = Number(meta['tokenizer.ggml.eos_token_id'] ?? 151645)
  const bosTokenId = meta['tokenizer.ggml.bos_token_id'] !== undefined ? Number(meta['tokenizer.ggml.bos_token_id']) : undefined
  const padTokenId = meta['tokenizer.ggml.padding_token_id'] !== undefined ? Number(meta['tokenizer.ggml.padding_token_id']) : undefined

  const theta = Number(p('rope.freq_base') ?? 1000000.0)
  const ropeScaleType = String(p('rope.scaling.type') ?? 'none')
  const rotaryDim = Number(p('rope.dimension_count') ?? headDim)
  const rope: RopeConfig = {
    theta,
    type: ropeScaleType === 'yarn' ? 'yarn' : archStr === 'qwen35' ? 'mrope' : 'normal',
    factor: Number(p('rope.scaling.factor') ?? 1.0),
    originalMaxPositions: Number(p('rope.scaling.original_context_length') ?? maxPositions),
    rotaryDim,
  }

  let hybrid: HybridLayerConfig | undefined
  if (archStr === 'qwen35') {
    const interval = reqNum('full_attention_interval')
    const linearKeyHeads = reqNum('ssm.group_count')
    const linearValueHeads = reqNum('ssm.time_step_rank')
    const linearHeadDim = reqNum('ssm.state_size')
    const convKernel = reqNum('ssm.conv_kernel')
    const layerTypes: Array<'linear' | 'full'> = []
    for (let li = 0; li < layers; li++) {
      layerTypes.push(li % interval === interval - 1 ? 'full' : 'linear')
    }
    hybrid = {
      layerTypes,
      linearKeyHeads,
      linearValueHeads,
      linearHeadDim,
      convKernel,
    }
  }

  let hadamard: HadamardConfig | undefined
  if (meta['prism.hadamard.version'] !== undefined) {
    const version = Number(meta['prism.hadamard.version'])
    const blockSize = Number(meta['prism.hadamard.block_size'] ?? 512)
    const tiedOutput = Boolean(meta['prism.hadamard.tied_output'] ?? false)
    const gdnVGrouped = Boolean(meta['prism.hadamard.gdn_v_grouped'] ?? false)
    const signMode = String(meta['prism.hadamard.sign_mode'] ?? 'identity') as 'identity' | 'explicit'
    const weightNames = (meta['prism.hadamard.weight_names'] as string[]) ?? []
    const inverseNames = (meta['prism.hadamard.inverse_weight_names'] as string[]) ?? []

    const signsByWidth = new Map<number, Int8Array>()
    if (signMode === 'explicit') {
      const widths = (meta['prism.hadamard.sign_widths'] as number[]) ?? []
      const values = (meta['prism.hadamard.sign_values'] as number[]) ?? []
      let offset = 0
      for (const w of widths) {
        const arr = new Int8Array(w)
        for (let i = 0; i < w; i++) arr[i] = values[offset + i]
        signsByWidth.set(w, arr)
        offset += w
      }
    }
    const foldedWeights = new Set<string>(weightNames)
    if (tiedOutput && inverseNames.includes('token_embd.weight')) {
      foldedWeights.add('output.weight')
    }
    hadamard = {
      version,
      blockSize,
      tiedOutput,
      gdnVGrouped,
      signMode,
      signsByWidth,
      foldedWeights,
      inverseWeights: new Set<string>(inverseNames),
    }
  }

  return {
    archName: archStr,
    modelName: String(meta['general.name'] ?? archStr),
    sizeLabel: String(meta['general.size_label'] ?? ''),
    layers,
    hidden,
    intermediate,
    heads,
    kvHeads,
    headDim,
    rmsEps,
    maxPositions,
    vocab,
    eosTokenId,
    bosTokenId,
    padTokenId,
    tieWordEmbeddings,
    rope,
    hybrid,
    hadamard,
  }
}

function parseTokenizerData(meta: Record<string, unknown>): TokenizerData | undefined {
  const tokens = meta['tokenizer.ggml.tokens'] as string[] | undefined
  const merges = meta['tokenizer.ggml.merges'] as string[] | undefined
  if (!tokens || !merges) return undefined
  return {
    tokens,
    tokenTypes: meta['tokenizer.ggml.token_type'] as number[] | undefined,
    merges,
    eosTokenId: Number(meta['tokenizer.ggml.eos_token_id'] ?? 151645),
    bosTokenId: meta['tokenizer.ggml.bos_token_id'] !== undefined ? Number(meta['tokenizer.ggml.bos_token_id']) : undefined,
    padTokenId: meta['tokenizer.ggml.padding_token_id'] !== undefined ? Number(meta['tokenizer.ggml.padding_token_id']) : undefined,
    addBosToken: Boolean(meta['tokenizer.ggml.add_bos_token'] ?? false),
    chatTemplate: meta['tokenizer.chat_template'] as string | undefined,
  }
}

/**
 * Default HTTP Range fetcher using `Range: bytes=start-endInclusive`.
 */
export async function defaultFetchRange(
  url: string,
  start: number,
  endInclusive: number,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const expectedLen = endInclusive - start + 1
  const res = await fetch(url, {
    headers: { Range: `bytes=${start}-${endInclusive}` },
    signal,
  })
  if (res.status === 206) {
    const ab = await res.arrayBuffer()
    return new Uint8Array(ab)
  }
  if (res.ok && res.body) {
    // Fallback if server ignores Range header: read only up to endInclusive and cancel stream immediately
    const reader = res.body.getReader()
    const out = new Uint8Array(expectedLen)
    let cursor = 0
    let written = 0
    try {
      while (written < expectedLen) {
        const { done, value } = await reader.read()
        if (done) break
        if (cursor + value.byteLength > start) {
          const sliceStart = Math.max(0, start - cursor)
          const sliceEnd = Math.min(value.byteLength, sliceStart + (expectedLen - written))
          out.set(value.subarray(sliceStart, sliceEnd), written)
          written += sliceEnd - sliceStart
        }
        cursor += value.byteLength
      }
    } finally {
      await reader.cancel().catch(() => {})
    }
    return out.subarray(0, written)
  }
  throw new Error(`hachiue: HTTP Range fetch failed (${res.status} ${res.statusText}) for ${url}`)
}

/**
 * Fetch and parse only the GGUF header from a remote URL via progressive HTTP Range requests.
 */
export async function fetchGgufHeader(url: string, options: StorageOptions = {}): Promise<ParsedGgufHeader> {
  const fetchRange = options.fetchRange ?? defaultFetchRange
  options.onProgress?.({ phase: 'header', message: 'Fetching GGUF header...' })

  let size = INITIAL_HEADER_FETCH
  let buf = new Uint8Array(await fetchRange(url, 0, size - 1, options.signal))

  for (;;) {
    try {
      return parseGgufHeaderBytes(buf)
    } catch (err) {
      if (!(err instanceof HeaderOutOfDataError)) throw err
      if (buf.byteLength < size) {
        throw new Error(`hachiue: GGUF file at ${url} is truncated inside the header (${buf.byteLength} bytes)`)
      }
      if (size >= MAX_HEADER_FETCH) {
        throw new Error(`hachiue: GGUF header exceeds ${MAX_HEADER_FETCH} bytes`)
      }
      const nextSize = Math.min(size * 2, MAX_HEADER_FETCH)
      const ext = await fetchRange(url, buf.byteLength, nextSize - 1, options.signal)
      const combined = new Uint8Array(buf.byteLength + ext.byteLength)
      combined.set(buf, 0)
      combined.set(ext, buf.byteLength)
      buf = combined
      size = nextSize
    }
  }
}
