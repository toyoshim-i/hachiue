/**
 * Supported GGML tensor quantization / floating-point types in hachiue.
 */
export enum GgmlType {
  F32 = 0,
  F16 = 1,
  Q8_0 = 8,
  /** Bonsai 1: 1-bit binary weights, 128 elements per block (2B fp16 scale + 16B sign bits = 18B) */
  Q1_0 = 41,
  /** Upstream ternary 2-bit weights, 64 elements per block (2B fp16 scale + 16B 2-bit codes = 18B) */
  Q2_0 = 42,
  /** PrismML ternary 2-bit weights, 128 elements per block (2B fp16 scale + 32B 2-bit codes = 34B) */
  PQ2_0 = 142,
  /** PrismML dense ternary trits, 128 elements per block (24B 5-trit qs + 2B 4-trit qh + 2B fp16 scale = 28B) */
  PTQ1_0 = 143,
}

export type QuantFormat = 'q1_0' | 'pq2_0' | 'ptq1_0' | 'q2_0' | 'f32' | 'f16'

export interface GgufTensorInfo {
  name: string
  dims: number[]
  type: GgmlType
  /** Byte offset relative to the start of the GGUF data section */
  relOffset: number
  /** Absolute byte offset in the GGUF file */
  absOffset: number
  /** Total byte length of this tensor in the GGUF file */
  byteLength: number
}

export interface HadamardConfig {
  version: number
  blockSize: number
  tiedOutput: boolean
  gdnVGrouped: boolean
  signMode: 'identity' | 'explicit'
  /** Map from input width (e.g. 2048, 6144) to +/-1 sign array of that length */
  signsByWidth: Map<number, Int8Array>
  /** Set of GGUF tensor names whose input activations must undergo forward Hadamard transform */
  foldedWeights: Set<string>
  /** Set of GGUF tensor names whose row-lookup outputs must undergo inverse Hadamard transform */
  inverseWeights: Set<string>
}

export interface RopeConfig {
  theta: number
  type: 'normal' | 'yarn' | 'mrope'
  factor: number
  originalMaxPositions: number
  rotaryDim: number
}

export interface HybridLayerConfig {
  layerTypes: Array<'linear' | 'full'>
  linearKeyHeads: number
  linearValueHeads: number
  linearHeadDim: number
  convKernel: number
}

export interface ModelArch {
  archName: 'qwen3' | 'qwen35'
  modelName: string
  sizeLabel: string
  layers: number
  hidden: number
  intermediate: number
  heads: number
  kvHeads: number
  headDim: number
  rmsEps: number
  maxPositions: number
  vocab: number
  eosTokenId: number
  bosTokenId?: number
  padTokenId?: number
  tieWordEmbeddings: boolean
  rope: RopeConfig
  hybrid?: HybridLayerConfig
  hadamard?: HadamardConfig
}

export interface TokenizerData {
  tokens: string[]
  tokenTypes?: number[]
  merges: string[]
  eosTokenId: number
  bosTokenId?: number
  padTokenId?: number
  addBosToken: boolean
  chatTemplate?: string
}

export interface ParsedGgufHeader {
  version: number
  tensorCount: number
  kvCount: number
  dataStartOffset: number
  totalFileSize: number
  meta: Record<string, unknown>
  tensors: Record<string, GgufTensorInfo>
  arch: ModelArch
  tokenizer?: TokenizerData
}

export interface LoadProgress {
  phase: 'header' | 'cache' | 'pipelines' | 'weights' | 'ready'
  loadedBytes?: number
  totalBytes?: number
  cached?: boolean
  message?: string
}

export interface StorageOptions {
  /** Chunk size in bytes for HTTP Range requests and GPU staging flushes (default: 16 MiB) */
  chunkBytes?: number
  /** Whether to persist downloaded chunks in OPFS / IndexedDB cache (default: true) */
  useCache?: boolean
  /** Custom fetch implementation or Range fetcher */
  fetchRange?: (url: string, start: number, endInclusive: number, signal?: AbortSignal) => Promise<Uint8Array>
  onProgress?: (progress: LoadProgress) => void
  signal?: AbortSignal
}

export interface EngineOptions extends StorageOptions {
  /** URL to the .gguf model on Hugging Face or any HTTP Range-capable server */
  url: string
  /** Maximum context length to allocate KV cache up to (default: 4096) */
  maxSeqLen?: number
  /** KV cache quantization mode (default: 'q8' for 4x memory savings) */
  kvCache?: 'q8' | 'f16' | 'f32'
  /** Optional existing GPUDevice */
  device?: GPUDevice
  onDeviceLost?: (info: { reason: string; message: string }) => void
}

export interface GenerateOptions {
  maxNewTokens?: number
  temperature?: number
  topK?: number
  topP?: number
  minP?: number
  repetitionPenalty?: number
  stopTokens?: number[]
  onToken?: (tokenId: number, textDelta: string) => void
  signal?: AbortSignal
}

export interface GenerateStats {
  promptTokens: number
  generatedTokens: number
  prefillMs: number
  decodeMs: number
  prefillTokensPerSec: number
  decodeTokensPerSec: number
  reusedCache: boolean
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}
