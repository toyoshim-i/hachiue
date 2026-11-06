export { HachiueEngine } from './engine.js'
export {
  computeTensorByteLength,
  defaultFetchRange,
  dequantizeBlock128Cpu,
  dequantizeBlockCpu,
  fetchGgufHeader,
  fp16ToFp32,
  fwhtBlockCpu,
  getBlockSpec,
  HeaderOutOfDataError,
  parseGgufHeaderBytes,
} from './gguf.js'
export { SHADERS } from './kernels.js'
export { type SampleConfig, sampleLogits } from './sampler.js'
export {
  clearModelCache,
  getCachedModelSize,
  isModelCached,
  prefetchModel,
  readCachedRange,
  streamModelChunks,
} from './storage.js'
export { GgufTokenizer } from './tokenizer.js'
export {
  buildInstructTermMessages,
  buildInstructTextMessages,
  buildPlamo3TranslationPrompt,
  extractCleanTermTranslation,
  extractCleanTextTranslation,
  type GlossaryEntry,
  isValidTargetLanguageTranslation,
  normalizeLangCode,
  stripModelReasoningAndTags,
  type TranslateTermOptions,
  type TranslateTextOptions,
  type TranslationHistoryEntry,
  type TranslationLangCode,
  type TranslationResult,
  translateTerm,
  translateText,
} from './translate.js'
export type {
  BackendInfo,
  BackendType,
  ChatMessage,
  EngineOptions,
  GenerateOptions,
  GenerateStats,
  GgufTensorInfo,
  HadamardConfig,
  HybridLayerConfig,
  LoadProgress,
  ModelArch,
  ParsedGgufHeader,
  QuantFormat,
  RopeConfig,
  SlidingWindowConfig,
  StorageOptions,
  TokenizerData,
} from './types.js'
export { GgmlType } from './types.js'
export { createWasmBackend, createWasmCore, type WasmBackend } from './wasm-backend.js'
export {
  getWasmKernelModuleBytes,
  WASM_MEM_CONSTS,
  WASM_MEM_FP16_TABLE,
  WASM_MEM_Q1_LUT,
  WASM_MEM_WEIGHTS_BASE,
  WASM_MEM_X_F32,
  WASM_MEM_X_F32_SUM,
  WASM_MEM_X_PQ2_Q16,
  WASM_MEM_X_Q16,
  WASM_MEM_X_SCALE,
  WASM_MEM_Y_F32,
  type WasmKernelExports,
} from './wasm-kernels.js'
