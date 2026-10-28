# 🪴 hachiue (鉢植え)

Zero-dependency, mobile-optimized WebGPU inference engine for **PrismML Bonsai 1 (1-bit `Q1_0` / Ternary `PQ2_0`)** and **Bonsai 2 (Ternary 1.58-bit `PTQ1_0` / `PQ2_0`)** GGUF models.

👉 **[Live Demo (GitHub Pages)](https://toyoshim-i.github.io/hachiue/)**

## Key Features

- **GGUF Direct (Zero Offline Conversion)**:
  - Point `HachiueEngine.create({ url })` directly at any `.gguf` URL on Hugging Face.
  - Includes a zero-dependency GGUF v3 parser and embedded BPE tokenizer extracted straight from the GGUF header.
- **Double-Buffered Mobile Streaming & Resumable Cache**:
  - Replaces `Response.clone()` and monolithic `CacheStorage.put()` with bounded **16 MiB HTTP `Range` requests** persisted chunk-by-chunk to **OPFS (Origin Private File System)** (with automatic IndexedDB chunk fallback on iOS Safari).
  - Uses a **2-slot double-buffered pipeline** that overlaps downloading chunk `N + 1` with writing chunk `N` to local cache and uploading tensors to WebGPU (peak chunk RAM $\le$ 32 MiB).
  - Enforces WebGPU queue backpressure (`await device.queue.onSubmittedWorkDone()`) and macrotask GC yields every 16 MiB so iOS Safari Jetsam limits are never exceeded—even when loading 1.1 GB 4B models on an iPhone.
- **Zero-Inflation Planar VRAM Layout & Zero-Allocation Runtime**:
  - Keeps `Q1_0` (18B / 128 weights), `PTQ1_0` (28B / 128 weights), and `PQ2_0` (34B / 128 weights) at their exact native GGUF footprint in GPU memory.
  - Packs `fp16` block scales into `u32` pairs unpacked via WGSL `unpack2x16float` (saving ~11% VRAM without requiring `shader-f16`).
  - Shares `token_embd.weight` and `output.weight` (`lm_head`) GPU buffers when `tie_word_embeddings` is enabled, and automatically splits large tensors across `64 MiB` storage buffer chunks.
  - Reuses a single `1.125 MiB` CPU staging buffer pair during loading and a pre-allocated `~2.8 MiB` GPU scratch buffer pool during inference (**zero dynamic GPU/CPU buffer allocations per token**).
- **Full Bonsai 1 & Bonsai 2 Architecture Support**:
  - Multi-row (`ROWS_PER_WG = 4`) WGSL compute kernels for `Q1_0`, `PTQ1_0` (base-3 dense trits), and `PQ2_0` (2-bit ternary slots).
  - Workgroup-shared Fast Walsh-Hadamard Transform (`hadamard_512`) and grouped head permutation (`permute_gdn`) for `prism.hadamard` folded weights and latent embeddings.
  - **Qwen3** dense GQA attention (`Q8_0` KV cache) and **Qwen3.5** Hybrid Attention (3:1 Gated DeltaNet recurrent linear attention + gated full attention), with automatic multi-turn prefix reuse for both KV cache and Gated DeltaNet recurrent state.

## Supported Models & Benchmarks

### Model Lineup & Measured Speed (`hachiue` WebGPU)

| Model | Base Arch | Format | bpw | File Size | iPhone 17 Pro Max (Safari) | Apple Silicon Mac (Chrome) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **Bonsai 1.7B** | Qwen3-1.7B | `Q1_0` (1-bit) | 1.125 | **237 MiB** | **51.0 tok/s** *(prefill 187 ms)* | **87.6 tok/s** *(prefill 127 ms)* |
| **Ternary Bonsai 1.7B** | Qwen3-1.7B | `PQ2_0` (Ternary) | 2.125 | **420 MiB** | Supported | Supported |
| **Bonsai 2 2B** | Qwen3.5-2B | `PTQ1_0` (Ternary) | 1.750 | **406 MiB** | Supported | **37.8 tok/s** |
| **Bonsai 2 2B** | Qwen3.5-2B | `PQ2_0` (Ternary) | 2.125 | **495 MiB** | Supported | Supported |
| **Bonsai 4B** | Qwen3-4B | `Q1_0` (1-bit) | 1.125 | **546 MiB** | Supported | Supported |
| **Bonsai 2 4B** | Qwen3.5-4B | `PTQ1_0` (Ternary) | 1.750 | **902 MiB** | Supported | Supported |
| **Bonsai 2 4B** | Qwen3.5-4B | `PQ2_0` (Ternary) | 2.125 | **1,093 MiB** | **20.6 tok/s** *(prefill 446 ms)* | Supported |
| **Bonsai 8B** | Qwen3-8B | `Q1_0` (1-bit) | 1.125 | **1,100 MiB** | Supported | Supported |

### Published Quality Benchmarks

#### 1. Quantization Retention (PrismML EvalScope v1.4.2)

| Model | Quantization | Size | **6-Task Avg** | **IFEval** *(Formatting)* | **MMLU-R** *(Knowledge)* | **MuSR** *(Long-Context)* | **GSM8K** *(Math)* |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **1-bit Bonsai 1.7B** | `Q1_0` (1-bit) | 0.24 GB | 49.6 | 63.0 | 43.2 | 45.1 | 66.3 |
| **Ternary Bonsai 1.7B** | `PQ2_0` (Ternary) | 0.44 GB | **58.5** | **70.1** | **52.9** | **50.8** | **74.2** |
| *(Ref) Qwen3 1.7B* | *FP16* | *3.44 GB* | *66.6* | *70.3* | *66.8* | *50.1* | *83.1* |
| **1-bit Bonsai 4B** | `Q1_0` (1-bit) | 0.57 GB | 62.7 | 69.6 | 58.7 | 41.4 | 87.3 |
| **Ternary Bonsai 4B** | `PQ2_0` (Ternary) | 1.02 GB | **70.7** | **72.1** | **69.7** | **45.1** | **90.5** |
| *(Ref) Qwen3 4B* | *FP16* | *8.04 GB* | *77.1* | *80.0* | *79.8* | *57.4* | *92.1* |
| **1-bit Bonsai 8B** | `Q1_0` (1-bit) | 1.15 GB | 70.5 | 79.8 | 65.7 | 50.0 | 88.0 |
| **Ternary Bonsai 8B** | `PQ2_0` (Ternary) | 2.18 GB | **75.5** | **81.8** | **72.6** | **56.2** | **91.0** |
| *(Ref) Qwen3 8B* | *FP16* | *16.38 GB* | *79.3* | *81.5* | *83.0* | *55.0* | *93.0* |

#### 2. Text Processing & Multilingual / Translation (Base Qwen3 vs Qwen3.5 Backbone)

Bonsai 2 adopts the **Qwen3.5** hybrid-attention backbone (`vocab = 248,320` vs `151,936` in Bonsai 1) plus Hadamard-rotated ternary quantization (~98.2% FP16 score retention):

| Benchmark | Capability | **Qwen3-1.7B** *(Bonsai 1.7B base)* | **Qwen3.5-2B** *(Bonsai 2 2B base)* | **Qwen3-4B** *(Bonsai 4B base)* | **Qwen3.5-4B** *(Bonsai 2 4B base)* |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **WMT24++** | Machine Translation (55 languages) | 39.3 | **45.8** | 58.9 | **66.6** |
| **MAXIFE** | Multilingual Instruction Following | 50.7 | **60.6** | 72.1 | **78.0** |
| **MMMLU** | Multilingual Knowledge (14 languages) | 57.0 | **63.1** | 70.8 | **76.1** |
| **IFEval** | Formatting & Constraint Following | 72.5 | **78.6** | 87.4 | **89.8** |
| **IFBench** | Complex Instruction Processing | 26.7 | **41.3** | 50.4 | **59.2** |
| **LongBench v2** | Long-Document Summarization & QA | 26.5 | **38.7** | 42.8 | **50.0** |

## Usage

```ts
import { HachiueEngine, MODEL_REGISTRY } from './dist/index.js';

const engine = await HachiueEngine.create({
  url: MODEL_REGISTRY['bonsai-2-2b-ptq1'].url,
  onProgress: (p) => {
    console.log(`${p.phase}: ${(p.loadedBytes / 1048576).toFixed(1)} MiB`);
  },
});

const result = await engine.generate({
  messages: [
    { role: 'user', content: 'Translate into Japanese: Small trees thrive in small pots.' },
  ],
  maxNewTokens: 256,
  temperature: 0.7,
  onToken: (_id, text) => process.stdout.write(text),
});

console.log(`\nSpeed: ${result.decodeTokensPerSec.toFixed(1)} tok/s (prefill ${result.prefillMs.toFixed(0)} ms)`);
```

## Development & Testing

```bash
npm install
npm run build
npm test              # Unit tests (quantization, FWHT-512, GGUF parser, Range loader)
npm run test:webgpu   # Headless Chrome WebGPU kernel & pipeline tests
```

To run the browser demo locally:
```bash
npx serve .
# Open http://localhost:3000/examples/
```
