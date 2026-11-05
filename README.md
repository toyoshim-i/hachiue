# 🪴 hachiue (鉢植え)

Zero-dependency, mobile-optimized WebGPU LLM & translation inference engine supporting:
- **Tier 1 (Core WebGPU LLM Engine)**:
  - **PrismML Bonsai 1** (`qwen3` dense GQA, 1-bit `Q1_0` and ternary `PQ2_0`).
  - **PrismML Bonsai 2** (`qwen35` 3:1 Gated DeltaNet hybrid linear attention + Hadamard-rotated ternary `PTQ1_0` / `PQ2_0`).
  - **PFN PLaMo 3 Translate** (`plamo3` 0.8B / 2B, Sliding Window Attention + YaRN RoPE + Unigram DP tokenizer, standard GGML `Q4_0`, `Q4_1`, `Q4_K`, `Q6_K`).
- **Tier 2 (Model-Agnostic Translation Helpers)**:
  - `translateText(engine, options)`: Sentence, paragraph, and multi-line document translation with optional `context`, `glossary`, `form` (`'polite' | 'plain'`), and multi-turn `history`.
  - `translateTerm(engine, options)`: Concise dictionary / short-phrase lookup with optional `context` disambiguation and automatic fallback.

👉 **[Live Demo — Translate & Chat (GitHub Pages)](https://toyoshim-i.github.io/hachiue/)**

## Key Features

- **GGUF Direct (Zero Offline Conversion)**:
  - Point `HachiueEngine.create({ url })` directly at any supported `.gguf` URL on Hugging Face.
  - Includes a zero-dependency GGUF v3 parser plus both GPT-2/Qwen BPE and PLaMo-3 Aho-Corasick / Unigram Trie DP tokenizers extracted straight from the GGUF header.
- **Double-Buffered Mobile Streaming & Resumable Cache**:
  - Replaces `Response.clone()` and monolithic `CacheStorage.put()` with bounded **16 MiB HTTP `Range` requests** persisted chunk-by-chunk to **OPFS (Origin Private File System)** (with automatic IndexedDB chunk fallback on iOS WebKit).
  - Uses a **2-slot double-buffered pipeline** that overlaps downloading chunk `N + 1` with writing chunk `N` to local cache and uploading tensors to WebGPU (peak chunk RAM $\le$ 32 MiB).
  - Enforces WebGPU queue backpressure (`await device.queue.onSubmittedWorkDone()`) and macrotask GC yields every 16 MiB so iOS Jetsam limits are never exceeded.
- **Zero-Inflation VRAM Layout & Zero-Allocation Runtime**:
  - Keeps both PrismML ultra-low-bit formats (`Q1_0`, `PTQ1_0`, `PQ2_0`) and standard GGML formats (`Q4_0`, `Q4_1`, `Q4_K`, `Q6_K`) at their exact native GGUF footprint in GPU memory.
  - Shares `token_embd.weight` and `output.weight` (`lm_head`) GPU buffers when `tie_word_embeddings` is enabled, and automatically splits large tensors across `64 MiB` storage buffer chunks.
  - Reuses a single `1.125 MiB` CPU staging buffer pair during loading and a pre-allocated GPU scratch buffer pool during inference (**zero dynamic GPU/CPU buffer allocations per token**).
- **Tier 2 Translation Helpers (`translateText` & `translateTerm`)**:
  - Absorbs prompt templates (PFML3 `<|plamo:tag|>` vs ChatML `<|im_start|>`), stop tokens, language code normalization, reasoning tag stripping, and target script validation across PLaMo 3 Translate and Bonsai 1 / 2.

## Supported Models & Benchmarks

### Model Lineup & Measured Speed (`hachiue` WebGPU)

| Model | Base Arch | Format | bpw | File Size | iPhone 17 Pro Max (Chrome) |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **Bonsai 1.7B** | Qwen3-1.7B | `Q1_0` (1-bit) | 1.125 | **237 MiB** | **51.0 tok/s** *(prefill 187 ms)* |
| **Ternary Bonsai 1.7B** | Qwen3-1.7B | `PQ2_0` (Ternary) | 2.125 | **420 MiB** | **42.0 tok/s** *(prefill 228 ms)* |
| **Bonsai 2 2B** | Qwen3.5-2B | `PTQ1_0` (Ternary) | 1.750 | **406 MiB** | **25.4 tok/s** *(prefill 417 ms)* |
| **Bonsai 2 2B** | Qwen3.5-2B | `PQ2_0` (Ternary) | 2.125 | **495 MiB** | **44.2 tok/s** *(prefill 243 ms)* |
| **Bonsai 4B** | Qwen3-4B | `Q1_0` (1-bit) | 1.125 | **546 MiB** | **30.7 tok/s** *(prefill 339 ms)* |
| **Ternary Bonsai 4B** | Qwen3-4B | `PQ2_0` (Ternary) | 2.125 | **1,025 MiB** | **28.9 tok/s** *(prefill 451 ms)* |
| **Bonsai 2 4B** | Qwen3.5-4B | `PTQ1_0` (Ternary) | 1.750 | **902 MiB** | **13.3 tok/s** *(prefill 836 ms)* |
| **Bonsai 2 4B** | Qwen3.5-4B | `PQ2_0` (Ternary) | 2.125 | **1,093 MiB** | **20.6 tok/s** *(prefill 446 ms)* |
| **Bonsai 8B** | Qwen3-8B | `Q1_0` (1-bit) | 1.125 | **1,100 MiB** | **26.4 tok/s** *(prefill 478 ms)* |
| **PLaMo 3 Translate 0.8B** | `plamo3` (SWA + YaRN) | `Q4_0` / `Q6_K` | ~4.3 | **522 MiB** | **33.9 tok/s** *(prefill 722 ms)* |
| **PLaMo 3 Translate 0.8B** | `plamo3` (SWA + YaRN) | `Q4_K_M` | ~4.5 | **546 MiB** | **31.0 tok/s** *(prefill 852 ms)* |
| **PLaMo 3 Translate 2B** | `plamo3` (SWA + YaRN) | `Q4_0` / `Q6_K` | ~4.3 | **1,620 MiB** | **12.7 tok/s** *(prefill 2695 ms)* |
| **PLaMo 3 Translate 2B** | `plamo3` (SWA + YaRN) | `Q4_K_M` | ~4.6 | **1,723 MiB** | **10.9 tok/s** *(prefill 3139 ms)* |

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

## Installation & Usage

```bash
npm install hachiue
```

Or import directly in the browser via CDN (zero build step):
```html
<script type="module">
  import { HachiueEngine, translateText, translateTerm } from 'https://cdn.jsdelivr.net/npm/hachiue/+esm';
</script>
```

### Tier 1: Generic LLM Inference (`HachiueEngine`)

```ts
import { HachiueEngine, prefetchModel } from 'hachiue';

const engine = await HachiueEngine.create({
  url: 'https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/resolve/main/plamo-3-translate-0.8b-Q4_0.gguf',
  onProgress: (p) => {
    if (p.totalBytes) console.log(`${p.phase}: ${(p.loadedBytes / 1048576).toFixed(1)} MiB`);
  },
});

const reply = await engine.generate(
  [{ role: 'user', content: 'Small trees thrive in small pots.' }],
  {
    maxNewTokens: 256,
    temperature: 0,
    onToken: (_id, delta) => process.stdout.write(delta),
  },
);
```

### Tier 2: Model-Agnostic Translation Helpers (`translateText` & `translateTerm`)

Works seamlessly across **PLaMo 3 Translate** (`plamo3`), **Bonsai 1** (`qwen3`), and **Bonsai 2** (`qwen35`):

```ts
import { HachiueEngine, translateText, translateTerm } from 'hachiue';

// 1. Document / Sentence Translation
const docResult = await translateText(engine, {
  text: 'Small trees thrive in small pots.',
  sourceLang: 'en',
  targetLang: 'ja',
  form: 'polite', // 'polite' (です・ます) | 'plain' (だ・である)
  glossary: [{ source: 'pots', target: '植木鉢' }],
});
console.log(docResult.translation);

// 2. Word / Short-Phrase Dictionary Lookup with Optional Context
const termResult = await translateTerm(engine, {
  term: 'bank',
  context: 'The river bank was overflowing after the heavy rain.',
  sourceLang: 'en',
  targetLang: 'ja',
});
console.log(termResult.translation); // -> "岸辺" (PLaMo 3) / "川岸" (Bonsai 2)
```

## Development & Testing

```bash
npm install
npm run build
npm test              # Unit tests (quantization, FWHT-512, GGUF parser, tokenizers, translation helpers)
npm run test:webgpu   # Headless Chrome WebGPU kernel & pipeline tests
```

To run the browser demo locally:
```bash
npx serve .
# Open http://localhost:3000/examples/
```
