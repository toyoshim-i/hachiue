# 🪴 hachiue (鉢植え)

Zero-dependency, mobile-optimized WebGPU inference engine for **PrismML Bonsai 1 (1-bit `Q1_0`)** and **Bonsai 2 (Ternary 1.58-bit `PTQ1_0` / `PQ2_0`)** GGUF models.

## Key Features

- **GGUF Direct (Zero Offline Conversion)**: Point `HachiueEngine.create({ url })` directly at any `.gguf` URL on Hugging Face. Includes a zero-dependency GGUF BPE tokenizer extracted straight from the GGUF header.
- **Crash-Free Mobile Streaming & Resumable Cache**:
  - Replaces `Response.clone()` and monolithic `CacheStorage.put()` with bounded **16 MiB HTTP `Range` requests** persisted chunk-by-chunk to **OPFS (Origin Private File System)** (with automatic IndexedDB chunk fallback on iOS Safari main thread).
  - Enforces WebGPU staging backpressure (`await device.queue.onSubmittedWorkDone()`) every 16 MiB so peak RAM never spikes on iOS Safari or Android Chrome.
- **Zero-Inflation Planar VRAM Layout**:
  - Keeps `Q1_0` (18B / 128 weights), `PTQ1_0` (28B / 128 weights), and `PQ2_0` (34B / 128 weights) at their exact native GGUF footprint in GPU memory.
  - Packs `fp16` block scales into `u32` pairs unpacked via WGSL `unpack2x16float` (saving ~11% VRAM without requiring `shader-f16`).
  - Shares `token_embd.weight` and `output.weight` (`lm_head`) GPU buffers when `tie_word_embeddings` is enabled, and automatically splits large tensors across the 128 MiB mobile `maxStorageBufferBindingSize` boundary.
- **Bonsai 2 Support**:
  - Native `PTQ1_0` (base-3 dense trits) and `PQ2_0` multi-row WGSL compute kernels.
  - Workgroup-shared Fast Walsh-Hadamard Transform (`hadamard_512`) for `prism.hadamard` folded weights and latent embeddings.
  - Qwen3.5 Hybrid Attention (3:1 Gated DeltaNet recurrent linear attention + gated full attention with `Q8_0` KV cache).

## Quick Start

```bash
npm install
npm run build
npm test
```

To try the browser demo locally:
```bash
npx serve .
# Open http://localhost:3000/examples/
```
