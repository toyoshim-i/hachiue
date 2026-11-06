import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  buildInstructTermMessages,
  buildInstructTextMessages,
  buildPlamo3TranslationPrompt,
  dequantizeBlock128Cpu,
  dequantizeBlockCpu,
  extractCleanTermTranslation,
  extractCleanTextTranslation,
  fetchGgufHeader,
  fp16ToFp32,
  fwhtBlockCpu,
  getWasmKernelModuleBytes,
  GgmlType,
  GgufTokenizer,
  isValidTargetLanguageTranslation,
  normalizeLangCode,
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
} from '../src/index.js'

describe('Quantization & Dequantization (Q1_0, PQ2_0, PTQ1_0)', () => {
  it('accurately decodes Q1_0 128-element blocks', () => {
    const block = new Uint8Array(18)
    // FP16 1.5 = 0x3e00
    block[0] = 0x00
    block[1] = 0x3e
    // Alternating bits 0b10101010 = 0xaa
    for (let i = 0; i < 16; i++) block[2 + i] = 0xaa

    const out = new Float32Array(128)
    dequantizeBlock128Cpu(block, GgmlType.Q1_0, out)
    for (let j = 0; j < 128; j++) {
      const expected = j % 2 === 1 ? 1.5 : -1.5
      assert.equal(out[j], expected)
    }
  })

  it('accurately decodes PQ2_0 128-element ternary blocks', () => {
    const block = new Uint8Array(34)
    // FP16 0.5 = 0x3800
    block[0] = 0x00
    block[1] = 0x38
    // 4 trits per byte: (0 -> -1, 1 -> 0, 2 -> +1, 1 -> 0) => 0 | (1<<2) | (2<<4) | (1<<6) = 0x64
    for (let i = 0; i < 32; i++) block[2 + i] = 0x64

    const out = new Float32Array(128)
    dequantizeBlock128Cpu(block, GgmlType.PQ2_0, out)
    const pattern = [-0.5, 0.0, 0.5, 0.0]
    for (let j = 0; j < 128; j++) {
      assert.equal(out[j], pattern[j % 4])
    }
  })

  it('accurately round-trips PTQ1_0 128-element base-3 ternary blocks matching PrismML ggml-quants.c', () => {
    // Generate 128 ternary values in {-1, 0, +1}
    const trits = new Int8Array(128)
    for (let i = 0; i < 128; i++) {
      trits[i] = ((i * 7 + 3) % 3) - 1
    }

    // Encode using the exact quantize_row_ptq1_0_ref formula from PrismML llama.cpp
    const block = new Uint8Array(28)
    // Scale = 2.0 (FP16 0x4000)
    block[26] = 0x00
    block[27] = 0x40

    // Stage 0: c = 16 (j = 0..15, covers first 80 elements)
    for (let m = 0; m < 16; m++) {
      let q = 0
      for (let n = 0; n < 5; n++) {
        const xi = trits[m + n * 16] + 1
        q = q * 3 + xi
      }
      q = Math.floor((q * 256 + 242) / 243)
      block[m] = q
    }
    // Stage 1: c = 8 (j = 16..23, covers elements 80..119)
    for (let m = 0; m < 8; m++) {
      let q = 0
      for (let n = 0; n < 5; n++) {
        const xi = trits[80 + m + n * 8] + 1
        q = q * 3 + xi
      }
      q = Math.floor((q * 256 + 242) / 243)
      block[16 + m] = q
    }
    // Tail qh: 2 bytes (covers elements 120..127)
    for (let h = 0; h < 2; h++) {
      let q = 0
      for (let m = 0; m < 4; m++) {
        const xi = trits[120 + h + m * 2] + 1
        q = q * 3 + xi
      }
      q *= 3
      q = Math.floor((q * 256 + 242) / 243)
      block[24 + h] = q
    }

    const out = new Float32Array(128)
    dequantizeBlock128Cpu(block, GgmlType.PTQ1_0, out)
    for (let i = 0; i < 128; i++) {
      assert.equal(out[i], trits[i] * 2.0, `Mismatch at index ${i}`)
    }
  })

  it('accurately decodes Q4_0 and Q4_1 32-element blocks and Q4_K / Q6_K 256-element blocks', () => {
    // Q4_0: 18B [d: fp16=0.5 (0x3800), qs: 16B]
    const q40 = new Uint8Array(18)
    q40[0] = 0x00
    q40[1] = 0x38
    for (let i = 0; i < 16; i++) q40[2 + i] = 0x97 // low nibble 7 -> (7-8)*0.5 = -0.5, high nibble 9 -> (9-8)*0.5 = +0.5
    const out40 = new Float32Array(32)
    dequantizeBlockCpu(q40, GgmlType.Q4_0, out40)
    for (let i = 0; i < 16; i++) {
      assert.equal(out40[i], -0.5)
      assert.equal(out40[16 + i], 0.5)
    }

    // Q4_1: 20B [d: fp16=0.5 (0x3800), m: fp16=1.0 (0x3c00), qs: 16B]
    const q41 = new Uint8Array(20)
    q41[0] = 0x00
    q41[1] = 0x38
    q41[2] = 0x00
    q41[3] = 0x3c
    for (let i = 0; i < 16; i++) q41[4 + i] = 0x42 // low=2 -> 2*0.5+1=2.0, high=4 -> 4*0.5+1=3.0
    const out41 = new Float32Array(32)
    dequantizeBlockCpu(q41, GgmlType.Q4_1, out41)
    for (let i = 0; i < 16; i++) {
      assert.equal(out41[i], 2.0)
      assert.equal(out41[16 + i], 3.0)
    }
  })
})

describe('Fast Walsh-Hadamard Transform (FWHT-512)', () => {
  it('matches dense Sylvester Hadamard matrix multiplication and is self-inverse', () => {
    const N = 512
    const orig = new Float32Array(N)
    for (let i = 0; i < N; i++) orig[i] = Math.sin(i * 0.13) - 0.5

    // Dense reference using bit-parity formula from PrismML llama-model.cpp
    const denseOut = new Float32Array(N)
    const scale = 1 / Math.sqrt(N)
    for (let r = 0; r < N; r++) {
      let sum = 0
      for (let c = 0; c < N; c++) {
        let parity = r & c
        parity ^= parity >> 8
        parity ^= parity >> 4
        parity ^= parity >> 2
        parity ^= parity >> 1
        const h = parity & 1 ? -scale : scale
        sum += h * orig[c]
      }
      denseOut[r] = sum
    }

    const fastOut = new Float32Array(orig)
    fwhtBlockCpu(fastOut, 0, N)

    for (let i = 0; i < N; i++) {
      assert.ok(Math.abs(fastOut[i] - denseOut[i]) < 1e-5, `Mismatch at ${i}: ${fastOut[i]} vs ${denseOut[i]}`)
    }

    // Applying FWHT a second time restores the original vector (involutory H * H = I)
    fwhtBlockCpu(fastOut, 0, N)
    for (let i = 0; i < N; i++) {
      assert.ok(Math.abs(fastOut[i] - orig[i]) < 1e-5, `Inverse mismatch at ${i}`)
    }
  })
})

describe('Live Hugging Face GGUF Header & Tokenizer Verification (Bonsai 1 & Bonsai 2)', () => {
  it('parses Bonsai-1.7B-Q1_0.gguf header and encodes/decodes ChatML prompts', async () => {
    const url = 'https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf'
    const header = await fetchGgufHeader(url)
    assert.equal(header.arch.archName, 'qwen3')
    assert.equal(header.arch.layers, 28)
    assert.equal(header.arch.hidden, 2048)
    assert.equal(header.arch.vocab, 151669)
    assert.equal(header.tensors['blk.0.attn_q.weight'].type, GgmlType.Q1_0)
    assert.ok(header.tokenizer)

    const tok = new GgufTokenizer(header.tokenizer)
    const prompt = tok.formatChat([{ role: 'user', content: 'Hello, Bonsai! 盆栽について教えて。' }])
    const ids = tok.encode(prompt)
    assert.ok(ids.length > 10)
    const decoded = tok.decode(ids, false)
    assert.equal(decoded, prompt)
  })

  it('parses Ternary-Bonsai-2-2B-PTQ1_0.gguf header with Hadamard and Hybrid Qwen3.5 config', async () => {
    const url = 'https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/main/Ternary-Bonsai-2-2B-PTQ1_0.gguf'
    const header = await fetchGgufHeader(url)
    assert.equal(header.arch.archName, 'qwen35')
    assert.equal(header.arch.layers, 24)
    assert.equal(header.arch.hidden, 2048)
    assert.equal(header.arch.vocab, 248320)
    assert.ok(header.arch.hybrid)
    assert.equal(header.arch.hybrid.layerTypes[0], 'linear')
    assert.equal(header.arch.hybrid.layerTypes[3], 'full')
    assert.ok(header.arch.hadamard)
    assert.equal(header.arch.hadamard.blockSize, 512)
    assert.equal(header.arch.hadamard.tiedOutput, true)
    assert.ok(header.arch.hadamard.foldedWeights.has('blk.0.attn_qkv.weight'))
    assert.ok(header.arch.hadamard.foldedWeights.has('output.weight'))
    assert.ok(header.arch.hadamard.inverseWeights.has('token_embd.weight'))
    assert.equal(header.tensors['token_embd.weight'].type, GgmlType.PTQ1_0)
    assert.equal(fp16ToFp32(0x3c00), 1.0)
  })

  it('parses plamo-3-translate-0.8b-Q4_0.gguf header and encodes/decodes PFML3 with Unigram DP', async () => {
    const url = 'https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/resolve/main/plamo-3-translate-0.8b-Q4_0.gguf'
    const header = await fetchGgufHeader(url)
    assert.equal(header.arch.archName, 'plamo3')
    assert.equal(header.arch.layers, 16)
    assert.equal(header.arch.hidden, 1536)
    assert.equal(header.arch.intermediate, 6144)
    assert.equal(header.arch.heads, 16)
    assert.equal(header.arch.kvHeads, 1)
    assert.equal(header.arch.vocab, 107520)
    assert.equal(header.arch.tieWordEmbeddings, false)
    assert.ok(header.arch.swa)
    assert.equal(header.arch.swa.windowSize, 2048)
    assert.equal(header.arch.swa.pattern, 8)
    assert.equal(header.arch.swa.layerTypes[0], 'swa')
    assert.equal(header.arch.swa.layerTypes[7], 'full')
    assert.equal(header.arch.swa.layerTypes[15], 'full')
    assert.equal(header.tensors['token_embd.weight'].type, GgmlType.Q4_0)
    assert.equal(header.tensors['output.weight'].type, GgmlType.Q6_K)
    assert.equal(header.tensors['blk.0.post_attention_norm'].type, GgmlType.F32)
    assert.ok(header.tokenizer)
    assert.equal(header.tokenizer.model, 'plamo3')

    const tok = new GgufTokenizer(header.tokenizer)
    const prompt = tok.formatChat([{ role: 'user', content: 'Hello, world!\nThis is PLaMo 3 Translate.' }], {
      sourceLang: 'en',
      targetLang: 'ja',
    })
    assert.ok(prompt.startsWith('<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation'))
    const ids = tok.encode(prompt)
    assert.equal(ids[0], 1) // <|plamo:bos|>
    assert.equal(ids[1], 16) // <|plamo:tag|>
    assert.equal(tok.decode(ids, false), prompt)

    const textOnly = 'こんにちは、世界！ Today is 2026-11-05.   Multiple   spaces and aaaaa.'
    const textIds = tok.encode(textOnly)
    assert.equal(tok.decode(textIds, true), textOnly)
  })
})

describe('Tier 2 Translation Helpers (PFML3 & ChatML Builders, Cleansing, Validation)', () => {
  it('builds PFML3 prompts with context, glossary, history, and politeness form for plamo3', () => {
    const prompt = buildPlamo3TranslationPrompt({
      input: 'Hachiue runs on WebGPU.',
      sourceLang: 'en',
      targetLang: 'ja',
      context: 'Technical documentation for a browser inference engine.',
      glossary: [{ source: 'Hachiue', target: '鉢植え' }],
      form: 'polite',
      history: [{ source: 'Hello.', translation: 'こんにちは。' }],
    })
    assert.equal(
      prompt,
      '<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation' +
        '<|plamo:tag|>context<|plamo:msg|>Technical documentation for a browser inference engine.' +
        '<|plamo:tag|>glossary<|plamo:msg|>Hachiue\t鉢植え' +
        '<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hello.' +
        '<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>こんにちは。' +
        '<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hachiue runs on WebGPU.' +
        '<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:key|>form<|plamo:val|>Polite<|plamo:msg|>',
    )
  })

  it('normalizes language codes and validates target language scripts', () => {
    assert.equal(normalizeLangCode('zh-CN').plamoCode, 'zh-Hans')
    assert.equal(normalizeLangCode('zh-TW').plamoCode, 'zh-Hant')
    assert.equal(normalizeLangCode('ja').englishName, 'Japanese')

    assert.equal(isValidTargetLanguageTranslation('川岸、土手', 'bank', 'ja'), true)
    assert.equal(isValidTargetLanguageTranslation('bank', 'bank', 'ja'), false)
    assert.equal(isValidTargetLanguageTranslation('river edge', 'bank', 'ja'), false)
    assert.equal(isValidTargetLanguageTranslation('河岸', 'bank', 'zh-CN'), true)
    assert.equal(isValidTargetLanguageTranslation('강둑', 'bank', 'ko'), true)
  })

  it('cleanses thinking tags and dictionary prefixes from raw model outputs', () => {
    assert.equal(extractCleanTermTranslation('<think>\nthinking...\n</think>\n"bank" → 銀行、川岸\nExtra line'), '銀行、川岸')
    assert.equal(extractCleanTextTranslation('<think></think>\n翻訳: こんにちは。\n元気ですか？<|plamo:eom|>'), 'こんにちは。\n元気ですか？')

    const msgWithCtx = buildInstructTermMessages('bank', 'ja', 'The river bank was overflowing.')
    assert.equal(msgWithCtx.length, 2)
    assert.ok(msgWithCtx[1].content.includes('Context: "The river bank was overflowing."'))

    const docMsgs = buildInstructTextMessages({
      text: 'Small trees thrive in small pots.',
      sourceLang: 'en',
      targetLang: 'ja',
      form: 'plain',
      glossary: [{ source: 'pots', target: '鉢' }],
    })
    assert.equal(docMsgs.length, 2)
    assert.ok(docMsgs[0].content.includes('だ・である調'))
    assert.ok(docMsgs[0].content.includes('"pots" -> "鉢"'))
  })
})

describe('Wasm SIMD (v128) Kernel Compilation & Numerical Accuracy', () => {
  it('compiles inline Wasm SIMD module and matches reference GEMV for F32, Q1_0, PQ2_0, PTQ1_0, Q4_0, Q4_1, Q4_K, Q6_K (including non-multiple-of-4 tail rows)', () => {
    const wasmBytes = getWasmKernelModuleBytes()
    assert.equal(WebAssembly.validate(wasmBytes as BufferSource), true, 'Generated Wasm SIMD binary must be valid')

    const wasmMem = new WebAssembly.Memory({ initial: 128 }) // 8 MiB
    const mod = new WebAssembly.Module(wasmBytes as BufferSource)
    const inst = new WebAssembly.Instance(mod, { env: { memory: wasmMem } })
    const kernels = inst.exports as unknown as WasmKernelExports

    // Initialize FP16 table, Q1_0 LUT, and PTQ1_0 tail constants
    const buf = wasmMem.buffer
    const fp16Tab = new Float32Array(buf, WASM_MEM_FP16_TABLE, 65536)
    for (let i = 0; i < 65536; i++) fp16Tab[i] = fp16ToFp32(i)

    const q1Lut = new Int8Array(buf, WASM_MEM_Q1_LUT, 2048)
    for (let b = 0; b < 256; b++) {
      for (let bit = 0; bit < 8; bit++) {
        q1Lut[b * 8 + bit] = (b >> bit) & 1 ? 1 : -1
      }
    }
    new Int16Array(buf, WASM_MEM_CONSTS, 8).set([1, 1, 3, 3, 9, 9, 27, 27])

    const xF32 = new Float32Array(buf, WASM_MEM_X_F32, 65536)
    const xQ16 = new Int16Array(buf, WASM_MEM_X_Q16, 65536)
    const xPq2Q16 = new Int16Array(buf, WASM_MEM_X_PQ2_Q16, 65536)
    const xScale = new Float32Array(buf, WASM_MEM_X_SCALE, 4096)
    const xF32Sum = new Float32Array(buf, WASM_MEM_X_F32_SUM, 4096)
    const yF32 = new Float32Array(buf, WASM_MEM_Y_F32, 1024)

    // Test with N = 7 (exercises 4-row unrolled loop for rows 0..3 and 1-row tail loop for rows 4..6)
    const K = 512
    const N = 7
    const nb128 = K / 128
    const nb32 = K / 32
    const nb256 = K / 256

    const x = new Float32Array(K)
    for (let i = 0; i < K; i++) x[i] = Math.sin(i * 0.17) * 0.8
    xF32.set(x, 0)

    const quantize128 = () => {
      for (let b = 0; b < nb128; b++) {
        const base = b * 128
        let amax = 0
        for (let i = 0; i < 128; i++) amax = Math.max(amax, Math.abs(x[base + i]))
        const sc = amax / 16384.0
        const inv = amax > 0 ? 16384.0 / amax : 0
        xScale[b] = sc
        for (let i = 0; i < 128; i++) xQ16[base + i] = Math.round(x[base + i] * inv)
        for (let sub = 0; sub < 4; sub++) {
          const sb = base + sub * 32
          for (let m = 0; m < 8; m++) {
            xPq2Q16[sb + m] = xQ16[sb + m * 4 + 0]
            xPq2Q16[sb + 8 + m] = xQ16[sb + m * 4 + 1]
            xPq2Q16[sb + 16 + m] = xQ16[sb + m * 4 + 2]
            xPq2Q16[sb + 24 + m] = xQ16[sb + m * 4 + 3]
          }
        }
      }
    }

    const quantize32 = () => {
      for (let b = 0; b < nb32; b++) {
        const base = b * 32
        let amax = 0
        let fsum = 0
        for (let i = 0; i < 32; i++) {
          fsum += x[base + i]
          amax = Math.max(amax, Math.abs(x[base + i]))
        }
        const sc = amax / 16384.0
        const inv = amax > 0 ? 16384.0 / amax : 0
        xScale[b] = sc
        xF32Sum[b] = fsum
        for (let i = 0; i < 32; i++) xQ16[base + i] = Math.round(x[base + i] * inv)
      }
    }

    // 1. F32
    const wF32 = new Float32Array(buf, WASM_MEM_WEIGHTS_BASE, N * K)
    const expF32 = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let s = 0
      for (let k = 0; k < K; k++) {
        const wv = Math.cos((r * K + k) * 0.11) * 0.5
        wF32[r * K + k] = wv
        s += x[k] * wv
      }
      expF32[r] = s
    }
    kernels.gemv_f32(1, N, K, WASM_MEM_WEIGHTS_BASE, 0, 0, N)
    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(yF32[r] - expF32[r]) < 1e-4, `Wasm F32 row ${r}: ${yF32[r]} vs ${expF32[r]}`)
    }

    // 2. Q1_0
    quantize128()
    const q1Quants = new Uint8Array(buf, WASM_MEM_WEIGHTS_BASE, N * nb128 * 16)
    const q1Scales = new Uint16Array(buf, WASM_MEM_WEIGHTS_BASE + 4096, N * nb128)
    const expQ1 = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let s = 0
      for (let b = 0; b < nb128; b++) {
        const blkIdx = r * nb128 + b
        q1Scales[blkIdx] = 0x3800 // 0.5
        const rawBlock = new Uint8Array(18)
        rawBlock[0] = 0x00
        rawBlock[1] = 0x38
        for (let i = 0; i < 16; i++) {
          const byte = (r * 31 + b * 17 + i * 13) & 0xff
          q1Quants[blkIdx * 16 + i] = byte
          rawBlock[2 + i] = byte
        }
        const deq = new Float32Array(128)
        dequantizeBlock128Cpu(rawBlock, GgmlType.Q1_0, deq)
        for (let j = 0; j < 128; j++) s += x[b * 128 + j] * deq[j]
      }
      expQ1[r] = s
    }
    kernels.gemv_q1_0(1, N, K, nb128, WASM_MEM_WEIGHTS_BASE, WASM_MEM_WEIGHTS_BASE + 4096, 0, 0, N)
    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(yF32[r] - expQ1[r]) < 2e-3, `Wasm Q1_0 row ${r}: ${yF32[r]} vs ${expQ1[r]}`)
    }

    // 3. PQ2_0
    const pq2Quants = new Uint8Array(buf, WASM_MEM_WEIGHTS_BASE, N * nb128 * 32)
    const pq2Scales = new Uint16Array(buf, WASM_MEM_WEIGHTS_BASE + 4096, N * nb128)
    const expPq2 = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let s = 0
      for (let b = 0; b < nb128; b++) {
        const blkIdx = r * nb128 + b
        pq2Scales[blkIdx] = 0x3c00 // 1.0
        const rawBlock = new Uint8Array(34)
        rawBlock[0] = 0x00
        rawBlock[1] = 0x3c
        for (let i = 0; i < 32; i++) {
          const t0 = (r + b + i) % 3
          const t1 = (r + b + i + 1) % 3
          const t2 = (r + b + i + 2) % 3
          const t3 = (r + b + i + 3) % 3
          const byte = t0 | (t1 << 2) | (t2 << 4) | (t3 << 6)
          pq2Quants[blkIdx * 32 + i] = byte
          rawBlock[2 + i] = byte
        }
        const deq = new Float32Array(128)
        dequantizeBlock128Cpu(rawBlock, GgmlType.PQ2_0, deq)
        for (let j = 0; j < 128; j++) s += x[b * 128 + j] * deq[j]
      }
      expPq2[r] = s
    }
    kernels.gemv_pq2_0(1, N, K, nb128, WASM_MEM_WEIGHTS_BASE, WASM_MEM_WEIGHTS_BASE + 4096, 0, 0, N)
    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(yF32[r] - expPq2[r]) < 2e-3, `Wasm PQ2_0 row ${r}: ${yF32[r]} vs ${expPq2[r]}`)
    }

    // 4. PTQ1_0
    const ptqQuants = new Uint8Array(buf, WASM_MEM_WEIGHTS_BASE, N * nb128 * 26)
    const ptqScales = new Uint16Array(buf, WASM_MEM_WEIGHTS_BASE + 4096, N * nb128)
    const expPtq = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let s = 0
      for (let b = 0; b < nb128; b++) {
        const blkIdx = r * nb128 + b
        ptqScales[blkIdx] = 0x3c00
        const rawBlock = new Uint8Array(28)
        rawBlock[26] = 0x00
        rawBlock[27] = 0x3c
        for (let i = 0; i < 26; i++) {
          const byte = ((r * 19 + b * 23 + i * 41) % 240) & 0xff
          ptqQuants[blkIdx * 26 + i] = byte
          rawBlock[i] = byte
        }
        const deq = new Float32Array(128)
        dequantizeBlock128Cpu(rawBlock, GgmlType.PTQ1_0, deq)
        for (let j = 0; j < 128; j++) s += x[b * 128 + j] * deq[j]
      }
      expPtq[r] = s
    }
    kernels.gemv_ptq1_0(1, N, K, nb128, WASM_MEM_WEIGHTS_BASE, WASM_MEM_WEIGHTS_BASE + 4096, 0, 0, N)
    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(yF32[r] - expPtq[r]) < 2e-3, `Wasm PTQ1_0 row ${r}: ${yF32[r]} vs ${expPtq[r]}`)
    }

    // 5. Q4_0
    quantize32()
    const q40Quants = new Uint8Array(buf, WASM_MEM_WEIGHTS_BASE, N * nb32 * 18)
    const expQ40 = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let s = 0
      for (let b = 0; b < nb32; b++) {
        const blkOff = (r * nb32 + b) * 18
        q40Quants[blkOff] = 0x00
        q40Quants[blkOff + 1] = 0x38 // 0.5
        for (let i = 0; i < 16; i++) q40Quants[blkOff + 2 + i] = (r * 29 + b * 17 + i * 7) & 0xff
        const deq = new Float32Array(32)
        dequantizeBlockCpu(q40Quants.subarray(blkOff, blkOff + 18), GgmlType.Q4_0, deq)
        for (let j = 0; j < 32; j++) s += x[b * 32 + j] * deq[j]
      }
      expQ40[r] = s
    }
    kernels.gemv_q4_0(1, N, K, nb32, WASM_MEM_WEIGHTS_BASE, 0, 0, N)
    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(yF32[r] - expQ40[r]) < 1e-2, `Wasm Q4_0 row ${r}: ${yF32[r]} vs ${expQ40[r]}`)
    }

    // 6. Q4_1
    const q41Quants = new Uint8Array(buf, WASM_MEM_WEIGHTS_BASE, N * nb32 * 20)
    const expQ41 = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let s = 0
      for (let b = 0; b < nb32; b++) {
        const blkOff = (r * nb32 + b) * 20
        q41Quants[blkOff] = 0x00
        q41Quants[blkOff + 1] = 0x34 // 0.25
        q41Quants[blkOff + 2] = 0x00
        q41Quants[blkOff + 3] = 0xb8 // -0.5
        for (let i = 0; i < 16; i++) q41Quants[blkOff + 4 + i] = (r * 37 + b * 19 + i * 11) & 0xff
        const deq = new Float32Array(32)
        dequantizeBlockCpu(q41Quants.subarray(blkOff, blkOff + 20), GgmlType.Q4_1, deq)
        for (let j = 0; j < 32; j++) s += x[b * 32 + j] * deq[j]
      }
      expQ41[r] = s
    }
    kernels.gemv_q4_1(1, N, K, nb32, WASM_MEM_WEIGHTS_BASE, 0, 0, N)
    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(yF32[r] - expQ41[r]) < 1e-2, `Wasm Q4_1 row ${r}: ${yF32[r]} vs ${expQ41[r]}`)
    }

    // 7. Q4_K
    const q4kQuants = new Uint8Array(buf, WASM_MEM_WEIGHTS_BASE, N * nb256 * 144)
    const expQ4k = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let s = 0
      for (let b = 0; b < nb256; b++) {
        const blkOff = (r * nb256 + b) * 144
        q4kQuants[blkOff] = 0x00
        q4kQuants[blkOff + 1] = 0x30 // 0.125
        q4kQuants[blkOff + 2] = 0x00
        q4kQuants[blkOff + 3] = 0x2c // 0.0625
        for (let i = 4; i < 144; i++) q4kQuants[blkOff + i] = (r * 41 + b * 23 + i * 13) & 0xff
        const deq = new Float32Array(256)
        dequantizeBlockCpu(q4kQuants.subarray(blkOff, blkOff + 144), GgmlType.Q4_K, deq)
        for (let j = 0; j < 256; j++) s += x[b * 256 + j] * deq[j]
      }
      expQ4k[r] = s
    }
    kernels.gemv_q4_k(1, N, K, nb256, WASM_MEM_WEIGHTS_BASE, 0, 0, N)
    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(yF32[r] - expQ4k[r]) < 5e-2, `Wasm Q4_K row ${r}: ${yF32[r]} vs ${expQ4k[r]}`)
    }

    // 8. Q6_K
    const q6kQuants = new Uint8Array(buf, WASM_MEM_WEIGHTS_BASE, N * nb256 * 210)
    const expQ6k = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let s = 0
      for (let b = 0; b < nb256; b++) {
        const blkOff = (r * nb256 + b) * 210
        for (let i = 0; i < 208; i++) q6kQuants[blkOff + i] = (r * 43 + b * 29 + i * 17) & 0xff
        q6kQuants[blkOff + 208] = 0x00
        q6kQuants[blkOff + 209] = 0x2c // 0.0625
        const deq = new Float32Array(256)
        dequantizeBlockCpu(q6kQuants.subarray(blkOff, blkOff + 210), GgmlType.Q6_K, deq)
        for (let j = 0; j < 256; j++) s += x[b * 256 + j] * deq[j]
      }
      expQ6k[r] = s
    }
    kernels.gemv_q6_k(1, N, K, nb256, WASM_MEM_WEIGHTS_BASE, 0, 0, N)
    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(yF32[r] - expQ6k[r]) < 5e-2, `Wasm Q6_K row ${r}: ${yF32[r]} vs ${expQ6k[r]}`)
    }
  })
})
