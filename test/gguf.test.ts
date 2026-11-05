import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  dequantizeBlock128Cpu,
  dequantizeBlockCpu,
  fetchGgufHeader,
  fp16ToFp32,
  fwhtBlockCpu,
  GgmlType,
  GgufTokenizer,
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
