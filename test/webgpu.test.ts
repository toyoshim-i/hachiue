import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { describe, it } from 'node:test'
import {
  dequantizeBlock128Cpu,
  dequantizeBlockCpu,
  fwhtBlockCpu,
  GgmlType,
  SHADERS,
} from '../src/index.js'

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

describe('Real Chrome WebGPU Shader Compilation & Kernel Numerical Accuracy', () => {
  it('compiles all 24 WGSL compute pipelines cleanly and matches CPU reference for Q1_0, PQ2_0, PTQ1_0, Q4_0, Q4_1, Q4_K, Q6_K GEMV & FWHT-512', async () => {
    // 1. Prepare deterministic test vectors on CPU
    const K = 512
    const N = 8
    const nb = K / 128 // 4 blocks per row

    const x = new Float32Array(K)
    for (let i = 0; i < K; i++) x[i] = Math.sin(i * 0.17) * 0.8

    // Prepare Q1_0 weights (N=8, K=512 -> 32 blocks)
    const q1Quants = new Uint8Array(N * nb * 16)
    const q1Scales16 = new Uint16Array(N * nb)
    const q1Expected = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let sum = 0
      for (let b = 0; b < nb; b++) {
        const blkIdx = r * nb + b
        q1Scales16[blkIdx] = 0x3800 // 0.5 in FP16
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
        for (let j = 0; j < 128; j++) sum += x[b * 128 + j] * deq[j]
      }
      q1Expected[r] = sum
    }

    // Prepare PQ2_0 weights (N=8, K=512 -> 32 blocks)
    const pq2Quants = new Uint8Array(N * nb * 32)
    const pq2Scales16 = new Uint16Array(N * nb)
    const pq2Expected = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let sum = 0
      for (let b = 0; b < nb; b++) {
        const blkIdx = r * nb + b
        pq2Scales16[blkIdx] = 0x3c00 // 1.0 in FP16
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
        for (let j = 0; j < 128; j++) sum += x[b * 128 + j] * deq[j]
      }
      pq2Expected[r] = sum
    }

    // Prepare PTQ1_0 weights (N=8, K=512 -> 32 blocks, 26B trits per block)
    // Note: in engine.ts, if N * nb * 26 is not a multiple of 4, u32 indexing still works since N=8, nb=4 => 32*26 = 832 (divisible by 4)
    const ptqQuants = new Uint8Array(N * nb * 26)
    const ptqScales16 = new Uint16Array(N * nb)
    const ptqExpected = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let sum = 0
      for (let b = 0; b < nb; b++) {
        const blkIdx = r * nb + b
        ptqScales16[blkIdx] = 0x3c00 // 1.0 in FP16
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
        for (let j = 0; j < 128; j++) sum += x[b * 128 + j] * deq[j]
      }
      ptqExpected[r] = sum
    }

    // Prepare FWHT-512 expected
    const fwhtExpected = new Float32Array(x)
    fwhtBlockCpu(fwhtExpected, 0, 512)

    // Prepare Q4_0 weights (N=8, K=512 -> nb32 = 16 blocks per row, 18B per block)
    const nb32 = K / 32
    const q40Quants = new Uint8Array(N * nb32 * 18)
    const q40Expected = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let sum = 0
      for (let b = 0; b < nb32; b++) {
        const blkOff = (r * nb32 + b) * 18
        q40Quants[blkOff] = 0x00
        q40Quants[blkOff + 1] = 0x38 // 0.5
        for (let i = 0; i < 16; i++) {
          q40Quants[blkOff + 2 + i] = (r * 29 + b * 17 + i * 7) & 0xff
        }
        const deq = new Float32Array(32)
        dequantizeBlockCpu(q40Quants.subarray(blkOff, blkOff + 18), GgmlType.Q4_0, deq)
        for (let j = 0; j < 32; j++) sum += x[b * 32 + j] * deq[j]
      }
      q40Expected[r] = sum
    }

    // Prepare Q4_1 weights (N=8, K=512 -> nb32 = 16 blocks per row, 20B per block)
    const q41Quants = new Uint8Array(N * nb32 * 20)
    const q41Expected = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let sum = 0
      for (let b = 0; b < nb32; b++) {
        const blkOff = (r * nb32 + b) * 20
        q41Quants[blkOff] = 0x00
        q41Quants[blkOff + 1] = 0x34 // 0.25
        q41Quants[blkOff + 2] = 0x00
        q41Quants[blkOff + 3] = 0xb8 // -0.5
        for (let i = 0; i < 16; i++) {
          q41Quants[blkOff + 4 + i] = (r * 37 + b * 19 + i * 11) & 0xff
        }
        const deq = new Float32Array(32)
        dequantizeBlockCpu(q41Quants.subarray(blkOff, blkOff + 20), GgmlType.Q4_1, deq)
        for (let j = 0; j < 32; j++) sum += x[b * 32 + j] * deq[j]
      }
      q41Expected[r] = sum
    }

    // Prepare Q4_K weights (N=8, K=512 -> nb256 = 2 blocks per row, 144B per block)
    const nb256 = K / 256
    const q4kQuants = new Uint8Array(N * nb256 * 144)
    const q4kExpected = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let sum = 0
      for (let b = 0; b < nb256; b++) {
        const blkOff = (r * nb256 + b) * 144
        q4kQuants[blkOff] = 0x00
        q4kQuants[blkOff + 1] = 0x30 // 0.125
        q4kQuants[blkOff + 2] = 0x00
        q4kQuants[blkOff + 3] = 0x2c // 0.0625
        for (let i = 4; i < 144; i++) {
          q4kQuants[blkOff + i] = (r * 41 + b * 23 + i * 13) & 0xff
        }
        const deq = new Float32Array(256)
        dequantizeBlockCpu(q4kQuants.subarray(blkOff, blkOff + 144), GgmlType.Q4_K, deq)
        for (let j = 0; j < 256; j++) sum += x[b * 256 + j] * deq[j]
      }
      q4kExpected[r] = sum
    }

    // Prepare Q6_K weights (N=8, K=512 -> nb256 = 2 blocks per row, 210B per block)
    const q6kQuants = new Uint8Array(N * nb256 * 210)
    const q6kExpected = new Float32Array(N)
    for (let r = 0; r < N; r++) {
      let sum = 0
      for (let b = 0; b < nb256; b++) {
        const blkOff = (r * nb256 + b) * 210
        for (let i = 0; i < 208; i++) {
          q6kQuants[blkOff + i] = (r * 43 + b * 29 + i * 17) & 0xff
        }
        q6kQuants[blkOff + 208] = 0x00
        q6kQuants[blkOff + 209] = 0x2c // 0.0625
        const deq = new Float32Array(256)
        dequantizeBlockCpu(q6kQuants.subarray(blkOff, blkOff + 210), GgmlType.Q6_K, deq)
        for (let j = 0; j < 256; j++) sum += x[b * 256 + j] * deq[j]
      }
      q6kExpected[r] = sum
    }

    const html = `<!doctype html>
<html><body><script type="module">
const SHADERS = ${JSON.stringify(SHADERS)};
const xData = new Float32Array(${JSON.stringify(Array.from(x))});
const q1Quants = new Uint8Array(${JSON.stringify(Array.from(q1Quants))});
const q1Scales = new Uint16Array(${JSON.stringify(Array.from(q1Scales16))});
const pq2Quants = new Uint8Array(${JSON.stringify(Array.from(pq2Quants))});
const pq2Scales = new Uint16Array(${JSON.stringify(Array.from(pq2Scales16))});
const ptqQuants = new Uint8Array(${JSON.stringify(Array.from(ptqQuants))});
const ptqScales = new Uint16Array(${JSON.stringify(Array.from(ptqScales16))});
const q40Quants = new Uint8Array(${JSON.stringify(Array.from(q40Quants))});
const q41Quants = new Uint8Array(${JSON.stringify(Array.from(q41Quants))});
const q4kQuants = new Uint8Array(${JSON.stringify(Array.from(q4kQuants))});
const q6kQuants = new Uint8Array(${JSON.stringify(Array.from(q6kQuants))});

async function run() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const pipelines = {};
    const compileErrors = {};

    for (const [name, code] of Object.entries(SHADERS)) {
      const mod = device.createShaderModule({ code, label: name });
      const info = await mod.getCompilationInfo();
      const errs = info.messages.filter(m => m.type === 'error');
      if (errs.length) {
        compileErrors[name] = errs;
        continue;
      }
      pipelines[name] = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: mod, entryPoint: 'main' }
      });
    }

    if (Object.keys(compileErrors).length > 0) {
      await fetch('/report', { method: 'POST', body: JSON.stringify({ compileErrors }) });
      return;
    }

    const mkBuf = (arr, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) => {
      const b = device.createBuffer({ size: Math.max(16, (arr.byteLength + 3) & ~3), usage });
      device.queue.writeBuffer(b, 0, arr);
      return b;
    };

    const readF32 = async (buf, count) => {
      const rb = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, rb, 0, count * 4);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const out = Array.from(new Float32Array(rb.getMappedRange().slice(0)));
      rb.unmap();
      return out;
    };

    const xBuf = mkBuf(xData);
    const dummyBuf = mkBuf(new Uint32Array(16));

    const runGemv = async (shaderName, quantsArr, scalesArr, nbCount = 4) => {
      const qBuf = mkBuf(quantsArr);
      const sBuf = mkBuf(scalesArr);
      const yBuf = mkBuf(new Float32Array(8));
      const uBuf = mkBuf(new Uint32Array([1, 8, 512, nbCount, 2, 0, 0, 8]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      let pipe = pipelines[shaderName];
      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uBuf } },
          { binding: 1, resource: { buffer: xBuf } },
          { binding: 2, resource: { buffer: qBuf } },
          { binding: 3, resource: { buffer: sBuf } },
          { binding: 4, resource: { buffer: dummyBuf } },
          { binding: 5, resource: { buffer: yBuf } },
        ]
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(2, 1, 1);
      pass.end();
      device.queue.submit([enc.finish()]);
      return await readF32(yBuf, 8);
    };

    const q1Gpu = await runGemv('gemv_q1_0', q1Quants, q1Scales, 4);
    const pq2Gpu = await runGemv('gemv_pq2_0', pq2Quants, pq2Scales, 4);
    const ptqGpu = await runGemv('gemv_ptq1_0', ptqQuants, ptqScales, 4);
    const q40Gpu = await runGemv('gemv_q4_0', q40Quants, new Uint16Array(8), 16);
    const q41Gpu = await runGemv('gemv_q4_1', q41Quants, new Uint16Array(8), 16);
    const q4kGpu = await runGemv('gemv_q4_k', q4kQuants, new Uint16Array(8), 2);
    const q6kGpu = await runGemv('gemv_q6_k', q6kQuants, new Uint16Array(8), 2);

    // Run FWHT-512
    const fwhtDst = mkBuf(new Float32Array(512));
    const uFwht = mkBuf(new Uint32Array([1, 512, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const pipeFwht = pipelines['hadamard_512'];
    const bgFwht = device.createBindGroup({
      layout: pipeFwht.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uFwht } },
        { binding: 1, resource: { buffer: xBuf } },
        { binding: 2, resource: { buffer: dummyBuf } },
        { binding: 3, resource: { buffer: fwhtDst } },
      ]
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeFwht);
    pass.setBindGroup(0, bgFwht);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();
    device.queue.submit([enc.finish()]);
    const fwhtGpu = await readF32(fwhtDst, 512);

    // Run split_qkv (S=1, qDim=4, kvDim=2 -> total 8 floats)
    const qkvSrc = mkBuf(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    const qDst = mkBuf(new Float32Array(4));
    const kDst = mkBuf(new Float32Array(2));
    const vDst = mkBuf(new Float32Array(2));
    const uSplitQkv = mkBuf(new Uint32Array([1, 4, 2, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const pipeSplitQkv = pipelines['split_qkv'];
    const bgSplitQkv = device.createBindGroup({
      layout: pipeSplitQkv.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uSplitQkv } },
        { binding: 1, resource: { buffer: qkvSrc } },
        { binding: 2, resource: { buffer: qDst } },
        { binding: 3, resource: { buffer: kDst } },
        { binding: 4, resource: { buffer: vDst } },
      ]
    });

    // Run swiglu_fused (S=1, F=4 -> gate=[0, 1, -1, 2], up=[2, 3, 4, 5])
    const gateUpSrc = mkBuf(new Float32Array([0, 1, -1, 2, 2, 3, 4, 5]));
    const swigluDst = mkBuf(new Float32Array(4));
    const uSwiFused = mkBuf(new Uint32Array([1, 4, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const pipeSwiFused = pipelines['swiglu_fused'];
    const bgSwiFused = device.createBindGroup({
      layout: pipeSwiFused.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uSwiFused } },
        { binding: 1, resource: { buffer: gateUpSrc } },
        { binding: 2, resource: { buffer: swigluDst } },
      ]
    });

    // Run add_inplace (N=4)
    const addA = mkBuf(new Float32Array([10, 20, 30, 40]));
    const addB = mkBuf(new Float32Array([1, 2, 3, 4]));
    const addOut = mkBuf(new Float32Array(4));
    const uAdd = mkBuf(new Uint32Array([4, 0, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const pipeAdd = pipelines['add_inplace'];
    const bgAdd = device.createBindGroup({
      layout: pipeAdd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uAdd } },
        { binding: 1, resource: { buffer: addA } },
        { binding: 2, resource: { buffer: addB } },
        { binding: 3, resource: { buffer: addOut } },
      ]
    });

    const enc2 = device.createCommandEncoder();
    const pass2 = enc2.beginComputePass();
    pass2.setPipeline(pipeSplitQkv);
    pass2.setBindGroup(0, bgSplitQkv);
    pass2.dispatchWorkgroups(1, 1, 1);
    pass2.setPipeline(pipeSwiFused);
    pass2.setBindGroup(0, bgSwiFused);
    pass2.dispatchWorkgroups(1, 1, 1);
    pass2.setPipeline(pipeAdd);
    pass2.setBindGroup(0, bgAdd);
    pass2.dispatchWorkgroups(1, 1, 1);
    pass2.end();
    device.queue.submit([enc2.finish()]);

    const qOut = await readF32(qDst, 4);
    const kOut = await readF32(kDst, 2);
    const vOut = await readF32(vDst, 2);
    const swigluGpu = await readF32(swigluDst, 4);
    const addGpu = await readF32(addOut, 4);

    await fetch('/report', {
      method: 'POST',
      body: JSON.stringify({
        ok: true,
        q1Gpu,
        pq2Gpu,
        ptqGpu,
        q40Gpu,
        q41Gpu,
        q4kGpu,
        q6kGpu,
        fwhtGpu,
        qOut,
        kOut,
        vOut,
        swigluGpu,
        addGpu,
      })
    });
  } catch (err) {
    await fetch('/report', { method: 'POST', body: JSON.stringify({ fatal: String(err.stack || err) }) });
  }
}
run();
</script></body></html>`

    const report = await new Promise<Record<string, unknown>>((resolve, reject) => {
      let chromeProc: ReturnType<typeof spawn> | null = null
      const server = http.createServer((req, res) => {
        if (req.url === '/report' && req.method === 'POST') {
          let body = ''
          req.on('data', (c) => (body += c))
          req.on('end', () => {
            res.end('ok')
            chromeProc?.kill()
            server.close()
            resolve(JSON.parse(body))
          })
          return
        }
        res.writeHead(200, { 'Content-Type': 'text/html' })
        res.end(html)
      })
      server.listen(0, '127.0.0.1', () => {
        const addr = server.address() as { port: number }
        chromeProc = spawn(
          CHROME_PATH,
          [
            '--headless=new',
            '--enable-unsafe-webgpu',
            '--use-angle=metal',
            '--no-first-run',
            '--user-data-dir=/tmp/hachiue-webgpu-test',
            `http://127.0.0.1:${addr.port}/`,
          ],
          { stdio: 'ignore' },
        )
        chromeProc.on('error', reject)
      })
    })

    assert.equal(report.ok, true, `WebGPU execution failed: ${JSON.stringify(report)}`)

    const q1Gpu = report.q1Gpu as number[]
    const pq2Gpu = report.pq2Gpu as number[]
    const ptqGpu = report.ptqGpu as number[]
    const q40Gpu = report.q40Gpu as number[]
    const q41Gpu = report.q41Gpu as number[]
    const q4kGpu = report.q4kGpu as number[]
    const q6kGpu = report.q6kGpu as number[]
    const fwhtGpu = report.fwhtGpu as number[]

    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(q1Gpu[r] - q1Expected[r]) < 1e-3, `Q1_0 row ${r}: ${q1Gpu[r]} vs ${q1Expected[r]}`)
      assert.ok(Math.abs(pq2Gpu[r] - pq2Expected[r]) < 1e-3, `PQ2_0 row ${r}: ${pq2Gpu[r]} vs ${pq2Expected[r]}`)
      assert.ok(Math.abs(ptqGpu[r] - ptqExpected[r]) < 1e-3, `PTQ1_0 row ${r}: ${ptqGpu[r]} vs ${ptqExpected[r]}`)
      assert.ok(Math.abs(q40Gpu[r] - q40Expected[r]) < 1e-2, `Q4_0 row ${r}: ${q40Gpu[r]} vs ${q40Expected[r]}`)
      assert.ok(Math.abs(q41Gpu[r] - q41Expected[r]) < 1e-2, `Q4_1 row ${r}: ${q41Gpu[r]} vs ${q41Expected[r]}`)
      assert.ok(Math.abs(q4kGpu[r] - q4kExpected[r]) < 1e-2, `Q4_K row ${r}: ${q4kGpu[r]} vs ${q4kExpected[r]}`)
      assert.ok(Math.abs(q6kGpu[r] - q6kExpected[r]) < 1e-2, `Q6_K row ${r}: ${q6kGpu[r]} vs ${q6kExpected[r]}`)
    }
    for (let i = 0; i < 512; i++) {
      assert.ok(Math.abs(fwhtGpu[i] - fwhtExpected[i]) < 1e-4, `FWHT idx ${i}: ${fwhtGpu[i]} vs ${fwhtExpected[i]}`)
    }

    assert.deepEqual(report.qOut, [1, 2, 3, 4])
    assert.deepEqual(report.kOut, [5, 6])
    assert.deepEqual(report.vOut, [7, 8])
    assert.deepEqual(report.addGpu, [11, 22, 33, 44])

    const swigluGpu = report.swigluGpu as number[]
    const gates = [0, 1, -1, 2]
    const ups = [2, 3, 4, 5]
    for (let i = 0; i < 4; i++) {
      const g = gates[i]
      const expected = (g / (1 + Math.exp(-g))) * ups[i]
      assert.ok(Math.abs(swigluGpu[i] - expected) < 1e-5, `swiglu_fused[${i}]: ${swigluGpu[i]} vs ${expected}`)
    }
  })

  it('spawns a real Blob Web Worker with Wasm SIMD in headless Chrome and runs forward pass & automatic WebGPU fallback', async () => {
    const html = `<!doctype html>
<html><body><script type="module">
import {
  createWasmCore,
  fp16ToFp32,
  getWasmKernelModuleBytes,
  GgmlType,
} from '/dist/index.js';

async function run() {
  try {
    const arch = {
      archName: 'qwen3',
      modelName: 'test-mini-qwen3',
      layers: 1,
      hidden: 128,
      intermediate: 128,
      heads: 2,
      kvHeads: 2,
      headDim: 64,
      vocab: 16,
      maxPositions: 64,
      rmsEps: 1e-6,
      ropeTheta: 10000,
      rope: {
        rotaryDim: 64,
        freqBase: 10000,
        freqScale: 1,
        extFactor: 0,
        attnFactor: 1,
        betaFast: 32,
        betaSlow: 1,
        origCtxLen: 64,
      },
      tieWordEmbeddings: true,
      quantFormat: 'Q1_0',
    };

    const tensors = [
      { name: 'token_embd.weight', type: GgmlType.Q1_0, N: 16, K: 128, nb: 1, hadamardFolded: false, byteLength: 16 * 18 },
      { name: 'output_norm.weight', type: GgmlType.F32, N: 1, K: 128, nb: 128, hadamardFolded: false, byteLength: 128 * 4 },
      { name: 'blk.0.attn_norm.weight', type: GgmlType.F32, N: 1, K: 128, nb: 128, hadamardFolded: false, byteLength: 128 * 4 },
      { name: 'blk.0.ffn_norm.weight', type: GgmlType.F32, N: 1, K: 128, nb: 128, hadamardFolded: false, byteLength: 128 * 4 },
      { name: 'blk.0.attn_q_norm.weight', type: GgmlType.F32, N: 1, K: 64, nb: 64, hadamardFolded: false, byteLength: 64 * 4 },
      { name: 'blk.0.attn_k_norm.weight', type: GgmlType.F32, N: 1, K: 64, nb: 64, hadamardFolded: false, byteLength: 64 * 4 },
      { name: 'blk.0.attn_q.weight', type: GgmlType.Q1_0, N: 128, K: 128, nb: 1, hadamardFolded: false, byteLength: 128 * 18 },
      { name: 'blk.0.attn_k.weight', type: GgmlType.Q1_0, N: 128, K: 128, nb: 1, hadamardFolded: false, byteLength: 128 * 18 },
      { name: 'blk.0.attn_v.weight', type: GgmlType.Q1_0, N: 128, K: 128, nb: 1, hadamardFolded: false, byteLength: 128 * 18 },
      { name: 'blk.0.attn_output.weight', type: GgmlType.Q1_0, N: 128, K: 128, nb: 1, hadamardFolded: false, byteLength: 128 * 18 },
      { name: 'blk.0.ffn_gate.weight', type: GgmlType.Q1_0, N: 128, K: 128, nb: 1, hadamardFolded: false, byteLength: 128 * 18 },
      { name: 'blk.0.ffn_up.weight', type: GgmlType.Q1_0, N: 128, K: 128, nb: 1, hadamardFolded: false, byteLength: 128 * 18 },
      { name: 'blk.0.ffn_down.weight', type: GgmlType.Q1_0, N: 128, K: 128, nb: 1, hadamardFolded: false, byteLength: 128 * 18 },
    ];

    const fp16Table = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) fp16Table[i] = fp16ToFp32(i);

    const cos = new Float32Array(64 * 32).fill(1);
    const sin = new Float32Array(64 * 32).fill(0);
    const wasmBytes = getWasmKernelModuleBytes();

    const initPayload = {
      arch,
      maxSeqLen: 64,
      wasmBytes,
      fp16Table,
      cos,
      sin,
      swaCos: null,
      swaSin: null,
      hadamardSigns: [],
      tensors,
    };

    // 1. Run in-process Wasm SIMD core
    const localCore = createWasmCore(initPayload);
    const rawWeights = new Map();
    for (let tIdx = 0; tIdx < tensors.length; tIdx++) {
      const t = tensors[tIdx];
      let bytes;
      if (t.type === GgmlType.F32) {
        const f32 = new Float32Array(t.K).fill(1.0);
        bytes = new Uint8Array(f32.buffer);
      } else {
        bytes = new Uint8Array(t.N * 18);
        for (let r = 0; r < t.N; r++) {
          bytes[r * 18] = 0x00;
          bytes[r * 18 + 1] = 0x30; // 0.125 in FP16
          for (let i = 0; i < 16; i++) {
            bytes[r * 18 + 2 + i] = ((tIdx * 37 + r * 19 + i * 13) & 0xff);
          }
        }
      }
      rawWeights.set(t.name, bytes);
      localCore.pushTensorBytes(t.name, bytes);
      localCore.finishTensor(t.name);
    }
    localCore.finalizeWeights();

    const localOut = localCore.forwardSegment([1, 2, 3], true, false);

    // 2. Spawn self-contained Blob Web Worker and run identical forward pass
    const workerSrc = \`const createWasmCore = \${createWasmCore.toString()};
    let core = null;
    self.onmessage = (ev) => {
      const msg = ev.data;
      if (msg.type === 'init') {
        core = createWasmCore(msg.payload);
        for (const [name, bytes] of msg.weights) {
          core.pushTensorBytes(name, bytes);
          core.finishTensor(name);
        }
        core.finalizeWeights();
        const res = core.forwardSegment([1, 2, 3], true, false);
        self.postMessage({ ok: true, logits: Array.from(res.logits), greedyId: res.greedyId });
      }
    };\`;
    const blobUrl = URL.createObjectURL(new Blob([workerSrc], { type: 'text/javascript' }));
    const worker = new Worker(blobUrl);
    const workerRes = await new Promise((resolve, reject) => {
      worker.onmessage = (e) => resolve(e.data);
      worker.onerror = (e) => reject(new Error(e.message || 'Worker error'));
      worker.postMessage({ type: 'init', payload: initPayload, weights: Array.from(rawWeights.entries()) });
    });
    worker.terminate();
    URL.revokeObjectURL(blobUrl);

    await fetch('/report', {
      method: 'POST',
      body: JSON.stringify({
        ok: true,
        localLogits: Array.from(localOut.logits),
        localGreedy: localOut.greedyId,
        workerLogits: workerRes.logits,
        workerGreedy: workerRes.greedyId,
      }),
    });
  } catch (err) {
    await fetch('/report', { method: 'POST', body: JSON.stringify({ fatal: String(err.stack || err) }) });
  }
}
run();
</script></body></html>`

    const distRoot = path.resolve(process.cwd(), 'dist')
    const report = await new Promise<Record<string, unknown>>((resolve, reject) => {
      let chromeProc: ReturnType<typeof spawn> | null = null
      const server = http.createServer((req, res) => {
        if (req.url === '/report' && req.method === 'POST') {
          let body = ''
          req.on('data', (c) => (body += c))
          req.on('end', () => {
            res.end('ok')
            chromeProc?.kill()
            server.close()
            resolve(JSON.parse(body))
          })
          return
        }
        if (req.url?.startsWith('/dist/')) {
          const rel = req.url.slice('/dist/'.length)
          const filePath = path.join(distRoot, rel)
          if (fs.existsSync(filePath)) {
            res.writeHead(200, { 'Content-Type': 'text/javascript' })
            res.end(fs.readFileSync(filePath))
            return
          }
          res.writeHead(404)
          res.end('Not found')
          return
        }
        res.writeHead(200, { 'Content-Type': 'text/html' })
        res.end(html)
      })
      server.listen(0, '127.0.0.1', () => {
        const addr = server.address() as { port: number }
        chromeProc = spawn(
          CHROME_PATH,
          [
            '--headless=new',
            '--no-first-run',
            '--user-data-dir=/tmp/hachiue-wasm-worker-test',
            `http://127.0.0.1:${addr.port}/`,
          ],
          { stdio: 'ignore' },
        )
        chromeProc.on('error', reject)
      })
    })

    assert.equal(report.ok, true, `Wasm Worker execution failed: ${JSON.stringify(report)}`)
    const localLogits = report.localLogits as number[]
    const workerLogits = report.workerLogits as number[]
    assert.equal(localLogits.length, 16)
    assert.deepEqual(workerLogits, localLogits)
    assert.equal(report.workerGreedy, report.localGreedy)
    for (const v of workerLogits) {
      assert.ok(Number.isFinite(v), `Expected finite logit, got ${v}`)
    }
  })
})
