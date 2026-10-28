import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import http from 'node:http'
import { describe, it } from 'node:test'
import { dequantizeBlock128Cpu, fwhtBlockCpu, GgmlType, SHADERS } from '../src/index.js'

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

describe('Real Chrome WebGPU Shader Compilation & Kernel Numerical Accuracy', () => {
  it('compiles all 17 WGSL compute pipelines cleanly and matches CPU reference for Q1_0, PQ2_0, PTQ1_0 GEMV & FWHT-512', async () => {
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

    const runGemv = async (shaderName, quantsArr, scalesArr) => {
      const qBuf = mkBuf(quantsArr);
      const sBuf = mkBuf(scalesArr);
      const yBuf = mkBuf(new Float32Array(8));
      const uBuf = mkBuf(new Uint32Array([1, 8, 512, 4, 2, 0, 0, 8]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const pipe = pipelines[shaderName];
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

    const q1Gpu = await runGemv('gemv_q1_0', q1Quants, q1Scales);
    const pq2Gpu = await runGemv('gemv_pq2_0', pq2Quants, pq2Scales);
    const ptqGpu = await runGemv('gemv_ptq1_0', ptqQuants, ptqScales);

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

    await fetch('/report', {
      method: 'POST',
      body: JSON.stringify({ ok: true, q1Gpu, pq2Gpu, ptqGpu, fwhtGpu })
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
    const fwhtGpu = report.fwhtGpu as number[]

    for (let r = 0; r < N; r++) {
      assert.ok(Math.abs(q1Gpu[r] - q1Expected[r]) < 1e-3, `Q1_0 row ${r}: ${q1Gpu[r]} vs ${q1Expected[r]}`)
      assert.ok(Math.abs(pq2Gpu[r] - pq2Expected[r]) < 1e-3, `PQ2_0 row ${r}: ${pq2Gpu[r]} vs ${pq2Expected[r]}`)
      assert.ok(Math.abs(ptqGpu[r] - ptqExpected[r]) < 1e-3, `PTQ1_0 row ${r}: ${ptqGpu[r]} vs ${ptqExpected[r]}`)
    }
    for (let i = 0; i < 512; i++) {
      assert.ok(Math.abs(fwhtGpu[i] - fwhtExpected[i]) < 1e-4, `FWHT idx ${i}: ${fwhtGpu[i]} vs ${fwhtExpected[i]}`)
    }
  })
})
