/**
 * WGSL Compute Shaders for hachiue.
 *
 * Key optimizations over existing engines:
 * 1. Native GGUF Planar layouts without weight expansion:
 *    - Q1_0: 16B sign bits (4x u32) + 2B fp16 scale per 128 weights
 *    - PQ2_0: 32B 2-bit codes (8x u32) + 2B fp16 scale per 128 weights
 *    - PTQ1_0: 26B base-3 trits (24B qs + 2B qh) + 2B fp16 scale per 128 weights
 * 2. Packed FP16 scales (2x fp16 per u32) unpacked via standard `unpack2x16float(u32)`
 *    without requiring `shader-f16` extension.
 * 3. Deferred per-128-block scale multiplication (multiplies once per 128-weight block
 *    instead of 32 times per block).
 * 4. Multi-row (ROWS=4) GEMV across ALL projections including QKV, Gate/Up, O/Down, and LM Head.
 * 5. Fast Walsh-Hadamard Transform (FWHT-512) in workgroup shared memory for Bonsai 2.
 */

export const SHADERS: Record<string, string> = {
  /**
   * Embedding gather for Q1_0, PQ2_0, or PTQ1_0 directly from the shared token_embd buffers.
   * Params: [H, mode (0=Q1_0, 1=PQ2_0, 2=PTQ1_0), tokenCount, rowOffset, rowCount]
   */
  embed_gather: /* wgsl */ `
struct Params { H: u32, mode: u32, S: u32, rowOffset: u32, rowCount: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> quants: array<u32>;
@group(0) @binding(3) var<storage, read> scales16: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn readByte(byteIdx: u32) -> u32 {
  return (quants[byteIdx >> 2u] >> ((byteIdx & 3u) * 8u)) & 0xffu;
}

fn readScale(scaleIdx: u32) -> f32 {
  let pair = unpack2x16float(scales16[scaleIdx >> 1u]);
  return select(pair.x, pair.y, (scaleIdx & 1u) != 0u);
}

fn ptqTrit(blockBaseByte: u32, idxInBlock: u32) -> f32 {
  let pow3 = array<u32, 5>(1u, 3u, 9u, 27u, 81u);
  var b: u32 = 0u;
  var p3: u32 = 1u;
  if (idxInBlock < 80u) {
    let n = idxInBlock / 16u;
    let m = idxInBlock % 16u;
    b = readByte(blockBaseByte + m);
    p3 = pow3[n];
  } else if (idxInBlock < 120u) {
    let rem = idxInBlock - 80u;
    let n = rem / 8u;
    let m = rem % 8u;
    b = readByte(blockBaseByte + 16u + m);
    p3 = pow3[n];
  } else {
    let rem = idxInBlock - 120u;
    let n = rem / 2u;
    let h = rem % 2u;
    b = readByte(blockBaseByte + 24u + h);
    p3 = pow3[n];
  }
  let q = (b * p3) & 0xffu;
  let xi = (q * 3u) >> 8u;
  return f32(i32(xi) - 1);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let s = wg.y;
  if (s >= p.S) { return; }
  let rawTok = tokens[s];
  if (rawTok < p.rowOffset || rawTok >= p.rowOffset + p.rowCount) { return; }
  let tok = rawTok - p.rowOffset;
  let nb = p.H >> 7u;
  for (var k = lid.x; k < p.H; k = k + 256u) {
    let blk = k >> 7u;
    let inBlk = k & 127u;
    let scale = readScale((tok * nb) + blk);
    var val = 0.0;
    if (p.mode == 0u) {
      // Q1_0: 16 bytes (4 u32) per block
      let wordIdx = (tok * nb + blk) * 4u + (inBlk >> 5u);
      let bit = (quants[wordIdx] >> (inBlk & 31u)) & 1u;
      val = select(-scale, scale, bit != 0u);
    } else if (p.mode == 1u) {
      // PQ2_0: 32 bytes (8 u32) per block
      let wordIdx = (tok * nb + blk) * 8u + (inBlk >> 4u);
      let shift = (inBlk & 15u) * 2u;
      let q = (quants[wordIdx] >> shift) & 3u;
      val = f32(i32(q) - 1) * scale;
    } else {
      // PTQ1_0: 26 bytes per block
      let baseByte = (tok * nb + blk) * 26u;
      val = ptqTrit(baseByte, inBlk) * scale;
    }
    out[s * p.H + k] = val;
  }
}
`,

  /**
   * Fast Walsh-Hadamard Transform (block_size = 512) with optional pre-sign or post-sign multiplication.
   * mode = 0 (forward): dst = H_512(signs * src)
   * mode = 1 (inverse): dst = signs * H_512(src)
   * One workgroup of 256 threads processes one 512-element block (2 elements per thread).
   */
  hadamard_512: /* wgsl */ `
struct Params { totalRows: u32, width: u32, mode: u32, hasSigns: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> signs: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

var<workgroup> sh: array<f32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let blocksPerRow = p.width >> 9u; // width / 512
  let blockIdx = wg.x;
  let row = blockIdx / blocksPerRow;
  if (row >= p.totalRows) { return; }
  let blkInRow = blockIdx % blocksPerRow;
  let colBase = blkInRow * 512u;
  let globalBase = row * p.width + colBase;

  let tid = lid.x;
  let i0 = tid;
  let i1 = tid + 256u;

  var v0 = src[globalBase + i0];
  var v1 = src[globalBase + i1];

  if (p.mode == 0u && p.hasSigns != 0u) {
    v0 = v0 * signs[colBase + i0];
    v1 = v1 * signs[colBase + i1];
  }

  sh[i0] = v0;
  sh[i1] = v1;
  workgroupBarrier();

  // 9 stages of butterfly for N = 512 (h = 1, 2, 4, 8, 16, 32, 64, 128, 256)
  for (var s = 0u; s < 9u; s = s + 1u) {
    let h = 1u << s;
    let idx0 = ((tid >> s) << (s + 1u)) | (tid & (h - 1u));
    let idx1 = idx0 + h;
    let a = sh[idx0];
    let b = sh[idx1];
    workgroupBarrier();
    sh[idx0] = a + b;
    sh[idx1] = a - b;
    workgroupBarrier();
  }

  // Normalization factor: 1 / sqrt(512) = 0.04419417382415922
  let norm = 0.04419417382415922;
  var out0 = sh[i0] * norm;
  var out1 = sh[i1] * norm;

  if (p.mode == 1u && p.hasSigns != 0u) {
    out0 = out0 * signs[colBase + i0];
    out1 = out1 * signs[colBase + i1];
  }

  dst[globalBase + i0] = out0;
  dst[globalBase + i1] = out1;
}
`,

  /**
   * Dense F32 GEMV for small projections (e.g. ssm_alpha / ssm_beta `[N, K]` in Qwen3.5 where F32 is used).
   * One workgroup of 64 threads computes 4 output rows.
   */
  gemv_f32: /* wgsl */ `
const WG: u32 = 64u;
const ROWS: u32 = 4u;

struct Params { M: u32, N: u32, K: u32, gridX: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rowBase = (wg.y * p.gridX + wg.x) * ROWS;
  let m = wg.z;
  if (m >= p.M) { return; }
  let tid = lid.x;
  let Kvec = p.K >> 2u;
  let xBase = m * Kvec;

  var acc = vec4<f32>(0.0);
  for (var kv = tid; kv < Kvec; kv = kv + WG) {
    let xv = x[xBase + kv];
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        acc[r] = acc[r] + dot(xv, w[n * Kvec + kv]);
      }
    }
  }
  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) {
    let totals = sdata[0];
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        y[m * p.N + n] = totals[r];
      }
    }
  }
}
`,

  /**
   * Optional channel permutation for Qwen3.5 Gated DeltaNet ssm_out when hadamard_gdn_v_grouped is enabled:
   * tiled [hd, nk, rep] -> grouped [hd, rep, nk]
   */
  permute_gdn: /* wgsl */ `
struct Params { S: u32, hd: u32, nk: u32, rep: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let width = p.hd * p.nk * p.rep;
  let total = p.S * width;
  let idx = wg.x * 256u + lid.x;
  if (idx >= total) { return; }
  let s = idx / width;
  let rem = idx % width;
  let d = rem % p.hd;
  let tmp = rem / p.hd;
  let r = tmp % p.rep;
  let k = tmp / p.rep;
  let srcIdx = s * width + (r * p.nk + k) * p.hd + d;
  dst[idx] = src[srcIdx];
}
`,

  /**
   * RMSNorm: y[row, i] = gamma[i] * x[row, i] / sqrt(mean(x^2) + eps)
   * One workgroup (256 threads) per row.
   */
  rmsnorm: /* wgsl */ `
struct Params { rows: u32, D: u32, eps: f32, addUnitOffset: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= p.rows) { return; }
  let tid = lid.x;
  let base = row * p.D;
  var sumSq = 0.0;
  for (var i = tid; i < p.D; i = i + 256u) {
    let v = x[base + i];
    sumSq = sumSq + v * v;
  }
  sdata[tid] = sumSq;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sdata[0] / f32(p.D) + p.eps);
  for (var i = tid; i < p.D; i = i + 256u) {
    let g = select(gamma[i], 1.0 + gamma[i], p.addUnitOffset != 0u);
    y[base + i] = x[base + i] * inv * g;
  }
}
`,

  /**
   * Multi-row GEMV / Small-Batch GEMM for Q1_0, PQ2_0, and PTQ1_0 with packed FP16 scales
   * and deferred per-128-block scale multiplication!
   *
   * Uses workgroup reduction (64 threads per workgroup, 4 output rows per workgroup -> 16 threads per row,
   * or 64 threads cooperating on 4 rows across K blocks).
   * Let's have all 64 threads iterate over 128-weight blocks (`blk = tid; blk < nb; blk += 64`),
   * computing 4 output rows simultaneously!
   * Even better: for K=2048, nb = 16 blocks of 128 weights.
   * Each 128-weight block has 4 sub-blocks of 32 weights (each 32 weights = 8 x vec4<f32>).
   * So `nb * 4` = 64 units of 32 weights for K=2048!
   * Each thread `tid` (0..63) handles 32 weights per step (`u = tid; u < nb * 4; u += 64`):
   * - `blk = u >> 2` (which 128-weight block)
   * - `sub = u & 3` (which 32-weight slice inside the 128-weight block)
   * Every thread loads a FULL u32 of Q1_0 sign bits (or 2x u32 of PQ2_0 codes, or 32 trits of PTQ1_0)
   * without ANY bit-waste across threads, accumulates all 32 weights in register, and multiplies by the
   * FP16 block scale ONCE per 32-weight slice!
   */
  gemv_q1_0: /* wgsl */ `
const WG: u32 = 64u;
const ROWS: u32 = 4u;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  nb: u32,
  gridX: u32,
  hasResid: u32,
  rowOffset: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> quants: array<u32>;
@group(0) @binding(3) var<storage, read> scales16: array<u32>;
@group(0) @binding(4) var<storage, read> resid: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 64>;

fn readScale(scaleIdx: u32) -> f32 {
  let pair = unpack2x16float(scales16[scaleIdx >> 1u]);
  return select(pair.x, pair.y, (scaleIdx & 1u) != 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rowBase = (wg.y * p.gridX + wg.x) * ROWS;
  let m = wg.z;
  if (m >= p.M) { return; }
  let tid = lid.x;
  let numUnits = p.nb * 4u; // each unit is 32 weights (1 u32 in Q1_0)
  let xRowBase = m * (p.K >> 2u);

  var acc = vec4<f32>(0.0);

  for (var u = tid; u < numUnits; u = u + WG) {
    let blk = u >> 2u;
    let sub = u & 3u;
    let xVecBase = xRowBase + u * 8u;

    // Load 32 activations (8 x vec4<f32>) into registers once for all 4 output rows
    let x0 = x[xVecBase + 0u];
    let x1 = x[xVecBase + 1u];
    let x2 = x[xVecBase + 2u];
    let x3 = x[xVecBase + 3u];
    let x4 = x[xVecBase + 4u];
    let x5 = x[xVecBase + 5u];
    let x6 = x[xVecBase + 6u];
    let x7 = x[xVecBase + 7u];

    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let word = quants[(n * p.nb + blk) * 4u + sub];
        let scale = readScale(n * p.nb + blk);
        var subSum = 0.0;
        subSum += dot(x0, vec4<f32>(select(-1.0, 1.0, (word & (1u << 0u)) != 0u), select(-1.0, 1.0, (word & (1u << 1u)) != 0u), select(-1.0, 1.0, (word & (1u << 2u)) != 0u), select(-1.0, 1.0, (word & (1u << 3u)) != 0u)));
        subSum += dot(x1, vec4<f32>(select(-1.0, 1.0, (word & (1u << 4u)) != 0u), select(-1.0, 1.0, (word & (1u << 5u)) != 0u), select(-1.0, 1.0, (word & (1u << 6u)) != 0u), select(-1.0, 1.0, (word & (1u << 7u)) != 0u)));
        subSum += dot(x2, vec4<f32>(select(-1.0, 1.0, (word & (1u << 8u)) != 0u), select(-1.0, 1.0, (word & (1u << 9u)) != 0u), select(-1.0, 1.0, (word & (1u << 10u)) != 0u), select(-1.0, 1.0, (word & (1u << 11u)) != 0u)));
        subSum += dot(x3, vec4<f32>(select(-1.0, 1.0, (word & (1u << 12u)) != 0u), select(-1.0, 1.0, (word & (1u << 13u)) != 0u), select(-1.0, 1.0, (word & (1u << 14u)) != 0u), select(-1.0, 1.0, (word & (1u << 15u)) != 0u)));
        subSum += dot(x4, vec4<f32>(select(-1.0, 1.0, (word & (1u << 16u)) != 0u), select(-1.0, 1.0, (word & (1u << 17u)) != 0u), select(-1.0, 1.0, (word & (1u << 18u)) != 0u), select(-1.0, 1.0, (word & (1u << 19u)) != 0u)));
        subSum += dot(x5, vec4<f32>(select(-1.0, 1.0, (word & (1u << 20u)) != 0u), select(-1.0, 1.0, (word & (1u << 21u)) != 0u), select(-1.0, 1.0, (word & (1u << 22u)) != 0u), select(-1.0, 1.0, (word & (1u << 23u)) != 0u)));
        subSum += dot(x6, vec4<f32>(select(-1.0, 1.0, (word & (1u << 24u)) != 0u), select(-1.0, 1.0, (word & (1u << 25u)) != 0u), select(-1.0, 1.0, (word & (1u << 26u)) != 0u), select(-1.0, 1.0, (word & (1u << 27u)) != 0u)));
        subSum += dot(x7, vec4<f32>(select(-1.0, 1.0, (word & (1u << 28u)) != 0u), select(-1.0, 1.0, (word & (1u << 29u)) != 0u), select(-1.0, 1.0, (word & (1u << 30u)) != 0u), select(-1.0, 1.0, (word & (1u << 31u)) != 0u)));
        acc[r] = acc[r] + subSum * scale;
      }
    }
  }

  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let totals = sdata[0];
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let outIdx = m * p.N + p.rowOffset + n;
        let resVal = select(0.0, resid[outIdx], p.hasResid != 0u);
        y[outIdx] = totals[r] + resVal;
      }
    }
  }
}
`,

  /**
   * Multi-row GEMV / Small-Batch GEMM for PQ2_0 (2-bit ternary codes, 8 u32 per 128-weight block).
   * Each unit `u` is 16 weights (1 u32 of 2-bit codes = 16 weights = 4 x vec4<f32>).
   */
  gemv_pq2_0: /* wgsl */ `
const WG: u32 = 64u;
const ROWS: u32 = 4u;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  nb: u32,
  gridX: u32,
  hasResid: u32,
  rowOffset: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> quants: array<u32>;
@group(0) @binding(3) var<storage, read> scales16: array<u32>;
@group(0) @binding(4) var<storage, read> resid: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 64>;

fn readScale(scaleIdx: u32) -> f32 {
  let pair = unpack2x16float(scales16[scaleIdx >> 1u]);
  return select(pair.x, pair.y, (scaleIdx & 1u) != 0u);
}

fn decode4x2bit(byteVal: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(i32(byteVal & 3u) - 1),
    f32(i32((byteVal >> 2u) & 3u) - 1),
    f32(i32((byteVal >> 4u) & 3u) - 1),
    f32(i32((byteVal >> 6u) & 3u) - 1)
  );
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rowBase = (wg.y * p.gridX + wg.x) * ROWS;
  let m = wg.z;
  if (m >= p.M) { return; }
  let tid = lid.x;
  let numUnits = p.nb * 8u; // each unit is 16 weights (1 u32 in PQ2_0)
  let xRowBase = m * (p.K >> 2u);

  var acc = vec4<f32>(0.0);

  for (var u = tid; u < numUnits; u = u + WG) {
    let blk = u >> 3u;
    let sub = u & 7u;
    let xVecBase = xRowBase + u * 4u;

    let x0 = x[xVecBase + 0u];
    let x1 = x[xVecBase + 1u];
    let x2 = x[xVecBase + 2u];
    let x3 = x[xVecBase + 3u];

    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let word = quants[(n * p.nb + blk) * 8u + sub];
        let scale = readScale(n * p.nb + blk);
        var subSum = 0.0;
        subSum += dot(x0, decode4x2bit(word & 0xffu));
        subSum += dot(x1, decode4x2bit((word >> 8u) & 0xffu));
        subSum += dot(x2, decode4x2bit((word >> 16u) & 0xffu));
        subSum += dot(x3, decode4x2bit((word >> 24u) & 0xffu));
        acc[r] = acc[r] + subSum * scale;
      }
    }
  }

  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let totals = sdata[0];
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let outIdx = m * p.N + p.rowOffset + n;
        let resVal = select(0.0, resid[outIdx], p.hasResid != 0u);
        y[outIdx] = totals[r] + resVal;
      }
    }
  }
}
`,

  /**
   * Multi-row GEMV / Small-Batch GEMM for PTQ1_0 (26B base-3 trits per 128-weight block).
   * In PTQ1_0, each 128-weight block has:
   * - bytes 0..15: 16 bytes, each encoding 5 trits at stride 16 (indices m + n*16 for n=0..4) -> 80 weights
   * - bytes 16..23: 8 bytes, each encoding 5 trits at stride 8 (indices 80 + m + n*8 for n=0..4) -> 40 weights
   * - bytes 24..25: 2 bytes, each encoding 4 trits at stride 2 (indices 120 + h + n*2 for n=0..3) -> 8 weights
   * Notice that iterating by byte `bIdx = 0..25` inside each block decodes 5 (or 4) trits from a SINGLE byte load!
   * Even simpler: 16 threads per 128-weight block (4 blocks per 64-thread workgroup step):
   * `sub = u & 15` (0..15):
   * - Thread `sub` (0..15) decodes byte `sub` (5 weights at `sub + n*16`).
   * - If `sub < 8`, it also decodes byte `16 + sub` (5 weights at `80 + sub + n*8`).
   * - If `sub < 2`, it also decodes byte `24 + sub` (4 weights at `120 + sub + n*2`).
   * Total byte loads per 128-weight block = 26 bytes (zero duplicate byte reads!), and only ONE scale multiply!
   */
  gemv_ptq1_0: /* wgsl */ `
const WG: u32 = 64u;
const ROWS: u32 = 4u;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  nb: u32,
  gridX: u32,
  hasResid: u32,
  rowOffset: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> quants: array<u32>;
@group(0) @binding(3) var<storage, read> scales16: array<u32>;
@group(0) @binding(4) var<storage, read> resid: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 64>;

fn readByte(byteIdx: u32) -> u32 {
  return (quants[byteIdx >> 2u] >> ((byteIdx & 3u) * 8u)) & 0xffu;
}

fn readScale(scaleIdx: u32) -> f32 {
  let pair = unpack2x16float(scales16[scaleIdx >> 1u]);
  return select(pair.x, pair.y, (scaleIdx & 1u) != 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rowBase = (wg.y * p.gridX + wg.x) * ROWS;
  let m = wg.z;
  if (m >= p.M) { return; }
  let tid = lid.x;
  let numUnits = p.nb * 16u; // 16 threads cover one 128-weight PTQ1_0 block
  let xRowBase = m * p.K;
  let pow3 = array<u32, 5>(1u, 3u, 9u, 27u, 81u);

  var acc = vec4<f32>(0.0);

  for (var u = tid; u < numUnits; u = u + WG) {
    let blk = u >> 4u;
    let sub = u & 15u;
    let xBlkBase = xRowBase + blk * 128u;

    // Preload activations for this thread's assigned trits
    let xa0 = x[xBlkBase + sub + 0u];
    let xa1 = x[xBlkBase + sub + 16u];
    let xa2 = x[xBlkBase + sub + 32u];
    let xa3 = x[xBlkBase + sub + 48u];
    let xa4 = x[xBlkBase + sub + 64u];

    var xb0 = 0.0; var xb1 = 0.0; var xb2 = 0.0; var xb3 = 0.0; var xb4 = 0.0;
    if (sub < 8u) {
      xb0 = x[xBlkBase + 80u + sub + 0u];
      xb1 = x[xBlkBase + 80u + sub + 8u];
      xb2 = x[xBlkBase + 80u + sub + 16u];
      xb3 = x[xBlkBase + 80u + sub + 24u];
      xb4 = x[xBlkBase + 80u + sub + 32u];
    }
    var xc0 = 0.0; var xc1 = 0.0; var xc2 = 0.0; var xc3 = 0.0;
    if (sub < 2u) {
      xc0 = x[xBlkBase + 120u + sub + 0u];
      xc1 = x[xBlkBase + 120u + sub + 2u];
      xc2 = x[xBlkBase + 120u + sub + 4u];
      xc3 = x[xBlkBase + 120u + sub + 6u];
    }

    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let baseByte = (n * p.nb + blk) * 26u;
        let scale = readScale(n * p.nb + blk);
        var subSum = 0.0;

        let ba = readByte(baseByte + sub);
        subSum += xa0 * f32(i32((((ba * 1u) & 0xffu) * 3u) >> 8u) - 1);
        subSum += xa1 * f32(i32((((ba * 3u) & 0xffu) * 3u) >> 8u) - 1);
        subSum += xa2 * f32(i32((((ba * 9u) & 0xffu) * 3u) >> 8u) - 1);
        subSum += xa3 * f32(i32((((ba * 27u) & 0xffu) * 3u) >> 8u) - 1);
        subSum += xa4 * f32(i32((((ba * 81u) & 0xffu) * 3u) >> 8u) - 1);

        if (sub < 8u) {
          let bb = readByte(baseByte + 16u + sub);
          subSum += xb0 * f32(i32((((bb * 1u) & 0xffu) * 3u) >> 8u) - 1);
          subSum += xb1 * f32(i32((((bb * 3u) & 0xffu) * 3u) >> 8u) - 1);
          subSum += xb2 * f32(i32((((bb * 9u) & 0xffu) * 3u) >> 8u) - 1);
          subSum += xb3 * f32(i32((((bb * 27u) & 0xffu) * 3u) >> 8u) - 1);
          subSum += xb4 * f32(i32((((bb * 81u) & 0xffu) * 3u) >> 8u) - 1);
        }
        if (sub < 2u) {
          let bc = readByte(baseByte + 24u + sub);
          subSum += xc0 * f32(i32((((bc * 1u) & 0xffu) * 3u) >> 8u) - 1);
          subSum += xc1 * f32(i32((((bc * 3u) & 0xffu) * 3u) >> 8u) - 1);
          subSum += xc2 * f32(i32((((bc * 9u) & 0xffu) * 3u) >> 8u) - 1);
          subSum += xc3 * f32(i32((((bc * 27u) & 0xffu) * 3u) >> 8u) - 1);
        }

        acc[r] = acc[r] + subSum * scale;
      }
    }
  }

  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let totals = sdata[0];
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let outIdx = m * p.N + p.rowOffset + n;
        let resVal = select(0.0, resid[outIdx], p.hasResid != 0u);
        y[outIdx] = totals[r] + resVal;
      }
    }
  }
}
`,

  /**
   * RoPE (supports full or partial rotary_dim, e.g. rotary_dim=64 on head_dim=256 for Qwen3.5).
   * In-place on q or k buffer `[S, H, Dh]`.
   */
  rope: /* wgsl */ `
struct Params { S: u32, H: u32, Dh: u32, rotaryDim: u32, posBase: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> cosTable: array<f32>;
@group(0) @binding(2) var<storage, read> sinTable: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let halfRot = p.rotaryDim >> 1u;
  let totalPairs = p.S * p.H * halfRot;
  let idx = wg.x * 64u + lid.x;
  if (idx >= totalPairs) { return; }

  let pair = idx % halfRot;
  let tmp = idx / halfRot;
  let h = tmp % p.H;
  let s = tmp / p.H;

  let pos = p.posBase + s;
  let csIdx = pos * halfRot + pair;
  let c = cosTable[csIdx];
  let sn = sinTable[csIdx];

  let base = (s * p.H + h) * p.Dh;
  let v0 = x[base + pair];
  let v1 = x[base + pair + halfRot];
  x[base + pair] = v0 * c - v1 * sn;
  x[base + pair + halfRot] = v1 * c + v0 * sn;
}
`,

  /**
   * Split doubled Q_proj `[S, H, 2 * Dh]` into `query [S, H, Dh]` and `gate [S, H, Dh]` for Qwen3.5 full-attention layers.
   */
  split_q_gate: /* wgsl */ `
struct Params { S: u32, H: u32, Dh: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qg: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let total = p.S * p.H * p.Dh;
  let idx = wg.x * 256u + lid.x;
  if (idx >= total) { return; }
  let d = idx % p.Dh;
  let sh = idx / p.Dh;
  let srcBase = sh * (p.Dh * 2u);
  q[idx] = qg[srcBase + d];
  gate[idx] = qg[srcBase + p.Dh + d];
}
`,

  /**
   * Multiply attention output by sigmoid(gate) for Qwen3.5 full-attention layers.
   */
  sigmoid_gate: /* wgsl */ `
struct Params { N: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> att: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x * 256u + lid.x;
  if (idx >= p.N) { return; }
  let g = gate[idx];
  att[idx] = att[idx] * (1.0 / (1.0 + exp(-g)));
}
`,

  /**
   * SwiGLU activation: out[i] = silu(gate[i]) * up[i]
   */
  swiglu: /* wgsl */ `
struct Params { N: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x * 256u + lid.x;
  if (idx >= p.N) { return; }
  let g = gate[idx];
  out[idx] = (g / (1.0 + exp(-g))) * up[idx];
}
`,

  /**
   * Quantize and append K or V `[S, KV, Dh]` into Q8_0 KV cache (`Kq [maxSeq, KV, Dh/4]` u32 + `Ks [maxSeq, KV, Dh/32]` f32).
   * One workgroup of 32 threads quantizes one 32-element block.
   */
  kv_append_q8: /* wgsl */ `
struct Params { S: u32, KV: u32, Dh: u32, posBase: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> cacheQ: array<u32>;
@group(0) @binding(3) var<storage, read_write> cacheS: array<f32>;

var<workgroup> smax: array<f32, 32>;
var<workgroup> sq: array<u32, 32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let blocksPerToken = p.KV * (p.Dh >> 5u);
  let totalBlocks = p.S * blocksPerToken;
  let blkIdx = wg.x;
  if (blkIdx >= totalBlocks) { return; }

  let tid = lid.x;
  let s = blkIdx / blocksPerToken;
  let rem = blkIdx % blocksPerToken;
  let srcBase = (s * blocksPerToken + rem) * 32u;
  let dstBlockIdx = ((p.posBase + s) * blocksPerToken) + rem;

  let v = src[srcBase + tid];
  smax[tid] = abs(v);
  workgroupBarrier();
  for (var st = 16u; st > 0u; st = st >> 1u) {
    if (tid < st) { smax[tid] = max(smax[tid], smax[tid + st]); }
    workgroupBarrier();
  }
  let amax = smax[0];
  let scale = amax / 127.0;
  let inv = select(0.0, 127.0 / amax, amax > 0.0);
  let q = clamp(i32(round(v * inv)), -127, 127);
  sq[tid] = u32(q & 0xff);
  workgroupBarrier();

  if (tid < 8u) {
    let b0 = sq[tid * 4u + 0u];
    let b1 = sq[tid * 4u + 1u];
    let b2 = sq[tid * 4u + 2u];
    let b3 = sq[tid * 4u + 3u];
    cacheQ[dstBlockIdx * 8u + tid] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
  }
  if (tid == 0u) {
    cacheS[dstBlockIdx] = scale * 127.0; // unpack4x8snorm maps [-127, 127] to [-1, 1]
  }
}
`,

  /**
   * Online Softmax GQA Attention with Q8_0 KV Cache.
   * One workgroup (32 threads) per (query_pos, query_head).
   */
  attention_q8: /* wgsl */ `
struct Params { S: u32, H: u32, KV: u32, Dh: u32, posBase: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Kq: array<u32>;
@group(0) @binding(3) var<storage, read> Ks: array<f32>;
@group(0) @binding(4) var<storage, read> Vq: array<u32>;
@group(0) @binding(5) var<storage, read> Vs: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<vec4<f32>>;

var<workgroup> sdot: array<f32, 32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);

  let W4 = p.Dh >> 2u;   // vec4 count per head (e.g. 32 for Dh=128, 64 for Dh=256)
  let B32 = p.Dh >> 5u;  // 32-elem block count per head (e.g. 4 for Dh=128, 8 for Dh=256)
  let qBase = (qi * p.H + h) * W4;
  let scale = inverseSqrt(f32(p.Dh));

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var m = -1e30;
  var l = 0.0;

  for (var j = 0u; j <= pos; j = j + 1u) {
    let kvRow = j * p.KV + kvh;
    let qWordBase = kvRow * W4;
    let sBlockBase = kvRow * B32;

    var part = 0.0;
    if (tid < W4) {
      let kw = unpack4x8snorm(Kq[qWordBase + tid]) * Ks[sBlockBase + (tid >> 3u)];
      part = part + dot(q[qBase + tid], kw);
    }
    if (tid + 32u < W4) {
      let kw2 = unpack4x8snorm(Kq[qWordBase + tid + 32u]) * Ks[sBlockBase + ((tid + 32u) >> 3u)];
      part = part + dot(q[qBase + tid + 32u], kw2);
    }
    sdot[tid] = part;
    workgroupBarrier();
    for (var s = 16u; s > 0u; s = s >> 1u) {
      if (tid < s) { sdot[tid] = sdot[tid] + sdot[tid + s]; }
      workgroupBarrier();
    }

    let score = sdot[0] * scale;
    let mNew = max(m, score);
    let corr = exp(m - mNew);
    let wgt = exp(score - mNew);
    l = l * corr + wgt;

    if (tid < W4) {
      let vw = unpack4x8snorm(Vq[qWordBase + tid]) * Vs[sBlockBase + (tid >> 3u)];
      acc0 = acc0 * corr + wgt * vw;
    }
    if (tid + 32u < W4) {
      let vw2 = unpack4x8snorm(Vq[qWordBase + tid + 32u]) * Vs[sBlockBase + ((tid + 32u) >> 3u)];
      acc1 = acc1 * corr + wgt * vw2;
    }
    m = mNew;
    workgroupBarrier();
  }

  let invL = 1.0 / l;
  if (tid < W4) {
    out[qBase + tid] = acc0 * invL;
  }
  if (tid + 32u < W4) {
    out[qBase + tid + 32u] = acc1 * invL;
  }
}
`,

  /**
   * Qwen3.5 Gated DeltaNet: Causal Conv1D (width K=4) + SiLU.
   */
  conv1d_causal: /* wgsl */ `
struct Params { S: u32, C: u32, K: u32, loadState: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> stateIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> stateOut: array<f32>;

fn ext(m: u32, c: u32) -> f32 {
  if (m + 1u < p.K) {
    return select(0.0, stateIn[m * p.C + c], p.loadState != 0u);
  }
  return x[(m - (p.K - 1u)) * p.C + c];
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  let outN = p.S * p.C;
  if (i < outN) {
    let c = i % p.C;
    let t = i / p.C;
    var acc = 0.0;
    for (var j = 0u; j < p.K; j = j + 1u) {
      acc = acc + w[c * p.K + j] * ext(t + j, c);
    }
    y[i] = acc / (1.0 + exp(-acc));
  } else if (i < outN + (p.K - 1u) * p.C) {
    let si = i - outN;
    let sc = si % p.C;
    let sk = si / p.C;
    stateOut[sk * p.C + sc] = ext(p.S + sk, sc);
  }
}
`,

  /**
   * Qwen3.5 Gated DeltaNet: compute g = a_neg * softplus(a + dt_bias) and beta = sigmoid(b).
   */
  deltanet_gbeta: /* wgsl */ `
struct Params { S: u32, H: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> aNeg: array<f32>;
@group(0) @binding(4) var<storage, read> dtBias: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wg.x * 64u + lid.x;
  let n = p.S * p.H;
  if (i >= n) { return; }
  let h = i % p.H;
  let x = a[i] + dtBias[h];
  let sp = max(x, 0.0) + log(1.0 + exp(-abs(x)));
  out[i] = aNeg[h] * sp;
  out[n + i] = 1.0 / (1.0 + exp(-b[i]));
}
`,

  /**
   * Qwen3.5 Gated DeltaNet: Recurrent O(1)-state scan over convOutput `[S, 2*keyDim + valDim]`.
   */
  deltanet_recur: /* wgsl */ `
struct Params { S: u32, H: u32, DK: u32, DV: u32, HK: u32, loadState: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qkvConv: array<f32>;
@group(0) @binding(2) var<storage, read> gbeta: array<f32>;
@group(0) @binding(3) var<storage, read> stateIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> core: array<f32>;
@group(0) @binding(5) var<storage, read_write> stateOut: array<f32>;

var<workgroup> ksh: array<f32, 128>;
var<workgroup> qsh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let hk = h % p.HK;
  let dv = lid.x;
  let DK = p.DK;
  let keyDim = p.HK * DK;
  let valDim = p.H * p.DV;
  let convDim = keyDim * 2u + valDim;
  let sbase = h * DK * p.DV + dv;
  let scale = inverseSqrt(f32(DK));
  let betaOff = p.S * p.H;

  var s: array<f32, 128>;
  for (var dk = 0u; dk < DK; dk = dk + 1u) {
    s[dk] = select(0.0, stateIn[sbase + dk * p.DV], p.loadState != 0u);
  }

  for (var t = 0u; t < p.S; t = t + 1u) {
    let rowConv = t * convDim;
    let qOff = rowConv + hk * DK;
    let kOff = rowConv + keyDim + hk * DK;
    let vOff = rowConv + keyDim * 2u + h * p.DV;
    let gbIdx = t * p.H + h;

    if (dv < DK) {
      qsh[dv] = qkvConv[qOff + dv];
      ksh[dv] = qkvConv[kOff + dv];
    }
    workgroupBarrier();

    var sk = 0.0;
    var sq = 0.0;
    for (var dk = 0u; dk < DK; dk = dk + 1u) {
      sk = sk + ksh[dk] * ksh[dk];
      sq = sq + qsh[dk] * qsh[dk];
    }
    let ik = inverseSqrt(sk + 1e-6);
    let iq = inverseSqrt(sq + 1e-6) * scale;

    if (dv < p.DV) {
      let gt = exp(gbeta[gbIdx]);
      let bt = gbeta[betaOff + gbIdx];
      var kv = 0.0;
      for (var dk = 0u; dk < DK; dk = dk + 1u) {
        s[dk] = s[dk] * gt;
        kv = kv + s[dk] * ksh[dk] * ik;
      }
      let delta = (qkvConv[vOff + dv] - kv) * bt;
      var o = 0.0;
      for (var dk = 0u; dk < DK; dk = dk + 1u) {
        s[dk] = s[dk] + ksh[dk] * ik * delta;
        o = o + s[dk] * qsh[dk] * iq;
      }
      core[(t * p.H + h) * p.DV + dv] = o;
    }
    workgroupBarrier();
  }

  if (dv < p.DV) {
    for (var dk = 0u; dk < DK; dk = dk + 1u) {
      stateOut[sbase + dk * p.DV] = s[dk];
    }
  }
}
`,

  /**
   * Qwen3.5 Gated DeltaNet: Gated RMSNorm `y = gamma * rmsnorm(core) * silu(z)`.
   */
  deltanet_norm_gate: /* wgsl */ `
struct Params { rows: u32, DV: u32, eps: f32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> core: array<f32>;
@group(0) @binding(2) var<storage, read> z: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= p.rows) { return; }
  let tid = lid.x;
  let base = row * p.DV;
  var s = 0.0;
  for (var i = tid; i < p.DV; i = i + 128u) {
    let c = core[base + i];
    s = s + c * c;
  }
  sdata[tid] = s;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st = st >> 1u) {
    if (tid < st) { sdata[tid] = sdata[tid] + sdata[tid + st]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sdata[0] / f32(p.DV) + p.eps);
  for (var i = tid; i < p.DV; i = i + 128u) {
    let zz = z[base + i];
    y[base + i] = gamma[i] * (core[base + i] * inv) * (zz / (1.0 + exp(-zz)));
  }
}
`,

  /**
   * Fast GPU Argmax over vocab logits (single workgroup of 256 threads).
   */
  argmax: /* wgsl */ `
struct Params { vocab: u32, rowOffset: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outToken: array<u32>;

var<workgroup> smax: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let base = p.rowOffset * p.vocab;
  var bestVal = -1e30;
  var bestIdx = 0u;
  for (var i = tid; i < p.vocab; i = i + 256u) {
    let v = logits[base + i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  smax[tid] = bestVal;
  sidx[tid] = bestIdx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      if (smax[tid + s] > smax[tid]) {
        smax[tid] = smax[tid + s];
        sidx[tid] = sidx[tid + s];
      }
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    outToken[0] = sidx[0];
  }
}
`,
}
