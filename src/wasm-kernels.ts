/**
 * Zero-dependency inline WebAssembly SIMD (v128) GEMV kernel generator for hachiue.
 *
 * Generates a compact, valid WebAssembly binary module in memory with native 128-bit SIMD
 * (`i32x4.dot_i16x8_s`, `i16x8.load8x8_u/s`, `f32x4`) matrix-vector multiplication kernels
 * for all supported GGUF quantization formats:
 * - F32
 * - Q1_0 (1-bit binary, 128 elements/block)
 * - PQ2_0 (2-bit ternary, 128 elements/block)
 * - PTQ1_0 (base-3 ternary, 128 elements/block)
 * - Q4_0 (4-bit symmetric, 32 elements/block)
 * - Q4_1 (4-bit affine, 32 elements/block)
 * - Q4_K (4-bit K-quant super-block, 256 elements/block)
 * - Q6_K (6-bit K-quant super-block, 256 elements/block)
 */

// Fixed scratch memory map inside the first 5 MiB of WebAssembly.Memory
export const WASM_MEM_FP16_TABLE = 0x000000 // 256 KiB (65536 * f32)
export const WASM_MEM_Q1_LUT = 0x040000 // 2 KiB (256 * 8 bytes of signed i8 {-1, +1})
export const WASM_MEM_CONSTS = 0x040800 // 2 KiB (PTQ1_0 tail constants, etc.)
export const WASM_MEM_X_F32 = 0x041000 // 256 KiB (up to 65536 * f32)
export const WASM_MEM_X_Q16 = 0x081000 // 128 KiB (up to 65536 * i16)
export const WASM_MEM_X_PQ2_Q16 = 0x0a1000 // 128 KiB (de-interleaved i16 for PQ2_0)
export const WASM_MEM_X_SCALE = 0x0c1000 // 16 KiB (4096 * f32 block scales)
export const WASM_MEM_X_F32_SUM = 0x0c5000 // 16 KiB (4096 * f32 block sums for Q4_1 / Q4_K)
export const WASM_MEM_Y_F32 = 0x100000 // 4 MiB (up to 1,048,576 * f32 outputs / logits)
export const WASM_MEM_WEIGHTS_BASE = 0x500000 // 5 MiB mark: start of model weight buffers

export interface WasmKernelExports {
  gemv_f32: (M: number, N: number, K: number, wPtr: number, hasResid: number, rowOffset: number, totalN: number) => void
  gemv_q1_0: (
    M: number,
    N: number,
    K: number,
    nb: number,
    quantsPtr: number,
    scalesPtr: number,
    hasResid: number,
    rowOffset: number,
    totalN: number,
  ) => void
  gemv_pq2_0: (
    M: number,
    N: number,
    K: number,
    nb: number,
    quantsPtr: number,
    scalesPtr: number,
    hasResid: number,
    rowOffset: number,
    totalN: number,
  ) => void
  gemv_ptq1_0: (
    M: number,
    N: number,
    K: number,
    nb: number,
    quantsPtr: number,
    scalesPtr: number,
    hasResid: number,
    rowOffset: number,
    totalN: number,
  ) => void
  gemv_q4_0: (
    M: number,
    N: number,
    K: number,
    nb: number,
    quantsPtr: number,
    hasResid: number,
    rowOffset: number,
    totalN: number,
  ) => void
  gemv_q4_1: (
    M: number,
    N: number,
    K: number,
    nb: number,
    quantsPtr: number,
    hasResid: number,
    rowOffset: number,
    totalN: number,
  ) => void
  gemv_q4_k: (
    M: number,
    N: number,
    K: number,
    nb: number,
    quantsPtr: number,
    hasResid: number,
    rowOffset: number,
    totalN: number,
  ) => void
  gemv_q6_k: (
    M: number,
    N: number,
    K: number,
    nb: number,
    quantsPtr: number,
    hasResid: number,
    rowOffset: number,
    totalN: number,
  ) => void
}

// Wasm Value Types
const I32 = 0x7f
const F32 = 0x7d
const V128 = 0x7b

// Standard Opcodes
const OP_BLOCK = 0x02
const OP_LOOP = 0x03
const OP_IF = 0x04
const OP_ELSE = 0x05
const OP_END = 0x0b
const OP_BR = 0x0c
const OP_BR_IF = 0x0d
const OP_LOCAL_GET = 0x20
const OP_LOCAL_SET = 0x21
const OP_LOCAL_TEE = 0x22
const OP_F32_LOAD = 0x2a
const OP_I32_LOAD8_S = 0x2c
const OP_I32_LOAD8_U = 0x2d
const OP_I32_LOAD16_U = 0x2f
const OP_F32_STORE = 0x38
const OP_I32_CONST = 0x41
const OP_F32_CONST = 0x43
const OP_I32_LT_U = 0x49
const OP_I32_GE_U = 0x4f
const OP_I32_ADD = 0x6a
const OP_I32_MUL = 0x6c
const OP_I32_AND = 0x71
const OP_I32_OR = 0x72
const OP_I32_SHL = 0x74
const OP_I32_SHR_U = 0x76
const OP_F32_ADD = 0x92
const OP_F32_SUB = 0x93
const OP_F32_MUL = 0x94
const OP_F32_CONVERT_I32_S = 0xb2

// SIMD Prefix (0xfd) + LEB128 sub-opcode
const SIMD_PREFIX = 0xfd
const SIMD_V128_LOAD = 0
const SIMD_I16X8_LOAD8X8_S = 1
const SIMD_I16X8_LOAD8X8_U = 2
const SIMD_V128_CONST = 12
const SIMD_I16X8_SPLAT = 16
const SIMD_I32X4_SPLAT = 17
const SIMD_F32X4_SPLAT = 19
const SIMD_I32X4_EXTRACT_LANE = 27
const SIMD_F32X4_EXTRACT_LANE = 31
const SIMD_V128_AND = 78
const SIMD_I16X8_EXTEND_LOW_I8X16_U = 136
const SIMD_I16X8_SHL = 139
const SIMD_I16X8_SHR_U = 141
const SIMD_I16X8_ADD = 142
const SIMD_I16X8_SUB = 145
const SIMD_I16X8_MUL = 149
const SIMD_I32X4_ADD = 174
const SIMD_I32X4_DOT_I16X8_S = 186
const SIMD_F32X4_ADD = 228
const SIMD_F32X4_MUL = 230

function uleb(val: number): number[] {
  const out: number[] = []
  let v = val >>> 0
  do {
    let b = v & 0x7f
    v >>>= 7
    if (v !== 0) b |= 0x80
    out.push(b)
  } while (v !== 0)
  return out
}

function sleb(val: number): number[] {
  const out: number[] = []
  let v = val | 0
  let more = true
  while (more) {
    let b = v & 0x7f
    v >>= 7
    if ((v === 0 && (b & 0x40) === 0) || (v === -1 && (b & 0x40) !== 0)) {
      more = false
    } else {
      b |= 0x80
    }
    out.push(b)
  }
  return out
}

function f32Bytes(val: number): number[] {
  const buf = new ArrayBuffer(4)
  new DataView(buf).setFloat32(0, val, true)
  return Array.from(new Uint8Array(buf))
}

function i16x8ConstBytes(v: number): number[] {
  const buf = new ArrayBuffer(16)
  const dv = new DataView(buf)
  for (let i = 0; i < 8; i++) dv.setInt16(i * 2, v, true)
  return [SIMD_PREFIX, ...uleb(SIMD_V128_CONST), ... new Uint8Array(buf)]
}

function simd(subOp: number, ...extra: number[]): number[] {
  return [SIMD_PREFIX, ...uleb(subOp), ...extra]
}

function simdMem(subOp: number, alignLog2: number, offset = 0): number[] {
  return [SIMD_PREFIX, ...uleb(subOp), ...uleb(alignLog2), ...uleb(offset)]
}

function memOp(op: number, alignLog2: number, offset = 0): number[] {
  return [op, ...uleb(alignLog2), ...uleb(offset)]
}

function i32Const(v: number): number[] {
  return [OP_I32_CONST, ...sleb(v)]
}

function f32Const(v: number): number[] {
  return [OP_F32_CONST, ...f32Bytes(v)]
}

function get(idx: number): number[] {
  return [OP_LOCAL_GET, ...uleb(idx)]
}

function set(idx: number): number[] {
  return [OP_LOCAL_SET, ...uleb(idx)]
}

function tee(idx: number): number[] {
  return [OP_LOCAL_TEE, ...uleb(idx)]
}

/**
 * Emit horizontal sum of an `i32x4` local `vecLocal` as an `i32` on the stack.
 */
function hsumI32x4(vecLocal: number): number[] {
  return [
    ...get(vecLocal),
    ...simd(SIMD_I32X4_EXTRACT_LANE, 0),
    ...get(vecLocal),
    ...simd(SIMD_I32X4_EXTRACT_LANE, 1),
    OP_I32_ADD,
    ...get(vecLocal),
    ...simd(SIMD_I32X4_EXTRACT_LANE, 2),
    OP_I32_ADD,
    ...get(vecLocal),
    ...simd(SIMD_I32X4_EXTRACT_LANE, 3),
    OP_I32_ADD,
  ]
}

/**
 * Emit horizontal sum of an `f32x4` local `vecLocal` as an `f32` on the stack.
 */
function hsumF32x4(vecLocal: number): number[] {
  return [
    ...get(vecLocal),
    ...simd(SIMD_F32X4_EXTRACT_LANE, 0),
    ...get(vecLocal),
    ...simd(SIMD_F32X4_EXTRACT_LANE, 1),
    OP_F32_ADD,
    ...get(vecLocal),
    ...simd(SIMD_F32X4_EXTRACT_LANE, 2),
    OP_F32_ADD,
    ...get(vecLocal),
    ...simd(SIMD_F32X4_EXTRACT_LANE, 3),
    OP_F32_ADD,
  ]
}

/**
 * Emit FP16 lookup: given an address on stack, loads u16 and looks up f32 in WASM_MEM_FP16_TABLE.
 */
function loadFp16AsF32(addrExpr: number[], byteOffset = 0): number[] {
  return [
    ...addrExpr,
    ...memOp(OP_I32_LOAD16_U, 1, byteOffset),
    ...i32Const(2),
    OP_I32_SHL,
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_FP16_TABLE),
  ]
}

/**
 * Store `fsumLocal` into `Y_F32[(m * totalN + rowOffset + n) * 4]`, adding existing value when `hasResid != 0`.
 */
function storeOutputWithResid(
  mLocal: number,
  nLocal: number,
  hasResidLocal: number,
  rowOffsetLocal: number,
  totalNLocal: number,
  fsumLocal: number,
  outAddrTemp: number,
): number[] {
  return [
    // outAddrTemp = WASM_MEM_Y_F32 + ((m * totalN + rowOffset + n) << 2)
    ...get(mLocal),
    ...get(totalNLocal),
    OP_I32_MUL,
    ...get(rowOffsetLocal),
    OP_I32_ADD,
    ...get(nLocal),
    OP_I32_ADD,
    ...i32Const(2),
    OP_I32_SHL,
    ...i32Const(WASM_MEM_Y_F32),
    OP_I32_ADD,
    ...tee(outAddrTemp),
    ...get(hasResidLocal),
    OP_IF,
    F32,
    ...get(outAddrTemp),
    ...memOp(OP_F32_LOAD, 2, 0),
    ...get(fsumLocal),
    OP_F32_ADD,
    OP_ELSE,
    ...get(fsumLocal),
    OP_END,
    ...memOp(OP_F32_STORE, 2, 0),
  ]
}

/**
 * Build the `gemv_f32` function body.
 * Params: 0:M, 1:N, 2:K, 3:wPtr, 4:hasResid, 5:rowOffset, 6:totalN
 */
function buildGemvF32(): number[] {
  // Locals: 7..12: i32 (m, n, k, xPtr, wRowPtr, outAddr), 13: f32 (fsum), 14: v128 (vacc)
  const m = 7,
    n = 8,
    k = 9,
    xPtr = 10,
    wRowPtr = 11,
    outAddr = 12,
    fsum = 13,
    vacc = 14

  const body: number[] = [
    ...i32Const(0),
    ...set(m),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(m),
    ...get(0), // M
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...set(n),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...get(1), // N
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    // vacc = 0
    ...f32Const(0),
    ...simd(SIMD_F32X4_SPLAT),
    ...set(vacc),

    // xPtr = WASM_MEM_X_F32 + (m * K) * 4
    ...get(m),
    ...get(2),
    OP_I32_MUL,
    ...i32Const(2),
    OP_I32_SHL,
    ...i32Const(WASM_MEM_X_F32),
    OP_I32_ADD,
    ...set(xPtr),

    // wRowPtr = wPtr + (n * K) * 4
    ...get(3),
    ...get(n),
    ...get(2),
    OP_I32_MUL,
    ...i32Const(2),
    OP_I32_SHL,
    OP_I32_ADD,
    ...set(wRowPtr),

    ...i32Const(0),
    ...set(k),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(k),
    ...get(2),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...get(vacc),
    ...get(xPtr),
    ...simdMem(SIMD_V128_LOAD, 2, 0),
    ...get(wRowPtr),
    ...simdMem(SIMD_V128_LOAD, 2, 0),
    ...simd(SIMD_F32X4_MUL),
    ...simd(SIMD_F32X4_ADD),
    ...set(vacc),

    ...get(xPtr),
    ...i32Const(16),
    OP_I32_ADD,
    ...set(xPtr),
    ...get(wRowPtr),
    ...i32Const(16),
    OP_I32_ADD,
    ...set(wRowPtr),
    ...get(k),
    ...i32Const(4),
    OP_I32_ADD,
    ...set(k),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...hsumF32x4(vacc),
    ...set(fsum),
    ...storeOutputWithResid(m, n, 4, 5, 6, fsum, outAddr),

    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(m),
    OP_BR,
    0,
    OP_END,
    OP_END,
    OP_END,
  ]

  return encodeFunc([
    [6, I32],
    [1, F32],
    [1, V128],
  ], body)
}

/**
 * Build the `gemv_q1_0` function body with 4-row unrolling (`ROWS = 4`) + 1-row tail.
 * Params: 0:M, 1:N, 2:K, 3:nb, 4:quantsPtr, 5:scalesPtr, 6:hasResid, 7:rowOffset, 8:totalN
 */
function buildGemvQ1_0(): number[] {
  // i32 locals (9..24): m, n, b, i, xBlkPtr, qPtr0, qPtr1, qPtr2, qPtr3, sPtr0, sPtr1, sPtr2, sPtr3, qStride, sStride, outAddr
  const m = 9,
    n = 10,
    b = 11,
    i = 12,
    xBlkPtr = 13,
    qPtr0 = 14,
    qPtr1 = 15,
    qPtr2 = 16,
    qPtr3 = 17,
    sPtr0 = 18,
    sPtr1 = 19,
    sPtr2 = 20,
    sPtr3 = 21,
    qStride = 22,
    sStride = 23,
    outAddr = 24
  // f32 locals (25..29): fsum0, fsum1, fsum2, fsum3, xScale
  const fsum0 = 25,
    fsum1 = 26,
    fsum2 = 27,
    fsum3 = 28,
    xScale = 29
  // v128 locals (30..34): iacc0, iacc1, iacc2, iacc3, xv
  const iacc0 = 30,
    iacc1 = 31,
    iacc2 = 32,
    iacc3 = 33,
    xv = 34

  const dotLutByte = (qPtrLocal: number, iaccLocal: number): number[] => [
    ...get(iaccLocal),
    ...get(qPtrLocal),
    ...get(i),
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_U, 0, 0),
    ...i32Const(3),
    OP_I32_SHL,
    ...simdMem(SIMD_I16X8_LOAD8X8_S, 3, WASM_MEM_Q1_LUT),
    ...get(xv),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),
    ...set(iaccLocal),
  ]

  const body: number[] = [
    // qStride = nb * 16, sStride = nb * 2
    ...get(3),
    ...i32Const(4),
    OP_I32_SHL,
    ...set(qStride),
    ...get(3),
    ...i32Const(1),
    OP_I32_SHL,
    ...set(sStride),

    ...i32Const(0),
    ...set(m),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(m),
    ...get(0),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...set(n),
    // 4-row main loop: while (n + 4 <= N) => !(n + 4 > N) => (n + 3 < N)
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...i32Const(3),
    OP_I32_ADD,
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum0),
    ...f32Const(0),
    ...set(fsum1),
    ...f32Const(0),
    ...set(fsum2),
    ...f32Const(0),
    ...set(fsum3),

    // Initialize row pointers
    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...tee(qPtr0),
    ...get(qStride),
    OP_I32_ADD,
    ...tee(qPtr1),
    ...get(qStride),
    OP_I32_ADD,
    ...tee(qPtr2),
    ...get(qStride),
    OP_I32_ADD,
    ...set(qPtr3),

    ...get(5),
    ...get(n),
    ...get(sStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...tee(sPtr0),
    ...get(sStride),
    OP_I32_ADD,
    ...tee(sPtr1),
    ...get(sStride),
    OP_I32_ADD,
    ...tee(sPtr2),
    ...get(sStride),
    OP_I32_ADD,
    ...set(sPtr3),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3), // nb
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...simd(SIMD_I32X4_SPLAT),
    ...tee(iacc0),
    ...tee(iacc1),
    ...tee(iacc2),
    ...set(iacc3),

    // xBlkPtr = WASM_MEM_X_Q16 + ((m * nb + b) << 8)
    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(8),
    OP_I32_SHL,
    ...i32Const(WASM_MEM_X_Q16),
    OP_I32_ADD,
    ...set(xBlkPtr),

    ...i32Const(0),
    ...set(i),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(i),
    ...i32Const(16),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    // xv = v128.load(xBlkPtr + (i << 4))
    ...get(xBlkPtr),
    ...get(i),
    ...i32Const(4),
    OP_I32_SHL,
    OP_I32_ADD,
    ...simdMem(SIMD_V128_LOAD, 4, 0),
    ...set(xv),

    ...dotLutByte(qPtr0, iacc0),
    ...dotLutByte(qPtr1, iacc1),
    ...dotLutByte(qPtr2, iacc2),
    ...dotLutByte(qPtr3, iacc3),

    ...get(i),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(i),
    OP_BR,
    0,
    OP_END,
    OP_END,

    // xScale = f32.load(WASM_MEM_X_SCALE + (m * nb + b) * 4)
    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(2),
    OP_I32_SHL,
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    ...set(xScale),

    // Accumulate scaled block dot for all 4 rows
    ...get(fsum0),
    ...hsumI32x4(iacc0),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr0), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum0),

    ...get(fsum1),
    ...hsumI32x4(iacc1),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr1), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum1),

    ...get(fsum2),
    ...hsumI32x4(iacc2),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr2), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum2),

    ...get(fsum3),
    ...hsumI32x4(iacc3),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr3), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum3),

    // Advance block pointers
    ...get(qPtr0),
    ...i32Const(16),
    OP_I32_ADD,
    ...set(qPtr0),
    ...get(qPtr1),
    ...i32Const(16),
    OP_I32_ADD,
    ...set(qPtr1),
    ...get(qPtr2),
    ...i32Const(16),
    OP_I32_ADD,
    ...set(qPtr2),
    ...get(qPtr3),
    ...i32Const(16),
    OP_I32_ADD,
    ...set(qPtr3),
    ...get(sPtr0),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr0),
    ...get(sPtr1),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr1),
    ...get(sPtr2),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr2),
    ...get(sPtr3),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr3),

    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    // Store 4 rows
    ...storeOutputWithResid(m, n, 6, 7, 8, fsum0, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    ...storeOutputWithResid(m, n, 6, 7, 8, fsum1, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    ...storeOutputWithResid(m, n, 6, 7, 8, fsum2, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    ...storeOutputWithResid(m, n, 6, 7, 8, fsum3, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),

    OP_BR,
    0,
    OP_END,
    OP_END,

    // 1-row tail loop: while (n < N)
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum0),
    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(qPtr0),
    ...get(5),
    ...get(n),
    ...get(sStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(sPtr0),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...simd(SIMD_I32X4_SPLAT),
    ...set(iacc0),
    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(8),
    OP_I32_SHL,
    ...i32Const(WASM_MEM_X_Q16),
    OP_I32_ADD,
    ...set(xBlkPtr),

    ...i32Const(0),
    ...set(i),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(i),
    ...i32Const(16),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...get(xBlkPtr),
    ...get(i),
    ...i32Const(4),
    OP_I32_SHL,
    OP_I32_ADD,
    ...simdMem(SIMD_V128_LOAD, 4, 0),
    ...set(xv),
    ...dotLutByte(qPtr0, iacc0),

    ...get(i),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(i),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(2),
    OP_I32_SHL,
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    ...set(xScale),

    ...get(fsum0),
    ...hsumI32x4(iacc0),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr0), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum0),

    ...get(qPtr0),
    ...i32Const(16),
    OP_I32_ADD,
    ...set(qPtr0),
    ...get(sPtr0),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr0),
    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...storeOutputWithResid(m, n, 6, 7, 8, fsum0, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(m),
    OP_BR,
    0,
    OP_END,
    OP_END,
    OP_END,
  ]

  return encodeFunc([
    [16, I32],
    [5, F32],
    [5, V128],
  ], body)
}

/**
 * Build the `gemv_pq2_0` function body with 4-row unrolling (`ROWS = 4`) + 1-row tail.
 * Params: 0:M, 1:N, 2:K, 3:nb, 4:quantsPtr, 5:scalesPtr, 6:hasResid, 7:rowOffset, 8:totalN
 */
function buildGemvPQ2_0(): number[] {
  const m = 9,
    n = 10,
    b = 11,
    sub = 12,
    xSubPtr = 13,
    qPtr0 = 14,
    qPtr1 = 15,
    qPtr2 = 16,
    qPtr3 = 17,
    sPtr0 = 18,
    sPtr1 = 19,
    sPtr2 = 20,
    sPtr3 = 21,
    qStride = 22,
    sStride = 23,
    outAddr = 24
  const fsum0 = 25,
    fsum1 = 26,
    fsum2 = 27,
    fsum3 = 28,
    xScale = 29
  const iacc0 = 30,
    iacc1 = 31,
    iacc2 = 32,
    iacc3 = 33,
    x0 = 34,
    x1 = 35,
    x2 = 36,
    x3 = 37,
    v = 38,
    c3 = 39,
    c1 = 40

  const dotPq2Sub = (qPtrLocal: number, iaccLocal: number): number[] => [
    ...get(qPtrLocal),
    ...get(sub),
    ...i32Const(3),
    OP_I32_SHL,
    OP_I32_ADD,
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 3, 0),
    ...set(v),

    ...get(iaccLocal),
    // (v & 3) - 1 dot x0
    ...get(v),
    ...get(c3),
    ...simd(SIMD_V128_AND),
    ...get(c1),
    ...simd(SIMD_I16X8_SUB),
    ...get(x0),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    // ((v >> 2) & 3) - 1 dot x1
    ...get(v),
    ...i32Const(2),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(c3),
    ...simd(SIMD_V128_AND),
    ...get(c1),
    ...simd(SIMD_I16X8_SUB),
    ...get(x1),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),
    // ((v >> 4) & 3) - 1 dot x2
    ...get(v),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(c3),
    ...simd(SIMD_V128_AND),
    ...get(c1),
    ...simd(SIMD_I16X8_SUB),
    ...get(x2),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    // ((v >> 6) - 1) dot x3
    ...get(v),
    ...i32Const(6),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(c1),
    ...simd(SIMD_I16X8_SUB),
    ...get(x3),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),
    ...simd(SIMD_I32X4_ADD),
    ...simd(SIMD_I32X4_ADD),
    ...set(iaccLocal),
  ]

  const body: number[] = [
    ...i16x8ConstBytes(3),
    ...set(c3),
    ...i16x8ConstBytes(1),
    ...set(c1),

    // qStride = nb * 32, sStride = nb * 2
    ...get(3),
    ...i32Const(5),
    OP_I32_SHL,
    ...set(qStride),
    ...get(3),
    ...i32Const(1),
    OP_I32_SHL,
    ...set(sStride),

    ...i32Const(0),
    ...set(m),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(m),
    ...get(0),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...set(n),
    // 4-row main loop
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...i32Const(3),
    OP_I32_ADD,
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum0),
    ...f32Const(0),
    ...set(fsum1),
    ...f32Const(0),
    ...set(fsum2),
    ...f32Const(0),
    ...set(fsum3),

    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...tee(qPtr0),
    ...get(qStride),
    OP_I32_ADD,
    ...tee(qPtr1),
    ...get(qStride),
    OP_I32_ADD,
    ...tee(qPtr2),
    ...get(qStride),
    OP_I32_ADD,
    ...set(qPtr3),

    ...get(5),
    ...get(n),
    ...get(sStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...tee(sPtr0),
    ...get(sStride),
    OP_I32_ADD,
    ...tee(sPtr1),
    ...get(sStride),
    OP_I32_ADD,
    ...tee(sPtr2),
    ...get(sStride),
    OP_I32_ADD,
    ...set(sPtr3),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...simd(SIMD_I32X4_SPLAT),
    ...tee(iacc0),
    ...tee(iacc1),
    ...tee(iacc2),
    ...set(iacc3),

    ...i32Const(0),
    ...set(sub),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(sub),
    ...i32Const(4),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    // xSubPtr = WASM_MEM_X_PQ2_Q16 + ((m * nb + b) << 8) + (sub << 6)
    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(8),
    OP_I32_SHL,
    ...get(sub),
    ...i32Const(6),
    OP_I32_SHL,
    OP_I32_ADD,
    ...i32Const(WASM_MEM_X_PQ2_Q16),
    OP_I32_ADD,
    ...tee(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 0),
    ...set(x0),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 16),
    ...set(x1),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 32),
    ...set(x2),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 48),
    ...set(x3),

    ...dotPq2Sub(qPtr0, iacc0),
    ...dotPq2Sub(qPtr1, iacc1),
    ...dotPq2Sub(qPtr2, iacc2),
    ...dotPq2Sub(qPtr3, iacc3),

    ...get(sub),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(sub),
    OP_BR,
    0,
    OP_END,
    OP_END,

    // xScale
    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(2),
    OP_I32_SHL,
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    ...set(xScale),

    ...get(fsum0),
    ...hsumI32x4(iacc0),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr0), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum0),

    ...get(fsum1),
    ...hsumI32x4(iacc1),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr1), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum1),

    ...get(fsum2),
    ...hsumI32x4(iacc2),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr2), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum2),

    ...get(fsum3),
    ...hsumI32x4(iacc3),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr3), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum3),

    ...get(qPtr0),
    ...i32Const(32),
    OP_I32_ADD,
    ...set(qPtr0),
    ...get(qPtr1),
    ...i32Const(32),
    OP_I32_ADD,
    ...set(qPtr1),
    ...get(qPtr2),
    ...i32Const(32),
    OP_I32_ADD,
    ...set(qPtr2),
    ...get(qPtr3),
    ...i32Const(32),
    OP_I32_ADD,
    ...set(qPtr3),
    ...get(sPtr0),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr0),
    ...get(sPtr1),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr1),
    ...get(sPtr2),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr2),
    ...get(sPtr3),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr3),

    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...storeOutputWithResid(m, n, 6, 7, 8, fsum0, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    ...storeOutputWithResid(m, n, 6, 7, 8, fsum1, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    ...storeOutputWithResid(m, n, 6, 7, 8, fsum2, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    ...storeOutputWithResid(m, n, 6, 7, 8, fsum3, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    // 1-row tail loop
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum0),
    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(qPtr0),
    ...get(5),
    ...get(n),
    ...get(sStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(sPtr0),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...simd(SIMD_I32X4_SPLAT),
    ...set(iacc0),

    ...i32Const(0),
    ...set(sub),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(sub),
    ...i32Const(4),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(8),
    OP_I32_SHL,
    ...get(sub),
    ...i32Const(6),
    OP_I32_SHL,
    OP_I32_ADD,
    ...i32Const(WASM_MEM_X_PQ2_Q16),
    OP_I32_ADD,
    ...tee(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 0),
    ...set(x0),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 16),
    ...set(x1),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 32),
    ...set(x2),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 48),
    ...set(x3),

    ...dotPq2Sub(qPtr0, iacc0),

    ...get(sub),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(sub),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(2),
    OP_I32_SHL,
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    ...set(xScale),

    ...get(fsum0),
    ...hsumI32x4(iacc0),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr0), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum0),

    ...get(qPtr0),
    ...i32Const(32),
    OP_I32_ADD,
    ...set(qPtr0),
    ...get(sPtr0),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr0),
    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...storeOutputWithResid(m, n, 6, 7, 8, fsum0, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(m),
    OP_BR,
    0,
    OP_END,
    OP_END,
    OP_END,
  ]

  return encodeFunc([
    [16, I32],
    [5, F32],
    [11, V128],
  ], body)
}

/**
 * Build the `gemv_ptq1_0` function body.
 * Params: 0:M, 1:N, 2:K, 3:nb, 4:quantsPtr, 5:scalesPtr, 6:hasResid, 7:rowOffset, 8:totalN
 */
function buildGemvPTQ1_0(): number[] {
  const m = 9,
    n = 10,
    b = 11,
    xBlkPtr = 12,
    qPtr = 13,
    sPtr = 14,
    qStride = 15,
    sStride = 16,
    outAddr = 17
  const fsum = 18,
    xScale = 19
  const iacc = 20,
    vLo = 21,
    vHi = 22,
    vMid = 23,
    c255 = 24,
    c3 = 25,
    c1 = 26,
    p3 = 27,
    p9 = 28,
    p27 = 29,
    p81 = 30,
    pTail = 31

  const unpackTritVec = (vLocal: number, pLocal: number | null): number[] => {
    const qExpr =
      pLocal === null
        ? get(vLocal)
        : [...get(vLocal), ...get(pLocal), ...simd(SIMD_I16X8_MUL), ...get(c255), ...simd(SIMD_V128_AND)]
    return [
      ...qExpr,
      ...get(c3),
      ...simd(SIMD_I16X8_MUL),
      ...i32Const(8),
      ...simd(SIMD_I16X8_SHR_U),
      ...get(c1),
      ...simd(SIMD_I16X8_SUB),
    ]
  }

  const pLocals: Array<number | null> = [null, p3, p9, p27, p81]
  const stage1And2Ops: number[] = []
  for (let stage = 0; stage < 5; stage++) {
    const pLoc = pLocals[stage]
    // Stage 1 low 8 (elements stage*16 + 0..7)
    stage1And2Ops.push(
      ...get(iacc),
      ...unpackTritVec(vLo, pLoc),
      ...get(xBlkPtr),
      ...simdMem(SIMD_V128_LOAD, 4, stage * 32),
      ...simd(SIMD_I32X4_DOT_I16X8_S),
      ...simd(SIMD_I32X4_ADD),
      ...set(iacc),
    )
    // Stage 1 high 8 (elements stage*16 + 8..15)
    stage1And2Ops.push(
      ...get(iacc),
      ...unpackTritVec(vHi, pLoc),
      ...get(xBlkPtr),
      ...simdMem(SIMD_V128_LOAD, 4, stage * 32 + 16),
      ...simd(SIMD_I32X4_DOT_I16X8_S),
      ...simd(SIMD_I32X4_ADD),
      ...set(iacc),
    )
    // Stage 2 (elements 80 + stage*8 + 0..7)
    stage1And2Ops.push(
      ...get(iacc),
      ...unpackTritVec(vMid, pLoc),
      ...get(xBlkPtr),
      ...simdMem(SIMD_V128_LOAD, 4, 160 + stage * 16),
      ...simd(SIMD_I32X4_DOT_I16X8_S),
      ...simd(SIMD_I32X4_ADD),
      ...set(iacc),
    )
  }

  const body: number[] = [
    ...i16x8ConstBytes(255),
    ...set(c255),
    ...i16x8ConstBytes(3),
    ...tee(c3),
    ...set(p3),
    ...i16x8ConstBytes(1),
    ...set(c1),
    ...i16x8ConstBytes(9),
    ...set(p9),
    ...i16x8ConstBytes(27),
    ...set(p27),
    ...i16x8ConstBytes(81),
    ...set(p81),
    ...i32Const(WASM_MEM_CONSTS),
    ...simdMem(SIMD_V128_LOAD, 4, 0),
    ...set(pTail),

    // qStride = nb * 26, sStride = nb * 2
    ...get(3),
    ...i32Const(26),
    OP_I32_MUL,
    ...set(qStride),
    ...get(3),
    ...i32Const(1),
    OP_I32_SHL,
    ...set(sStride),

    ...i32Const(0),
    ...set(m),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(m),
    ...get(0),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...set(n),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum),
    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(qPtr),
    ...get(5),
    ...get(n),
    ...get(sStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(sPtr),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...simd(SIMD_I32X4_SPLAT),
    ...set(iacc),

    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(8),
    OP_I32_SHL,
    ...i32Const(WASM_MEM_X_Q16),
    OP_I32_ADD,
    ...set(xBlkPtr),

    ...get(qPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 0),
    ...set(vLo),
    ...get(qPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 8),
    ...set(vHi),
    ...get(qPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 16),
    ...set(vMid),

    ...stage1And2Ops,

    // Stage 3 tail (elements 120..127 from 2 bytes at qPtr + 24)
    ...get(iacc),
    ...get(qPtr),
    ...memOp(OP_I32_LOAD16_U, 0, 24),
    ...simd(SIMD_I16X8_SPLAT),
    ...simd(SIMD_I16X8_EXTEND_LOW_I8X16_U),
    ...get(pTail),
    ...simd(SIMD_I16X8_MUL),
    ...get(c255),
    ...simd(SIMD_V128_AND),
    ...get(c3),
    ...simd(SIMD_I16X8_MUL),
    ...i32Const(8),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(c1),
    ...simd(SIMD_I16X8_SUB),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 240),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),
    ...set(iacc),

    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(2),
    OP_I32_SHL,
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    ...set(xScale),

    ...get(fsum),
    ...hsumI32x4(iacc),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(sPtr), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum),

    ...get(qPtr),
    ...i32Const(26),
    OP_I32_ADD,
    ...set(qPtr),
    ...get(sPtr),
    ...i32Const(2),
    OP_I32_ADD,
    ...set(sPtr),
    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...storeOutputWithResid(m, n, 6, 7, 8, fsum, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(m),
    OP_BR,
    0,
    OP_END,
    OP_END,
    OP_END,
  ]

  return encodeFunc([
    [9, I32],
    [2, F32],
    [12, V128],
  ], body)
}

/**
 * Build the `gemv_q4_0` function body with 4-row unrolling (`ROWS = 4`) + 1-row tail.
 * Params: 0:M, 1:N, 2:K, 3:nb, 4:quantsPtr, 5:hasResid, 6:rowOffset, 7:totalN
 */
function buildGemvQ4_0(): number[] {
  const m = 8,
    n = 9,
    b = 10,
    xBlkPtr = 11,
    qPtr0 = 12,
    qPtr1 = 13,
    qPtr2 = 14,
    qPtr3 = 15,
    qStride = 16,
    outAddr = 17
  const fsum0 = 18,
    fsum1 = 19,
    fsum2 = 20,
    fsum3 = 21,
    xScale = 22
  const iacc = 23,
    x0 = 24,
    x1 = 25,
    x2 = 26,
    x3 = 27,
    v0 = 28,
    v1 = 29,
    c15 = 30,
    c8 = 31

  const dotQ40Row = (qPtrLocal: number, fsumLocal: number): number[] => [
    ...get(qPtrLocal),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 2),
    ...set(v0),
    ...get(qPtrLocal),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 10),
    ...set(v1),

    // (v0 & 15) - 8 dot x0 + (v1 & 15) - 8 dot x1 + (v0 >> 4) - 8 dot x2 + (v1 >> 4) - 8 dot x3
    ...get(v0),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(c8),
    ...simd(SIMD_I16X8_SUB),
    ...get(x0),
    ...simd(SIMD_I32X4_DOT_I16X8_S),

    ...get(v1),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(c8),
    ...simd(SIMD_I16X8_SUB),
    ...get(x1),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),

    ...get(v0),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(c8),
    ...simd(SIMD_I16X8_SUB),
    ...get(x2),
    ...simd(SIMD_I32X4_DOT_I16X8_S),

    ...get(v1),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(c8),
    ...simd(SIMD_I16X8_SUB),
    ...get(x3),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),

    ...simd(SIMD_I32X4_ADD),
    ...set(iacc),

    ...get(fsumLocal),
    ...hsumI32x4(iacc),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(qPtrLocal), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsumLocal),
  ]

  const body: number[] = [
    ...i16x8ConstBytes(15),
    ...set(c15),
    ...i16x8ConstBytes(8),
    ...set(c8),

    // qStride = nb * 18
    ...get(3),
    ...i32Const(18),
    OP_I32_MUL,
    ...set(qStride),

    ...i32Const(0),
    ...set(m),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(m),
    ...get(0),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...set(n),
    // 4-row main loop
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...i32Const(3),
    OP_I32_ADD,
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum0),
    ...f32Const(0),
    ...set(fsum1),
    ...f32Const(0),
    ...set(fsum2),
    ...f32Const(0),
    ...set(fsum3),

    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...tee(qPtr0),
    ...get(qStride),
    OP_I32_ADD,
    ...tee(qPtr1),
    ...get(qStride),
    OP_I32_ADD,
    ...tee(qPtr2),
    ...get(qStride),
    OP_I32_ADD,
    ...set(qPtr3),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    // xBlkPtr = WASM_MEM_X_Q16 + ((m * nb + b) << 6)
    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(6),
    OP_I32_SHL,
    ...i32Const(WASM_MEM_X_Q16),
    OP_I32_ADD,
    ...tee(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 0),
    ...set(x0),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 16),
    ...set(x1),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 32),
    ...set(x2),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 48),
    ...set(x3),

    // xScale
    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(2),
    OP_I32_SHL,
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    ...set(xScale),

    ...dotQ40Row(qPtr0, fsum0),
    ...dotQ40Row(qPtr1, fsum1),
    ...dotQ40Row(qPtr2, fsum2),
    ...dotQ40Row(qPtr3, fsum3),

    ...get(qPtr0),
    ...i32Const(18),
    OP_I32_ADD,
    ...set(qPtr0),
    ...get(qPtr1),
    ...i32Const(18),
    OP_I32_ADD,
    ...set(qPtr1),
    ...get(qPtr2),
    ...i32Const(18),
    OP_I32_ADD,
    ...set(qPtr2),
    ...get(qPtr3),
    ...i32Const(18),
    OP_I32_ADD,
    ...set(qPtr3),

    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...storeOutputWithResid(m, n, 5, 6, 7, fsum0, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    ...storeOutputWithResid(m, n, 5, 6, 7, fsum1, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    ...storeOutputWithResid(m, n, 5, 6, 7, fsum2, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    ...storeOutputWithResid(m, n, 5, 6, 7, fsum3, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    // 1-row tail loop
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum0),
    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(qPtr0),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(6),
    OP_I32_SHL,
    ...i32Const(WASM_MEM_X_Q16),
    OP_I32_ADD,
    ...tee(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 0),
    ...set(x0),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 16),
    ...set(x1),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 32),
    ...set(x2),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 48),
    ...set(x3),

    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(2),
    OP_I32_SHL,
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    ...set(xScale),

    ...dotQ40Row(qPtr0, fsum0),

    ...get(qPtr0),
    ...i32Const(18),
    OP_I32_ADD,
    ...set(qPtr0),
    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...storeOutputWithResid(m, n, 5, 6, 7, fsum0, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(m),
    OP_BR,
    0,
    OP_END,
    OP_END,
    OP_END,
  ]

  return encodeFunc([
    [10, I32],
    [5, F32],
    [9, V128],
  ], body)
}

/**
 * Build the `gemv_q4_1` function body.
 * Params: 0:M, 1:N, 2:K, 3:nb, 4:quantsPtr, 5:hasResid, 6:rowOffset, 7:totalN
 */
function buildGemvQ4_1(): number[] {
  const m = 8,
    n = 9,
    b = 10,
    xBlkPtr = 11,
    qPtr0 = 12,
    qStride = 13,
    outAddr = 14,
    blkIdx = 15
  const fsum0 = 16,
    xScale = 17,
    xSum = 18
  const iacc = 19,
    x0 = 20,
    x1 = 21,
    x2 = 22,
    x3 = 23,
    v0 = 24,
    v1 = 25,
    c15 = 26

  const body: number[] = [
    ...i16x8ConstBytes(15),
    ...set(c15),

    // qStride = nb * 20
    ...get(3),
    ...i32Const(20),
    OP_I32_MUL,
    ...set(qStride),

    ...i32Const(0),
    ...set(m),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(m),
    ...get(0),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...set(n),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum0),
    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(qPtr0),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...tee(blkIdx),
    ...i32Const(6),
    OP_I32_SHL,
    ...i32Const(WASM_MEM_X_Q16),
    OP_I32_ADD,
    ...tee(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 0),
    ...set(x0),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 16),
    ...set(x1),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 32),
    ...set(x2),
    ...get(xBlkPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 48),
    ...set(x3),

    ...get(blkIdx),
    ...i32Const(2),
    OP_I32_SHL,
    ...tee(blkIdx),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    ...set(xScale),
    ...get(blkIdx),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_F32_SUM),
    ...set(xSum),

    ...get(qPtr0),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 4),
    ...set(v0),
    ...get(qPtr0),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 12),
    ...set(v1),

    ...get(v0),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(x0),
    ...simd(SIMD_I32X4_DOT_I16X8_S),

    ...get(v1),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(x1),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),

    ...get(v0),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(x2),
    ...simd(SIMD_I32X4_DOT_I16X8_S),

    ...get(v1),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(x3),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),

    ...simd(SIMD_I32X4_ADD),
    ...set(iacc),

    // fsum0 += hsum(iacc) * d * xScale + m * xSum
    ...get(fsum0),
    ...hsumI32x4(iacc),
    OP_F32_CONVERT_I32_S,
    ...loadFp16AsF32(get(qPtr0), 0),
    OP_F32_MUL,
    ...get(xScale),
    OP_F32_MUL,
    ...loadFp16AsF32(get(qPtr0), 2),
    ...get(xSum),
    OP_F32_MUL,
    OP_F32_ADD,
    OP_F32_ADD,
    ...set(fsum0),

    ...get(qPtr0),
    ...i32Const(20),
    OP_I32_ADD,
    ...set(qPtr0),
    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...storeOutputWithResid(m, n, 5, 6, 7, fsum0, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(m),
    OP_BR,
    0,
    OP_END,
    OP_END,
    OP_END,
  ]

  return encodeFunc([
    [8, I32],
    [3, F32],
    [8, V128],
  ], body)
}

/**
 * Build the `gemv_q4_k` function body.
 * Params: 0:M, 1:N, 2:K, 3:nb, 4:quantsPtr, 5:hasResid, 6:rowOffset, 7:totalN
 */
function buildGemvQ4_K(): number[] {
  const m = 8,
    n = 9,
    b = 10,
    j = 11,
    is = 12,
    subIdx = 13,
    xSubPtr = 14,
    qPtr = 15,
    qsPtr = 16,
    qStride = 17,
    outAddr = 18,
    sc0 = 19,
    mn0 = 20,
    sc1 = 21,
    mn1 = 22
  const fsum = 23,
    d = 24,
    dmin = 25
  const iacc0 = 26,
    iacc1 = 27,
    v0 = 28,
    v1 = 29,
    v2 = 30,
    v3 = 31,
    c15 = 32

  const decodeScMn = (idxExpr: number[], scLocal: number, mnLocal: number): number[] => [
    ...idxExpr,
    ...i32Const(4),
    OP_I32_LT_U,
    OP_IF,
    0x40,
    ...get(qPtr),
    ...idxExpr,
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_U, 0, 4),
    ...i32Const(63),
    OP_I32_AND,
    ...set(scLocal),
    ...get(qPtr),
    ...idxExpr,
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_U, 0, 8),
    ...i32Const(63),
    OP_I32_AND,
    ...set(mnLocal),
    OP_ELSE,
    ...get(qPtr),
    ...idxExpr,
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_U, 0, 8),
    ...i32Const(15),
    OP_I32_AND,
    ...get(qPtr),
    ...idxExpr,
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_U, 0, 0),
    ...i32Const(6),
    OP_I32_SHR_U,
    ...i32Const(4),
    OP_I32_SHL,
    OP_I32_OR,
    ...set(scLocal),

    ...get(qPtr),
    ...idxExpr,
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_U, 0, 8),
    ...i32Const(4),
    OP_I32_SHR_U,
    ...get(qPtr),
    ...idxExpr,
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_U, 0, 4),
    ...i32Const(6),
    OP_I32_SHR_U,
    ...i32Const(4),
    OP_I32_SHL,
    OP_I32_OR,
    ...set(mnLocal),
    OP_END,
  ]

  const body: number[] = [
    ...i16x8ConstBytes(15),
    ...set(c15),

    // qStride = nb * 144
    ...get(3),
    ...i32Const(144),
    OP_I32_MUL,
    ...set(qStride),

    ...i32Const(0),
    ...set(m),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(m),
    ...get(0),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...set(n),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum),
    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(qPtr),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...loadFp16AsF32(get(qPtr), 0),
    ...set(d),
    ...loadFp16AsF32(get(qPtr), 2),
    ...set(dmin),

    ...i32Const(0),
    ...set(j),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(j),
    ...i32Const(4),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...get(j),
    ...i32Const(1),
    OP_I32_SHL,
    ...set(is),

    ...decodeScMn(get(is), sc0, mn0),
    ...decodeScMn([...get(is), ...i32Const(1), OP_I32_ADD], sc1, mn1),

    // qsPtr = qPtr + 16 + j * 32
    ...get(qPtr),
    ...get(j),
    ...i32Const(5),
    OP_I32_SHL,
    OP_I32_ADD,
    ...tee(qsPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 16),
    ...set(v0),
    ...get(qsPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 24),
    ...set(v1),
    ...get(qsPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 32),
    ...set(v2),
    ...get(qsPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 40),
    ...set(v3),

    // subIdx = (m * nb + b) * 8 + is
    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(3),
    OP_I32_SHL,
    ...get(is),
    OP_I32_ADD,
    ...tee(subIdx),
    ...i32Const(6),
    OP_I32_SHL,
    ...i32Const(WASM_MEM_X_Q16),
    OP_I32_ADD,
    ...set(xSubPtr),

    // Low nibbles dot subIdx
    ...get(v0),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 0),
    ...simd(SIMD_I32X4_DOT_I16X8_S),

    ...get(v1),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 16),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),

    ...get(v2),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 32),
    ...simd(SIMD_I32X4_DOT_I16X8_S),

    ...get(v3),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 48),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),

    ...simd(SIMD_I32X4_ADD),
    ...set(iacc0),

    // High nibbles dot (subIdx + 1)
    ...get(v0),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 64),
    ...simd(SIMD_I32X4_DOT_I16X8_S),

    ...get(v1),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 80),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),

    ...get(v2),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 96),
    ...simd(SIMD_I32X4_DOT_I16X8_S),

    ...get(v3),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(xSubPtr),
    ...simdMem(SIMD_V128_LOAD, 4, 112),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),

    ...simd(SIMD_I32X4_ADD),
    ...set(iacc1),

    // subIdx <<= 2 (byte offset into X_SCALE and X_F32_SUM)
    ...get(subIdx),
    ...i32Const(2),
    OP_I32_SHL,
    ...set(subIdx),

    // fsum += d * sc0 * x_scale[sub0] * hsum(iacc0) - dmin * mn0 * x_sum[sub0]
    ...get(fsum),
    ...get(d),
    ...get(sc0),
    OP_F32_CONVERT_I32_S,
    OP_F32_MUL,
    ...get(subIdx),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    OP_F32_MUL,
    ...hsumI32x4(iacc0),
    OP_F32_CONVERT_I32_S,
    OP_F32_MUL,
    ...get(dmin),
    ...get(mn0),
    OP_F32_CONVERT_I32_S,
    OP_F32_MUL,
    ...get(subIdx),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_F32_SUM),
    OP_F32_MUL,
    OP_F32_SUB,
    OP_F32_ADD,

    // + d * sc1 * x_scale[sub1] * hsum(iacc1) - dmin * mn1 * x_sum[sub1]
    ...get(d),
    ...get(sc1),
    OP_F32_CONVERT_I32_S,
    OP_F32_MUL,
    ...get(subIdx),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE + 4),
    OP_F32_MUL,
    ...hsumI32x4(iacc1),
    OP_F32_CONVERT_I32_S,
    OP_F32_MUL,
    ...get(dmin),
    ...get(mn1),
    OP_F32_CONVERT_I32_S,
    OP_F32_MUL,
    ...get(subIdx),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_F32_SUM + 4),
    OP_F32_MUL,
    OP_F32_SUB,
    OP_F32_ADD,
    ...set(fsum),

    ...get(j),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(j),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(qPtr),
    ...i32Const(144),
    OP_I32_ADD,
    ...set(qPtr),
    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...storeOutputWithResid(m, n, 5, 6, 7, fsum, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(m),
    OP_BR,
    0,
    OP_END,
    OP_END,
    OP_END,
  ]

  return encodeFunc([
    [15, I32],
    [3, F32],
    [7, V128],
  ], body)
}

/**
 * Build the `gemv_q6_k` function body.
 * Params: 0:M, 1:N, 2:K, 3:nb, 4:quantsPtr, 5:hasResid, 6:rowOffset, 7:totalN
 */
function buildGemvQ6_K(): number[] {
  const m = 8,
    n = 9,
    b = 10,
    half = 11,
    is = 12,
    qlPtr = 13,
    qhPtr = 14,
    scPtr = 15,
    subBase = 16,
    xBase = 17,
    qPtr = 18,
    qStride = 19,
    outAddr = 20
  const fsum = 21,
    d = 22
  const iacc0 = 23,
    iacc1 = 24,
    iacc2 = 25,
    iacc3 = 26,
    ql0 = 27,
    ql32 = 28,
    qh = 29,
    c15 = 30,
    c48 = 31,
    c32 = 32,
    sv0 = 33,
    sv2 = 34,
    sv4 = 35,
    sv6 = 36

  const dotQ6Slice = (byteOff: number): number[] => [
    ...get(qlPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, byteOff),
    ...set(ql0),
    ...get(qlPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, 32 + byteOff),
    ...set(ql32),
    ...get(qhPtr),
    ...simdMem(SIMD_I16X8_LOAD8X8_U, 0, byteOff),
    ...set(qh),

    // q1 = ((ql0 & 15) | ((qh << 4) & 48) - 32) * sv0 dot x[0 + byteOff]
    ...get(iacc0),
    ...get(ql0),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(qh),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHL),
    ...get(c48),
    ...simd(SIMD_V128_AND),
    ...simd(SIMD_I16X8_ADD),
    ...get(c32),
    ...simd(SIMD_I16X8_SUB),
    ...get(sv0),
    ...simd(SIMD_I16X8_MUL),
    ...get(xBase),
    ...simdMem(SIMD_V128_LOAD, 4, byteOff * 2),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),
    ...set(iacc0),

    // q2 = ((ql32 & 15) | ((qh << 2) & 48) - 32) * sv2 dot x[32 + byteOff]
    ...get(iacc1),
    ...get(ql32),
    ...get(c15),
    ...simd(SIMD_V128_AND),
    ...get(qh),
    ...i32Const(2),
    ...simd(SIMD_I16X8_SHL),
    ...get(c48),
    ...simd(SIMD_V128_AND),
    ...simd(SIMD_I16X8_ADD),
    ...get(c32),
    ...simd(SIMD_I16X8_SUB),
    ...get(sv2),
    ...simd(SIMD_I16X8_MUL),
    ...get(xBase),
    ...simdMem(SIMD_V128_LOAD, 4, 64 + byteOff * 2),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),
    ...set(iacc1),

    // q3 = ((ql0 >> 4) | (qh & 48) - 32) * sv4 dot x[64 + byteOff]
    ...get(iacc2),
    ...get(ql0),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(qh),
    ...get(c48),
    ...simd(SIMD_V128_AND),
    ...simd(SIMD_I16X8_ADD),
    ...get(c32),
    ...simd(SIMD_I16X8_SUB),
    ...get(sv4),
    ...simd(SIMD_I16X8_MUL),
    ...get(xBase),
    ...simdMem(SIMD_V128_LOAD, 4, 128 + byteOff * 2),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),
    ...set(iacc2),

    // q4 = ((ql32 >> 4) | ((qh >> 2) & 48) - 32) * sv6 dot x[96 + byteOff]
    ...get(iacc3),
    ...get(ql32),
    ...i32Const(4),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(qh),
    ...i32Const(2),
    ...simd(SIMD_I16X8_SHR_U),
    ...get(c48),
    ...simd(SIMD_V128_AND),
    ...simd(SIMD_I16X8_ADD),
    ...get(c32),
    ...simd(SIMD_I16X8_SUB),
    ...get(sv6),
    ...simd(SIMD_I16X8_MUL),
    ...get(xBase),
    ...simdMem(SIMD_V128_LOAD, 4, 192 + byteOff * 2),
    ...simd(SIMD_I32X4_DOT_I16X8_S),
    ...simd(SIMD_I32X4_ADD),
    ...set(iacc3),
  ]

  const body: number[] = [
    ...i16x8ConstBytes(15),
    ...set(c15),
    ...i16x8ConstBytes(48),
    ...set(c48),
    ...i16x8ConstBytes(32),
    ...set(c32),

    // qStride = nb * 210
    ...get(3),
    ...i32Const(210),
    OP_I32_MUL,
    ...set(qStride),

    ...i32Const(0),
    ...set(m),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(m),
    ...get(0),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...set(n),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(n),
    ...get(1),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...f32Const(0),
    ...set(fsum),
    ...get(4),
    ...get(n),
    ...get(qStride),
    OP_I32_MUL,
    OP_I32_ADD,
    ...set(qPtr),

    ...i32Const(0),
    ...set(b),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(b),
    ...get(3),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...loadFp16AsF32(get(qPtr), 208),
    ...set(d),

    ...i32Const(0),
    ...set(half),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(half),
    ...i32Const(2),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    ...i32Const(0),
    ...simd(SIMD_I32X4_SPLAT),
    ...tee(iacc0),
    ...tee(iacc1),
    ...tee(iacc2),
    ...set(iacc3),

    // subBase = (m * nb + b) * 8 + half * 4
    ...get(m),
    ...get(3),
    OP_I32_MUL,
    ...get(b),
    OP_I32_ADD,
    ...i32Const(3),
    OP_I32_SHL,
    ...get(half),
    ...i32Const(2),
    OP_I32_SHL,
    OP_I32_ADD,
    ...set(subBase),

    // qlPtr = qPtr + half * 64, qhPtr = qPtr + 128 + half * 32, scPtr = qPtr + 192 + half * 8
    ...get(qPtr),
    ...get(half),
    ...i32Const(6),
    OP_I32_SHL,
    OP_I32_ADD,
    ...set(qlPtr),
    ...get(qPtr),
    ...i32Const(128),
    OP_I32_ADD,
    ...get(half),
    ...i32Const(5),
    OP_I32_SHL,
    OP_I32_ADD,
    ...set(qhPtr),
    ...get(qPtr),
    ...i32Const(192),
    OP_I32_ADD,
    ...get(half),
    ...i32Const(3),
    OP_I32_SHL,
    OP_I32_ADD,
    ...set(scPtr),

    ...i32Const(0),
    ...set(is),
    OP_BLOCK,
    0x40,
    OP_LOOP,
    0x40,
    ...get(is),
    ...i32Const(2),
    OP_I32_GE_U,
    OP_BR_IF,
    1,

    // Load 4 int8 scales for this 16-element half
    ...get(scPtr),
    ...get(is),
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_S, 0, 0),
    ...simd(SIMD_I16X8_SPLAT),
    ...set(sv0),
    ...get(scPtr),
    ...get(is),
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_S, 0, 2),
    ...simd(SIMD_I16X8_SPLAT),
    ...set(sv2),
    ...get(scPtr),
    ...get(is),
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_S, 0, 4),
    ...simd(SIMD_I16X8_SPLAT),
    ...set(sv4),
    ...get(scPtr),
    ...get(is),
    OP_I32_ADD,
    ...memOp(OP_I32_LOAD8_S, 0, 6),
    ...simd(SIMD_I16X8_SPLAT),
    ...set(sv6),

    // xBase = WASM_MEM_X_Q16 + subBase * 64 + is * 32
    ...get(subBase),
    ...i32Const(6),
    OP_I32_SHL,
    ...get(is),
    ...i32Const(5),
    OP_I32_SHL,
    OP_I32_ADD,
    ...i32Const(WASM_MEM_X_Q16),
    OP_I32_ADD,
    ...set(xBase),

    ...dotQ6Slice(0),
    ...dotQ6Slice(8),

    ...get(qlPtr),
    ...i32Const(16),
    OP_I32_ADD,
    ...set(qlPtr),
    ...get(qhPtr),
    ...i32Const(16),
    OP_I32_ADD,
    ...set(qhPtr),
    ...get(is),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(is),
    OP_BR,
    0,
    OP_END,
    OP_END,

    // subBase <<= 2 (byte offset into X_SCALE)
    ...get(subBase),
    ...i32Const(2),
    OP_I32_SHL,
    ...set(subBase),

    // fsum += d * (hsum(iacc0)*xScale[0] + hsum(iacc1)*xScale[1] + hsum(iacc2)*xScale[2] + hsum(iacc3)*xScale[3])
    ...get(fsum),
    ...get(d),
    ...hsumI32x4(iacc0),
    OP_F32_CONVERT_I32_S,
    ...get(subBase),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE),
    OP_F32_MUL,
    ...hsumI32x4(iacc1),
    OP_F32_CONVERT_I32_S,
    ...get(subBase),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE + 4),
    OP_F32_MUL,
    OP_F32_ADD,
    ...hsumI32x4(iacc2),
    OP_F32_CONVERT_I32_S,
    ...get(subBase),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE + 8),
    OP_F32_MUL,
    ...hsumI32x4(iacc3),
    OP_F32_CONVERT_I32_S,
    ...get(subBase),
    ...memOp(OP_F32_LOAD, 2, WASM_MEM_X_SCALE + 12),
    OP_F32_MUL,
    OP_F32_ADD,
    OP_F32_ADD,
    OP_F32_MUL,
    OP_F32_ADD,
    ...set(fsum),

    ...get(half),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(half),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(qPtr),
    ...i32Const(210),
    OP_I32_ADD,
    ...set(qPtr),
    ...get(b),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(b),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...storeOutputWithResid(m, n, 5, 6, 7, fsum, outAddr),
    ...get(n),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(n),
    OP_BR,
    0,
    OP_END,
    OP_END,

    ...get(m),
    ...i32Const(1),
    OP_I32_ADD,
    ...set(m),
    OP_BR,
    0,
    OP_END,
    OP_END,
    OP_END,
  ]

  return encodeFunc([
    [13, I32],
    [2, F32],
    [14, V128],
  ], body)
}

function encodeFunc(locals: Array<[number, number]>, body: number[]): number[] {
  const localBytes: number[] = [...uleb(locals.length)]
  for (const [count, type] of locals) {
    localBytes.push(...uleb(count), type)
  }
  const full = [...localBytes, ...body]
  return [...uleb(full.length), ...full]
}

function encodeSection(id: number, payload: number[]): number[] {
  return [id, ...uleb(payload.length), ...payload]
}

function encodeString(str: string): number[] {
  const bytes = Array.from(new TextEncoder().encode(str))
  return [...uleb(bytes.length), ...bytes]
}

let cachedWasmBytes: Uint8Array | null = null

/**
 * Build the complete WebAssembly SIMD binary module as a `Uint8Array`.
 */
export function getWasmKernelModuleBytes(): Uint8Array {
  if (cachedWasmBytes) return cachedWasmBytes

  // Type 0: (M, N, K, wPtr, hasResid, rowOffset, totalN) -> () [7 x i32]
  // Type 1: (M, N, K, nb, quantsPtr, scalesPtr, hasResid, rowOffset, totalN) -> () [9 x i32]
  // Type 2: (M, N, K, nb, quantsPtr, hasResid, rowOffset, totalN) -> () [8 x i32]
  const typeSection = encodeSection(1, [
    3, // 3 function types
    0x60,
    7,
    I32,
    I32,
    I32,
    I32,
    I32,
    I32,
    I32,
    0,
    0x60,
    9,
    I32,
    I32,
    I32,
    I32,
    I32,
    I32,
    I32,
    I32,
    I32,
    0,
    0x60,
    8,
    I32,
    I32,
    I32,
    I32,
    I32,
    I32,
    I32,
    I32,
    0,
  ])

  // Import section: env.memory (min 1 page)
  const importSection = encodeSection(2, [
    1,
    ...encodeString('env'),
    ...encodeString('memory'),
    0x02, // memory import
    0x00, // flags: no max
    1, // initial: 1 page
  ])

  // Function section: 8 functions
  // 0: gemv_f32 (type 0)
  // 1: gemv_q1_0 (type 1)
  // 2: gemv_pq2_0 (type 1)
  // 3: gemv_ptq1_0 (type 1)
  // 4: gemv_q4_0 (type 2)
  // 5: gemv_q4_1 (type 2)
  // 6: gemv_q4_k (type 2)
  // 7: gemv_q6_k (type 2)
  const funcSection = encodeSection(3, [8, 0, 1, 1, 1, 2, 2, 2, 2])

  const exportNames = [
    'gemv_f32',
    'gemv_q1_0',
    'gemv_pq2_0',
    'gemv_ptq1_0',
    'gemv_q4_0',
    'gemv_q4_1',
    'gemv_q4_k',
    'gemv_q6_k',
  ]
  const exportPayload: number[] = [exportNames.length]
  for (let i = 0; i < exportNames.length; i++) {
    exportPayload.push(...encodeString(exportNames[i]), 0x00, i)
  }
  const exportSection = encodeSection(7, exportPayload)

  const codeBodies = [
    buildGemvF32(),
    buildGemvQ1_0(),
    buildGemvPQ2_0(),
    buildGemvPTQ1_0(),
    buildGemvQ4_0(),
    buildGemvQ4_1(),
    buildGemvQ4_K(),
    buildGemvQ6_K(),
  ]
  const codePayload: number[] = [codeBodies.length]
  for (const b of codeBodies) codePayload.push(...b)
  const codeSection = encodeSection(10, codePayload)

  cachedWasmBytes = new Uint8Array([
    0x00,
    0x61,
    0x73,
    0x6d, // "\0asm"
    0x01,
    0x00,
    0x00,
    0x00, // version 1
    ...typeSection,
    ...importSection,
    ...funcSection,
    ...exportSection,
    ...codeSection,
  ])
  return cachedWasmBytes
}
