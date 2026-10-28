export interface SampleConfig {
  temperature?: number
  topK?: number
  topP?: number
  minP?: number
  repetitionPenalty?: number
}

/**
 * Sample a token ID from logits on CPU with Repetition Penalty, Top-K, Top-P, Min-P, and Temperature.
 */
export function sampleLogits(
  logits: Float32Array,
  vocab: number,
  history: number[],
  config: SampleConfig = {},
): number {
  const temp = config.temperature ?? 0.6
  const topK = config.topK ?? 20
  const topP = config.topP ?? 0.95
  const minP = config.minP ?? 0.0
  const repPen = config.repetitionPenalty ?? 1.0

  // Apply repetition penalty on recent tokens (last 128 tokens)
  if (repPen !== 1.0 && history.length > 0) {
    const seen = new Set(history.slice(-128))
    for (const id of seen) {
      if (id >= 0 && id < vocab) {
        const v = logits[id]
        logits[id] = v < 0 ? v * repPen : v / repPen
      }
    }
  }

  // Greedy argmax when temperature <= 0
  if (temp <= 0) {
    let bestId = 0
    let bestVal = logits[0]
    for (let i = 1; i < vocab; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i]
        bestId = i
      }
    }
    return bestId
  }

  // Find Top-K candidates using a min-heap or partial selection
  const k = Math.max(1, Math.min(topK > 0 ? topK : 64, vocab))
  const candIds = new Int32Array(k).fill(-1)
  const candVals = new Float32Array(k).fill(-Infinity)
  let minIdx = 0
  let minVal = -Infinity

  for (let i = 0; i < vocab; i++) {
    const v = logits[i]
    if (v > minVal) {
      candIds[minIdx] = i
      candVals[minIdx] = v
      minIdx = 0
      minVal = candVals[0]
      for (let j = 1; j < k; j++) {
        if (candVals[j] < minVal) {
          minVal = candVals[j]
          minIdx = j
        }
      }
    }
  }

  // Sort candidates descending
  const order = Array.from({ length: k }, (_, i) => i)
    .filter((i) => candIds[i] >= 0)
    .sort((a, b) => candVals[b] - candVals[a])

  if (order.length === 0) return 0
  if (order.length === 1) return candIds[order[0]]

  const maxLogit = candVals[order[0]]
  const invTemp = 1.0 / temp
  const probs = new Float32Array(order.length)
  let sumProb = 0
  for (let i = 0; i < order.length; i++) {
    const p = Math.exp((candVals[order[i]] - maxLogit) * invTemp)
    probs[i] = p
    sumProb += p
  }
  for (let i = 0; i < probs.length; i++) {
    probs[i] /= sumProb
  }

  // Apply Min-P and Top-P cutoffs
  const maxProb = probs[0]
  const minThreshold = minP * maxProb
  let cumProb = 0
  let cutoff = probs.length
  for (let i = 0; i < probs.length; i++) {
    if (i > 0 && probs[i] < minThreshold) {
      cutoff = i
      break
    }
    cumProb += probs[i]
    if (cumProb >= topP) {
      cutoff = i + 1
      break
    }
  }

  let keptSum = 0
  for (let i = 0; i < cutoff; i++) keptSum += probs[i]

  let r = Math.random() * keptSum
  for (let i = 0; i < cutoff; i++) {
    r -= probs[i]
    if (r <= 0) return candIds[order[i]]
  }
  return candIds[order[cutoff - 1]]
}
