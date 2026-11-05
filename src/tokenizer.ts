import type { ChatMessage, TokenizerData } from './types.js'

/**
 * Build GPT-2 / Qwen byte-to-unicode mapping tables.
 */
function buildByteEncoder(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const bs: number[] = []
  for (let b = 33; b <= 126; b++) bs.push(b)
  for (let b = 161; b <= 172; b++) bs.push(b)
  for (let b = 174; b <= 255; b++) bs.push(b)
  const cs = [...bs]
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n)
      n++
    }
  }
  const byteToChar = new Array<string>(256)
  const charToByte = new Map<string, number>()
  for (let i = 0; i < 256; i++) {
    const ch = String.fromCodePoint(cs[i])
    byteToChar[bs[i]] = ch
    charToByte.set(ch, bs[i])
  }
  return { byteToChar, charToByte }
}

const { byteToChar, charToByte } = buildByteEncoder()

// Qwen2 / Qwen3 / GPT-4 style pre-tokenization regex (splits contractions, words, single digits/numbers, punctuation, newlines, spaces)
const PRE_TOKENIZE_REGEX =
  /(?:'[sSdDmMtT]|'[rR][eE]|'[vV][eE]|'[lL][lL])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu

// PLaMo-3 Unigram DP constants matching pfnet/plamo-3-translate tokenization_plamo.py
const PLAMO_INVALID_SCORE = -20000000
const PLAMO_UNKNOWN_SCORE = -10000000
const PLAMO_BOUNDARY_CHAR = '\uee00'
const PLAMO_BOUNDARY_CP = 0xee00
const PLAMO_BOUNDARY_TOKEN_ID = 10000000

interface UnigramTrieNode {
  children: Map<number, UnigramTrieNode>
  tokenId: number
  score: number
}

export class GgufTokenizer {
  readonly model: string
  private readonly tokens: string[]
  private readonly tokenToIdMap: Map<string, number>
  private readonly mergeRanks: Map<string, number>
  private readonly specialTokens: string[]
  private readonly specialTokenIds: Set<number>
  private readonly specialRegex: RegExp | null
  // Unigram / PLaMo-3 fields
  private readonly isPlamo3: boolean
  private readonly unigramRoot: UnigramTrieNode | null
  private readonly byteTokenIds: Int32Array | null
  private readonly plamoSpRegex: RegExp | null
  private readonly plamoBoundaryRegex: RegExp | null
  readonly addBosToken: boolean
  readonly eosTokenId: number
  readonly eomTokenId?: number
  readonly eotTokenId?: number
  readonly bosTokenId?: number
  readonly padTokenId?: number

  constructor(data: TokenizerData) {
    this.model = data.model ?? 'gpt2'
    this.isPlamo3 = this.model === 'plamo3'
    this.tokens = data.tokens
    this.addBosToken = data.addBosToken
    this.eosTokenId = data.eosTokenId
    this.eomTokenId = data.eomTokenId
    this.eotTokenId = data.eotTokenId
    this.bosTokenId = data.bosTokenId
    this.padTokenId = data.padTokenId

    this.tokenToIdMap = new Map<string, number>()
    this.specialTokens = []
    this.specialTokenIds = new Set<number>()

    for (let id = 0; id < data.tokens.length; id++) {
      const t = data.tokens[id]
      this.tokenToIdMap.set(t, id)
      const ttype = data.tokenTypes?.[id]
      if (this.isPlamo3) {
        // In plamo3: type 2 = unknown, type 3 = control (<|plamo:*|>), type 6 = byte (<0xXX>)
        if (ttype === 2 || ttype === 3) {
          this.specialTokens.push(t)
          this.specialTokenIds.add(id)
        }
      } else {
        // GGUF token_type: 1 = normal, 3 = control, 4 = user_defined (e.g. <think>, <|im_start|>)
        if ((ttype !== undefined && ttype !== 1) || (t.startsWith('<') && t.endsWith('>') && t.length > 2)) {
          if (!t.startsWith('[PAD')) {
            this.specialTokens.push(t)
            this.specialTokenIds.add(id)
          }
        }
      }
    }

    if (this.isPlamo3) {
      this.specialRegex = null
      this.mergeRanks = new Map()
      const root: UnigramTrieNode = { children: new Map(), tokenId: -1, score: PLAMO_INVALID_SCORE }
      const byteIds = new Int32Array(256)
      const scores = data.scores ?? []
      const ttypes = data.tokenTypes ?? []

      for (let id = 0; id < data.tokens.length; id++) {
        const tok = data.tokens[id]
        const tt = ttypes[id] ?? 1
        if (tt === 6 && tok.length === 6 && tok.startsWith('<0x') && tok.endsWith('>')) {
          const byteVal = Number.parseInt(tok.slice(3, 5), 16)
          if (!Number.isNaN(byteVal) && byteVal >= 0 && byteVal < 256) {
            byteIds[byteVal] = id
          }
          continue
        }
        let node = root
        for (let i = 0; i < tok.length; ) {
          const cp = tok.codePointAt(i)!
          i += cp > 0xffff ? 2 : 1
          let child = node.children.get(cp)
          if (!child) {
            child = { children: new Map(), tokenId: -1, score: PLAMO_INVALID_SCORE }
            node.children.set(cp, child)
          }
          node = child
        }
        node.tokenId = id
        const sc = scores[id] ?? 0
        node.score = Number.isFinite(sc) ? Math.round(sc * 1e4) : PLAMO_INVALID_SCORE
      }

      // Add internal boundary character U+EE00 matching tokenization_plamo.py
      root.children.set(PLAMO_BOUNDARY_CP, {
        children: new Map(),
        tokenId: PLAMO_BOUNDARY_TOKEN_ID,
        score: 0,
      })

      this.unigramRoot = root
      this.byteTokenIds = byteIds
      this.plamoSpRegex = /(<\|plamo:[^|\s]{0,64}\|>)/g

      const spaceThresh = data.breakAroundConsecutiveSpacesThreshold ?? 2
      const repeatThresh = data.breakAroundRepeatedCharsThreshold ?? 4
      const patterns: string[] = []
      if (repeatThresh > 1) {
        patterns.push(`(.)\\2{${repeatThresh - 1},}`)
      }
      if (spaceThresh > 1) {
        patterns.push(` {${spaceThresh},}`)
      }
      this.plamoBoundaryRegex = patterns.length > 0 ? new RegExp(`(${patterns.join('|')})`, 'gu') : null
    } else {
      this.unigramRoot = null
      this.byteTokenIds = null
      this.plamoSpRegex = null
      this.plamoBoundaryRegex = null

      // Sort longer special tokens first so greedy alternation matches longest
      this.specialTokens.sort((a, b) => b.length - a.length)
      if (this.specialTokens.length > 0) {
        const escaped = this.specialTokens.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        this.specialRegex = new RegExp(`(${escaped.join('|')})`, 'g')
      } else {
        this.specialRegex = null
      }

      this.mergeRanks = new Map<string, number>()
      for (let i = 0; i < data.merges.length; i++) {
        this.mergeRanks.set(data.merges[i], i)
      }
    }
  }

  getTokenId(token: string): number | undefined {
    return this.tokenToIdMap.get(token)
  }

  private bpe(piece: string): number[] {
    const direct = this.tokenToIdMap.get(piece)
    if (direct !== undefined && piece.length <= 1) return [direct]

    let word = Array.from(piece)
    while (word.length > 1) {
      let minRank = Infinity
      let bestPairIdx = -1
      for (let i = 0; i < word.length - 1; i++) {
        const key = `${word[i]} ${word[i + 1]}`
        const rank = this.mergeRanks.get(key)
        if (rank !== undefined && rank < minRank) {
          minRank = rank
          bestPairIdx = i
        }
      }
      if (bestPairIdx === -1) break
      const first = word[bestPairIdx]
      const second = word[bestPairIdx + 1]
      const nextWord: string[] = []
      let i = 0
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          nextWord.push(first + second)
          i += 2
        } else {
          nextWord.push(word[i])
          i++
        }
      }
      word = nextWord
    }

    const ids: number[] = []
    for (const sym of word) {
      const id = this.tokenToIdMap.get(sym)
      if (id !== undefined) ids.push(id)
    }
    return ids
  }

  private encodePlainSegment(segment: string): number[] {
    if (!segment) return []
    const utf8 = new TextEncoder()
    const ids: number[] = []
    const matches = segment.match(PRE_TOKENIZE_REGEX) ?? [segment]
    for (const m of matches) {
      const bytes = utf8.encode(m)
      let mapped = ''
      for (let i = 0; i < bytes.length; i++) {
        mapped += byteToChar[bytes[i]]
      }
      ids.push(...this.bpe(mapped))
    }
    return ids
  }

  private encodePlamo3(text: string): number[] {
    if (!text || !this.unigramRoot || !this.byteTokenIds) return []
    let s = text
    if (this.plamoSpRegex) {
      s = s.replace(this.plamoSpRegex, `${PLAMO_BOUNDARY_CHAR}$1${PLAMO_BOUNDARY_CHAR}`)
    }
    if (this.plamoBoundaryRegex) {
      s = s.replace(this.plamoBoundaryRegex, `${PLAMO_BOUNDARY_CHAR}$1${PLAMO_BOUNDARY_CHAR}`)
    }

    const cps: number[] = []
    for (let i = 0; i < s.length; ) {
      const cp = s.codePointAt(i)!
      cps.push(cp)
      i += cp > 0xffff ? 2 : 1
    }

    const n = cps.length
    const dpCost = new Float64Array(n + 1)
    dpCost.fill(1e18)
    dpCost[n] = 0
    const pathLen = new Int32Array(n + 1)
    const pathTid = new Int32Array(n + 1)
    pathTid.fill(-1)

    const matchLens: number[] = []
    const matchTids: number[] = []
    const matchScores: number[] = []

    for (let i = n - 1; i >= 0; i--) {
      matchLens.length = 0
      matchTids.length = 0
      matchScores.length = 0

      let node: UnigramTrieNode | undefined = this.unigramRoot
      let j = i
      while (j < n) {
        node = node.children.get(cps[j])
        if (!node) break
        j++
        if (node.score > PLAMO_INVALID_SCORE) {
          matchLens.push(j - i)
          matchTids.push(node.tokenId)
          matchScores.push(node.score)
        }
      }

      // Iterate from longest prefix to shortest with strict `<` (matches tokenization_plamo.py tie-breaking)
      let bestCost = 1e18
      let bestLen = 1
      let bestTid = -1
      for (let k = matchLens.length - 1; k >= 0; k--) {
        const plen = matchLens[k]
        const cost = dpCost[i + plen] - matchScores[k]
        if (cost < bestCost) {
          bestCost = cost
          bestLen = plen
          bestTid = matchTids[k]
        }
      }

      // Sentinel row (length 1, score UNKNOWN_SCORE, tokenId -1)
      const sentinelCost = dpCost[i + 1] - PLAMO_UNKNOWN_SCORE
      if (sentinelCost < bestCost) {
        bestCost = sentinelCost
        bestLen = 1
        bestTid = -1
      }

      dpCost[i] = bestCost
      pathLen[i] = bestLen
      pathTid[i] = bestTid
    }

    const utf8 = new TextEncoder()
    const out: number[] = []
    let pos = 0
    while (pos < n) {
      const tid = pathTid[pos]
      if (tid >= 0) {
        if (tid !== PLAMO_BOUNDARY_TOKEN_ID) {
          out.push(tid)
        }
      } else {
        const bytes = utf8.encode(String.fromCodePoint(cps[pos]))
        for (let b = 0; b < bytes.length; b++) {
          out.push(this.byteTokenIds[bytes[b]])
        }
      }
      pos += pathLen[pos]
    }
    return out
  }

  /**
   * Encode text into token IDs, recognizing ChatML / Qwen or PLaMo-3 special tokens.
   */
  encode(text: string): number[] {
    if (!text) return []
    if (this.isPlamo3) {
      return this.encodePlamo3(text)
    }
    if (!this.specialRegex) return this.encodePlainSegment(text)

    const ids: number[] = []
    let lastIndex = 0
    this.specialRegex.lastIndex = 0
    let match: RegExpExecArray | null
    while ((match = this.specialRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        ids.push(...this.encodePlainSegment(text.slice(lastIndex, match.index)))
      }
      const spId = this.tokenToIdMap.get(match[0])
      if (spId !== undefined) {
        ids.push(spId)
      } else {
        ids.push(...this.encodePlainSegment(match[0]))
      }
      lastIndex = this.specialRegex.lastIndex
    }
    if (lastIndex < text.length) {
      ids.push(...this.encodePlainSegment(text.slice(lastIndex)))
    }
    return ids
  }

  /**
   * Decode an array of token IDs back into a UTF-8 string.
   */
  decode(ids: number[], skipSpecialTokens = true): string {
    const rawBytes: number[] = []
    const utf8 = new TextEncoder()
    for (const id of ids) {
      if (skipSpecialTokens && this.specialTokenIds.has(id)) continue
      const tok = this.tokens[id]
      if (tok === undefined) continue
      if (this.isPlamo3) {
        if (tok.length === 6 && tok.startsWith('<0x') && tok.endsWith('>')) {
          const byteVal = Number.parseInt(tok.slice(3, 5), 16)
          if (!Number.isNaN(byteVal)) {
            rawBytes.push(byteVal)
            continue
          }
        }
        const encoded = utf8.encode(tok)
        for (let i = 0; i < encoded.length; i++) rawBytes.push(encoded[i])
      } else if (this.specialTokenIds.has(id)) {
        const encoded = utf8.encode(tok)
        for (const b of encoded) rawBytes.push(b)
      } else {
        for (const ch of tok) {
          const b = charToByte.get(ch)
          if (b !== undefined) {
            rawBytes.push(b)
          } else {
            const encoded = utf8.encode(ch)
            for (const eb of encoded) rawBytes.push(eb)
          }
        }
      }
    }
    return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(rawBytes))
  }

  /**
   * Create a streaming decoder that holds back incomplete trailing UTF-8 multibyte sequences.
   */
  createDecoderStream(skipSpecialTokens = true): { push(tokenId: number): string; flush(): string } {
    const ids: number[] = []
    let emittedLen = 0
    return {
      push: (tokenId: number): string => {
        ids.push(tokenId)
        const text = this.decode(ids, skipSpecialTokens)
        let safeEnd = text.length
        while (safeEnd > emittedLen && text.charCodeAt(safeEnd - 1) === 0xfffd) {
          safeEnd--
        }
        const delta = text.slice(emittedLen, safeEnd)
        emittedLen = safeEnd
        return delta
      },
      flush: (): string => {
        if (ids.length === 0) return ''
        const text = this.decode(ids, skipSpecialTokens)
        const delta = text.slice(emittedLen)
        emittedLen = text.length
        return delta
      },
    }
  }

  /**
   * Look up a single token string and return its vocabulary ID if present.
   */
  tokenToId(token: string): number | undefined {
    return this.tokenToIdMap.get(token)
  }

  /**
   * Format a ChatML / Qwen3 / Bonsai or PLaMo-3 conversation into a prompt string.
   */
  formatChat(
    messages: ChatMessage[],
    options: { addGenerationPrompt?: boolean; enableThinking?: boolean; targetLang?: string; sourceLang?: string } = {},
  ): string {
    if (this.isPlamo3) {
      // PLaMo-3 Translate embedded GGUF template (PFML3 format)
      const targetLang = options.targetLang ?? 'ja'
      const bos = this.bosTokenId !== undefined ? (this.tokens[this.bosTokenId] ?? '<|plamo:bos|>') : '<|plamo:bos|>'
      let out = `${bos}<|plamo:tag|>dataset<|plamo:msg|>translation`
      const nonSystem = messages.filter((m) => m.role !== 'system')
      const sysMsg = messages.find((m) => m.role === 'system')
      if (sysMsg?.content?.trim()) {
        out += `<|plamo:tag|>context<|plamo:msg|>${sysMsg.content.trim()}`
      }
      for (let i = 0; i < nonSystem.length - 1; i++) {
        const m = nonSystem[i]
        if (m.role === 'user') {
          out += '<|plamo:tag|>input'
          if (options.sourceLang) out += `<|plamo:key|>lang<|plamo:val|>${options.sourceLang}`
          out += `<|plamo:msg|>${m.content}`
        } else {
          out += `<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>${targetLang}<|plamo:msg|>${m.content}`
        }
      }
      const last = nonSystem[nonSystem.length - 1]
      if (last) {
        out += '<|plamo:tag|>input'
        if (options.sourceLang) out += `<|plamo:key|>lang<|plamo:val|>${options.sourceLang}`
        out += `<|plamo:msg|>${last.content}<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>${targetLang}<|plamo:msg|>`
      }
      return out
    }

    const addGen = options.addGenerationPrompt ?? true
    const think = options.enableThinking ?? false
    let out = ''
    for (const msg of messages) {
      if (
        msg.role === 'assistant' &&
        !think &&
        this.tokenToIdMap.has('<think>') &&
        this.tokenToIdMap.has('</think>') &&
        !msg.content.startsWith('<think>')
      ) {
        out += `<|im_start|>assistant\n<think>\n\n</think>\n\n${msg.content}<|im_end|>\n`
      } else {
        out += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`
      }
    }
    if (addGen) {
      out += `<|im_start|>assistant\n`
      // Qwen3 / Bonsai models disable thinking block by pre-filling empty <think>\n\n</think>\n\n when thinking is off
      if (!think && this.tokenToIdMap.has('<think>') && this.tokenToIdMap.has('</think>')) {
        out += `<think>\n\n</think>\n\n`
      } else if (think && this.tokenToIdMap.has('<think>')) {
        out += `<think>\n`
      }
    }
    return out
  }
}
