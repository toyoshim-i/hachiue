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

export class GgufTokenizer {
  private readonly tokens: string[]
  private readonly tokenToIdMap: Map<string, number>
  private readonly mergeRanks: Map<string, number>
  private readonly specialTokens: string[]
  private readonly specialTokenIds: Set<number>
  private readonly specialRegex: RegExp | null
  readonly eosTokenId: number
  readonly bosTokenId?: number
  readonly padTokenId?: number

  constructor(data: TokenizerData) {
    this.tokens = data.tokens
    this.eosTokenId = data.eosTokenId
    this.bosTokenId = data.bosTokenId
    this.padTokenId = data.padTokenId

    this.tokenToIdMap = new Map<string, number>()
    this.specialTokens = []
    this.specialTokenIds = new Set<number>()

    for (let id = 0; id < data.tokens.length; id++) {
      const t = data.tokens[id]
      this.tokenToIdMap.set(t, id)
      const ttype = data.tokenTypes?.[id]
      // GGUF token_type: 1 = normal, 3 = control, 4 = user_defined (e.g. <think>, <|im_start|>)
      if ((ttype !== undefined && ttype !== 1) || (t.startsWith('<') && t.endsWith('>') && t.length > 2)) {
        if (!t.startsWith('[PAD')) {
          this.specialTokens.push(t)
          this.specialTokenIds.add(id)
        }
      }
    }

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

  /**
   * Encode text into token IDs, recognizing ChatML / Qwen special tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, etc.).
   */
  encode(text: string): number[] {
    if (!text) return []
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
    for (const id of ids) {
      if (skipSpecialTokens && this.specialTokenIds.has(id)) continue
      const tok = this.tokens[id]
      if (tok === undefined) continue
      if (this.specialTokenIds.has(id)) {
        const encoded = new TextEncoder().encode(tok)
        for (const b of encoded) rawBytes.push(b)
      } else {
        for (const ch of tok) {
          const b = charToByte.get(ch)
          if (b !== undefined) {
            rawBytes.push(b)
          } else {
            const encoded = new TextEncoder().encode(ch)
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
   * Format a ChatML / Qwen3 / Bonsai conversation into a prompt string.
   */
  formatChat(messages: ChatMessage[], options: { addGenerationPrompt?: boolean; enableThinking?: boolean } = {}): string {
    const addGen = options.addGenerationPrompt ?? true
    const think = options.enableThinking ?? false
    let out = ''
    for (const msg of messages) {
      out += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`
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
