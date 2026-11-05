import type { HachiueEngine } from './engine.js'
import type { ChatMessage, GenerateOptions, GenerateStats } from './types.js'

/**
 * Supported language codes for Tier 2 translation helpers (`translateTerm` and `translateText`).
 * Accepts standard BCP-47 / ISO-639 codes as well as full English/native language names.
 */
export type TranslationLangCode =
  | 'en'
  | 'ja'
  | 'ja-easy'
  | 'zh'
  | 'zh-CN'
  | 'zh-Hans'
  | 'zh-TW'
  | 'zh-Hant'
  | 'ko'
  | 'es'
  | 'fr'
  | 'de'
  | 'it'
  | 'pt'
  | 'pt-BR'
  | 'ru'
  | 'ar'
  | 'vi'
  | 'th'
  | 'id'
  | 'nl'
  | 'pl'
  | 'tr'
  | 'uk'
  | 'hi'
  | (string & {})

export interface GlossaryEntry {
  source: string
  target: string
}

export interface TranslationHistoryEntry {
  source: string
  translation: string
  sourceLang?: TranslationLangCode
  targetLang?: TranslationLangCode
}

export interface TranslateTermOptions {
  /** The single word or short phrase to translate (dictionary/lexicon lookup). */
  term: string
  /** Target language code (e.g. `'ja'`, `'zh-CN'`, `'ko'`, `'es'`, `'fr'`, `'de'`, `'en'`). */
  targetLang: TranslationLangCode
  /** Optional source language code (default: `'en'`). */
  sourceLang?: TranslationLangCode
  /**
   * Optional surrounding sentence or context where `term` appears.
   * - On `plamo3` (PLaMo 3 Translate): passed via native `<|plamo:tag|>context` tag so the model
   *   disambiguates polysemous words without translating the entire context sentence.
   * - On `qwen3` / `qwen35` (Bonsai 1 / Bonsai 2): passed via a context-disambiguated dictionary
   *   prompt with automatic validation and fallback if the model echoes the full context sentence.
   */
  context?: string
  /** Maximum tokens to generate (default: 28). */
  maxNewTokens?: number
  /** Sampling temperature (default: 0 for deterministic dictionary output). */
  temperature?: number
  /** Optional streaming callback receiving cleaned text delta. */
  onToken?: (tokenId: number, textDelta: string) => void
  /** Optional AbortSignal to cancel in-flight translation. */
  signal?: AbortSignal
}

export interface TranslateTextOptions {
  /** The sentence, paragraph, or multi-line document to translate. */
  text: string
  /** Target language code (e.g. `'ja'`, `'en'`, `'zh-CN'`, `'zh-TW'`, `'ko'`, `'es'`, `'fr'`, `'de'`). */
  targetLang: TranslationLangCode
  /** Optional source language code (auto-detected by `plamo3` if omitted; defaults to auto/detected for Bonsai). */
  sourceLang?: TranslationLangCode
  /** Optional background context or domain notes. */
  context?: string
  /** Optional glossary term pairs (`[{ source, target }]`) to enforce terminology. */
  glossary?: GlossaryEntry[]
  /**
   * Optional Japanese politeness / register style (`'polite'` = Desu/Masu, `'plain'` = Da/Dearu).
   * Natively supported by `plamo3` via `<|plamo:key|>form` and instructed in system prompt for `qwen3`/`qwen35`.
   */
  form?: 'polite' | 'plain'
  /** Optional prior translation turns for document-level consistency. */
  history?: TranslationHistoryEntry[]
  /** Maximum tokens to generate (default: 512). */
  maxNewTokens?: number
  /** Sampling temperature (default: 0). */
  temperature?: number
  /** Optional repetition penalty. */
  repetitionPenalty?: number
  /** Optional streaming callback. */
  onToken?: (tokenId: number, textDelta: string) => void
  /** Optional AbortSignal. */
  signal?: AbortSignal
}

export interface TranslationResult {
  /** Cleaned translated text in the target language. */
  translation: string
  /** Raw uncleaned text emitted by the model. */
  rawText: string
  /** Whether the `context` option was used for disambiguation. */
  usedContext: boolean
  /** Underlying generation performance statistics. */
  stats: GenerateStats
}

interface NormalizedLangInfo {
  canonical: string
  plamoCode: string
  englishName: string
}

const LANG_MAP: Record<string, NormalizedLangInfo> = {
  en: { canonical: 'en', plamoCode: 'en', englishName: 'English' },
  english: { canonical: 'en', plamoCode: 'en', englishName: 'English' },
  ja: { canonical: 'ja', plamoCode: 'ja', englishName: 'Japanese' },
  jp: { canonical: 'ja', plamoCode: 'ja', englishName: 'Japanese' },
  japanese: { canonical: 'ja', plamoCode: 'ja', englishName: 'Japanese' },
  'ja-easy': { canonical: 'ja-easy', plamoCode: 'ja-easy', englishName: 'Easy Japanese' },
  zh: { canonical: 'zh-CN', plamoCode: 'zh-Hans', englishName: 'Simplified Chinese' },
  'zh-cn': { canonical: 'zh-CN', plamoCode: 'zh-Hans', englishName: 'Simplified Chinese' },
  'zh-hans': { canonical: 'zh-CN', plamoCode: 'zh-Hans', englishName: 'Simplified Chinese' },
  'zh-tw': { canonical: 'zh-TW', plamoCode: 'zh-Hant', englishName: 'Traditional Chinese' },
  'zh-hant': { canonical: 'zh-TW', plamoCode: 'zh-Hant', englishName: 'Traditional Chinese' },
  ko: { canonical: 'ko', plamoCode: 'ko', englishName: 'Korean' },
  korean: { canonical: 'ko', plamoCode: 'ko', englishName: 'Korean' },
  es: { canonical: 'es', plamoCode: 'es', englishName: 'Spanish' },
  spanish: { canonical: 'es', plamoCode: 'es', englishName: 'Spanish' },
  fr: { canonical: 'fr', plamoCode: 'fr', englishName: 'French' },
  french: { canonical: 'fr', plamoCode: 'fr', englishName: 'French' },
  de: { canonical: 'de', plamoCode: 'de', englishName: 'German' },
  german: { canonical: 'de', plamoCode: 'de', englishName: 'German' },
  it: { canonical: 'it', plamoCode: 'it', englishName: 'Italian' },
  italian: { canonical: 'it', plamoCode: 'it', englishName: 'Italian' },
  pt: { canonical: 'pt', plamoCode: 'pt', englishName: 'Portuguese' },
  'pt-br': { canonical: 'pt-BR', plamoCode: 'pt-BR', englishName: 'Brazilian Portuguese' },
  ru: { canonical: 'ru', plamoCode: 'ru', englishName: 'Russian' },
  ar: { canonical: 'ar', plamoCode: 'ar', englishName: 'Arabic' },
  vi: { canonical: 'vi', plamoCode: 'vi', englishName: 'Vietnamese' },
  th: { canonical: 'th', plamoCode: 'th', englishName: 'Thai' },
  id: { canonical: 'id', plamoCode: 'id', englishName: 'Indonesian' },
  nl: { canonical: 'nl', plamoCode: 'nl', englishName: 'Dutch' },
  pl: { canonical: 'pl', plamoCode: 'pl', englishName: 'Polish' },
  tr: { canonical: 'tr', plamoCode: 'tr', englishName: 'Turkish' },
  uk: { canonical: 'uk', plamoCode: 'uk', englishName: 'Ukrainian' },
  hi: { canonical: 'hi', plamoCode: 'hi', englishName: 'Hindi' },
}

export function normalizeLangCode(code: TranslationLangCode): NormalizedLangInfo {
  const key = code.trim().toLowerCase()
  return (
    LANG_MAP[key] ?? {
      canonical: code.trim(),
      plamoCode: code.trim(),
      englishName: code.trim(),
    }
  )
}

/**
 * Strip thinking blocks (`<think>...</think>`, `<|thought|>...`) and PLaMo control tags from raw model output.
 */
export function stripModelReasoningAndTags(raw: string): string {
  return raw
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<think>[\s\S]*$/gi, '')
    .replace(/<\|thought\|>[\s\S]*?(?:<\|thought\|>|<\/thought>|<thought\|>|$)/gi, '')
    .replace(/<\|plamo:[^>]*\|>/g, '')
    .replace(/<\|im_end\|>/g, '')
    .trim()
}

/**
 * Extract a clean single-line dictionary translation from model output (for `translateTerm`).
 */
export function extractCleanTermTranslation(raw: string): string {
  const withoutTags = stripModelReasoningAndTags(raw)
  const firstNonEmptyLine =
    withoutTags
      .split('\n')
      .map((line) => line.trim())
      .find((line) => line.length > 0) || ''
  const afterArrow =
    firstNonEmptyLine.includes('→') || firstNonEmptyLine.includes('->')
      ? (firstNonEmptyLine.split(/→|->/).pop() || '').trim()
      : firstNonEmptyLine
  return afterArrow
    .replace(/^(?:日本語訳|日本語|意味|訳|Translation|Meaning)\s*[:：]\s*/i, '')
    .replace(/^["'「『`]+|["'」』`]+$/g, '')
    .trim()
}

/**
 * Extract cleaned multi-line text translation from model output (for `translateText`).
 */
export function extractCleanTextTranslation(raw: string): string {
  const withoutTags = stripModelReasoningAndTags(raw)
  return withoutTags.replace(/^(?:日本語訳|翻訳|Translation)\s*[:：]\s*/i, '').trim()
}

/**
 * Verify that `output` contains valid script characters for `targetLang` and is not just an unchanged echo of `sourceText`.
 */
export function isValidTargetLanguageTranslation(
  output: string,
  sourceText: string,
  targetLang: TranslationLangCode,
): boolean {
  if (!output) return false
  const cleanOut = output.trim()
  const cleanSrc = sourceText.trim()
  if (!cleanOut) return false
  if (cleanOut.toLowerCase() === cleanSrc.toLowerCase() && cleanSrc.length > 1) {
    return false
  }
  const norm = normalizeLangCode(targetLang).canonical
  if (norm === 'ja' || norm === 'ja-easy') {
    return /[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]/.test(cleanOut)
  }
  if (norm === 'zh-CN' || norm === 'zh-TW') {
    return /[\u3400-\u4DBF\u4E00-\u9FFF]/.test(cleanOut)
  }
  if (norm === 'ko') {
    return /[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]/.test(cleanOut)
  }
  return cleanOut.length > 0
}

/**
 * Build a PFML3 prompt string specifically for `plamo3` (`pfnet/plamo-3-translate-*`).
 * Supports `<|plamo:tag|>context`, `<|plamo:tag|>glossary`, multi-turn history, and `<|plamo:key|>form`.
 */
export function buildPlamo3TranslationPrompt(options: {
  input: string
  targetLang: TranslationLangCode
  sourceLang?: TranslationLangCode
  context?: string
  glossary?: GlossaryEntry[]
  form?: 'polite' | 'plain'
  history?: TranslationHistoryEntry[]
}): string {
  const tgt = normalizeLangCode(options.targetLang)
  const src = options.sourceLang ? normalizeLangCode(options.sourceLang) : null

  let prompt = '<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation'

  const contextLines: string[] = []
  if (options.context?.trim()) {
    contextLines.push(options.context.trim())
  }
  if (contextLines.length > 0) {
    prompt += `<|plamo:tag|>context<|plamo:msg|>${contextLines.join('\n')}`
  }

  if (options.glossary && options.glossary.length > 0) {
    const validGlossary = options.glossary.filter((g) => g.source.trim() && g.target.trim())
    if (validGlossary.length > 0) {
      const glossaryBody = validGlossary.map((g) => `${g.source.trim()}\t${g.target.trim()}`).join('\n')
      prompt += `<|plamo:tag|>glossary<|plamo:msg|>${glossaryBody}`
    }
  }

  if (options.history && options.history.length > 0) {
    for (const h of options.history) {
      if (!h.source.trim() || !h.translation.trim()) continue
      const hSrc = h.sourceLang ? normalizeLangCode(h.sourceLang).plamoCode : src?.plamoCode
      const hTgt = h.targetLang ? normalizeLangCode(h.targetLang).plamoCode : tgt.plamoCode
      prompt += '<|plamo:tag|>input'
      if (hSrc) prompt += `<|plamo:key|>lang<|plamo:val|>${hSrc}`
      prompt += `<|plamo:msg|>${h.source.trim()}`
      prompt += `<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>${hTgt}<|plamo:msg|>${h.translation.trim()}`
    }
  }

  prompt += '<|plamo:tag|>input'
  if (src) {
    prompt += `<|plamo:key|>lang<|plamo:val|>${src.plamoCode}`
  }
  prompt += `<|plamo:msg|>${options.input}`

  prompt += `<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>${tgt.plamoCode}`
  if (options.form && (tgt.plamoCode === 'ja' || tgt.plamoCode === 'ja-easy')) {
    const formVal = options.form === 'polite' ? 'Polite' : 'Plain'
    prompt += `<|plamo:key|>form<|plamo:val|>${formVal}`
  }
  prompt += '<|plamo:msg|>'
  return prompt
}

/**
 * Build ChatML messages for dictionary / short-phrase translation (`translateTerm`) on general instruct models (`qwen3` / `qwen35`).
 */
export function buildInstructTermMessages(
  cleanTerm: string,
  targetLang: TranslationLangCode,
  context?: string,
): ChatMessage[] {
  const norm = normalizeLangCode(targetLang)
  const cleanCtx = context?.trim()

  if (cleanCtx) {
    const ctxTemplates: Record<string, { system: string; user: string }> = {
      ja: {
        system:
          'あなたは文脈を考慮する英和辞典です。与えられた文脈（Context）における対象語（Target）の簡潔な日本語訳（1〜3語）のみを出力してください。英単語は含めず、文全体の翻訳や説明も絶対に出力しないでください。',
        user: `例:\nContext: "The river bank was overflowing."\nTarget: "bank" → 川岸\n\nContext: "The metal coil spring broke."\nTarget: "spring" → バネ、ぜんまい\n\nContext: "${cleanCtx}"\nTarget: "${cleanTerm}" →`,
      },
      'zh-CN': {
        system:
          '你是一本结合上下文的英汉词典。请只输出目标词（Target）在给定句子（Context）中的简短中文词义（1至3个词），绝不要翻译整句话。',
        user: `示例:\nContext: "The river bank was overflowing."\nTarget: "bank" → 河岸\n\nContext: "${cleanCtx}"\nTarget: "${cleanTerm}" →`,
      },
      'zh-TW': {
        system:
          '你是一本結合上下文的英漢詞典。請只輸出目標詞（Target）在給定句子（Context）中的簡短繁體中文詞義（1至3個詞），絕不要翻譯整句話。',
        user: `示例:\nContext: "The river bank was overflowing."\nTarget: "bank" → 河岸\n\nContext: "${cleanCtx}"\nTarget: "${cleanTerm}" →`,
      },
      ko: {
        system:
          '당신은 문맥을 고려하는 영한사전입니다. 주어진 문맥(Context)에서 대상 단어(Target)의 간결한 한국어 뜻만 출력하세요. 문장 전체를 번역하지 마세요.',
        user: `예시:\nContext: "The river bank was overflowing."\nTarget: "bank" → 강둑\n\nContext: "${cleanCtx}"\nTarget: "${cleanTerm}" →`,
      },
      es: {
        system:
          'Eres un diccionario inglés-español contextual. Escribe SOLO la traducción breve de la palabra objetivo (Target) según el contexto (Context). No traduzcas toda la oración.',
        user: `Ejemplo:\nContext: "The river bank was overflowing."\nTarget: "bank" → orilla\n\nContext: "${cleanCtx}"\nTarget: "${cleanTerm}" →`,
      },
      fr: {
        system:
          'Vous êtes un dictionnaire anglais-français contextuel. Donnez UNIQUEMENT la traduction courte du mot cible (Target) selon le contexte (Context). Ne traduisez pas toute la phrase.',
        user: `Exemple:\nContext: "The river bank was overflowing."\nTarget: "bank" → rive, berge\n\nContext: "${cleanCtx}"\nTarget: "${cleanTerm}" →`,
      },
      de: {
        system:
          'Du bist ein kontextbezogenes Englisch-Deutsch-Wörterbuch. Gib NUR die kurze deutsche Übersetzung des Zielwortes (Target) im Kontext (Context) aus. Übersetze nicht den ganzen Satz.',
        user: `Beispiel:\nContext: "The river bank was overflowing."\nTarget: "bank" → Flussufer\n\nContext: "${cleanCtx}"\nTarget: "${cleanTerm}" →`,
      },
    }
    const tpl = ctxTemplates[norm.canonical] ?? {
      system: `You are a context-aware dictionary. Output ONLY the concise ${norm.englishName} translation (1-3 words) of the Target term in the given Context. Never translate the entire sentence.`,
      user: `Example:\nContext: "The river bank was overflowing."\nTarget: "bank" → river edge\n\nContext: "${cleanCtx}"\nTarget: "${cleanTerm}" →`,
    }
    return [
      { role: 'system', content: tpl.system },
      { role: 'user', content: tpl.user },
    ]
  }

  const localizedTemplates: Record<string, { system: string; user: string }> = {
    ja: {
      system:
        'あなたは英和辞典です。与えられた英語の簡潔な日本語訳のみを出力してください。英単語や類義語は出力しないでください。',
      user: `例:\n"rapid" → 急速な、速い\n"extract vital cues" → 重要な手がかりを抽出する\n"${cleanTerm}" →`,
    },
    'zh-CN': {
      system: '你是一本英汉词典。请只输出所给英文的简短中文翻译，不要输出英文或解释。',
      user: `示例:\n"rapid" → 快速的、迅速的\n"extract vital cues" → 提取重要线索\n"${cleanTerm}" →`,
    },
    'zh-TW': {
      system: '你是一本英漢詞典。請只輸出所給英文的簡短繁體中文翻譯，不要輸出英文或解釋。',
      user: `示例:\n"rapid" → 快速的、迅速的\n"extract vital cues" → 提取重要線索\n"${cleanTerm}" →`,
    },
    ko: {
      system: '당신은 영한사전입니다. 주어진 영어의 간결한 한국어 뜻만 출력하세요.',
      user: `예시:\n"rapid" → 빠른, 신속한\n"extract vital cues" → 중요한 단서를 추출하다\n"${cleanTerm}" →`,
    },
    es: {
      system: 'Eres un diccionario inglés-español. Escribe SOLO la traducción breve al español.',
      user: `Ejemplo:\n"rapid" → rápido, veloz\n"extract vital cues" → extraer pistas vitales\n"${cleanTerm}" →`,
    },
    fr: {
      system: 'Vous êtes un dictionnaire anglais-français. Donnez UNIQUEMENT la traduction courte en français.',
      user: `Exemple:\n"rapid" → rapide, véloce\n"extract vital cues" → extraire des indices vitaux\n"${cleanTerm}" →`,
    },
    de: {
      system: 'Du bist ein Englisch-Deutsch-Wörterbuch. Gib NUR die kurze deutsche Übersetzung aus.',
      user: `Beispiel:\n"rapid" → schnell, rasch\n"extract vital cues" → wichtige Hinweise extrahieren\n"${cleanTerm}" →`,
    },
  }

  const tpl = localizedTemplates[norm.canonical] ?? {
    system: `You are a concise dictionary. Translate ONLY the given term into ${norm.englishName}. Output ONLY the ${norm.englishName} meaning without explanation.`,
    user: `Example:\n"rapid" → fast\n"${cleanTerm}" →`,
  }

  return [
    { role: 'system', content: tpl.system },
    { role: 'user', content: tpl.user },
  ]
}

/**
 * Build ChatML messages for full sentence / paragraph / document translation (`translateText`) on general instruct models (`qwen3` / `qwen35`).
 */
export function buildInstructTextMessages(options: TranslateTextOptions): ChatMessage[] {
  const tgt = normalizeLangCode(options.targetLang)
  const src = options.sourceLang ? normalizeLangCode(options.sourceLang) : null

  const instructions: string[] = [
    src
      ? `You are a professional translator from ${src.englishName} to ${tgt.englishName}.`
      : `You are a professional translator into ${tgt.englishName}.`,
    `Output ONLY the ${tgt.englishName} translation of the user's text, without any introductory remarks, notes, or explanations.`,
  ]

  if (options.form && (tgt.canonical === 'ja' || tgt.canonical === 'ja-easy')) {
    instructions.push(
      options.form === 'polite'
        ? 'Use polite Japanese style (です・ます調).'
        : 'Use plain Japanese style (だ・である調).',
    )
  }

  if (options.context?.trim()) {
    instructions.push(`Context: ${options.context.trim()}`)
  }

  if (options.glossary && options.glossary.length > 0) {
    const terms = options.glossary
      .filter((g) => g.source.trim() && g.target.trim())
      .map((g) => `"${g.source.trim()}" -> "${g.target.trim()}"`)
      .join(', ')
    if (terms) {
      instructions.push(`Use the following glossary terms: ${terms}.`)
    }
  }

  const messages: ChatMessage[] = [{ role: 'system', content: instructions.join(' ') }]

  if (options.history && options.history.length > 0) {
    for (const h of options.history) {
      if (h.source.trim() && h.translation.trim()) {
        messages.push({ role: 'user', content: h.source.trim() })
        messages.push({ role: 'assistant', content: h.translation.trim() })
      }
    }
  }

  messages.push({ role: 'user', content: options.text.trim() })
  return messages
}

/**
 * Heuristic check to detect if a small instruct model accidentally translated the entire `context`
 * sentence or leaked English words from the context instead of outputting a clean dictionary translation.
 */
function looksLikeInvalidContextTermOutput(
  term: string,
  context: string | undefined,
  translation: string,
  targetLang: TranslationLangCode,
): boolean {
  if (!context) return false
  const termWords = term.trim().split(/\s+/).length
  if (termWords > 3) return false
  // Single-word or 2-word term translation should rarely exceed 24 chars or end with a full-stop period
  if (translation.length > 26 || /[。．.!！?？]$/.test(translation)) {
    return true
  }
  const norm = normalizeLangCode(targetLang).canonical
  if (norm === 'ja' || norm === 'ja-easy' || norm === 'zh-CN' || norm === 'zh-TW' || norm === 'ko') {
    // Reject if the output contains ASCII words (3+ letters) that aren't in the target term itself
    const asciiWords = translation.match(/[A-Za-z]{3,}/g)
    if (asciiWords) {
      const lowerTerm = term.toLowerCase()
      if (asciiWords.some((w) => !lowerTerm.includes(w.toLowerCase()))) {
        return true
      }
    }
  }
  return false
}

/**
 * Tier 2 Helper: Translate a single word or short phrase (`term`), with optional surrounding `context`.
 * Automatically adapts prompt formatting, stop tokens, and output extraction across `plamo3`, `qwen3` (Bonsai 1), and `qwen35` (Bonsai 2).
 */
export async function translateTerm(engine: HachiueEngine, options: TranslateTermOptions): Promise<TranslationResult> {
  const cleanTerm = options.term.trim()
  if (!cleanTerm) {
    throw new Error('hachiue: translateTerm requires a non-empty term')
  }

  const maxNewTokens = options.maxNewTokens ?? 28
  const temperature = options.temperature ?? 0
  const isPlamo3 = engine.arch.archName === 'plamo3'
  const hasContext = Boolean(options.context?.trim())

  const runOnce = async (useContext: boolean): Promise<{ out: string; rawText: string; stats: GenerateStats }> => {
    const lineAbort = new AbortController()
    const onAbort = () => lineAbort.abort()
    options.signal?.addEventListener('abort', onAbort, { once: true })

    let streamedText = ''
    const genOpts: GenerateOptions = {
      maxNewTokens,
      temperature,
      signal: lineAbort.signal,
      onToken: (tokenId, delta) => {
        streamedText += delta
        options.onToken?.(tokenId, delta)
        // Stop immediately once the first non-empty line finishes
        if (streamedText.trim().length > 0 && delta.includes('\n')) {
          lineAbort.abort()
        }
      },
    }

    try {
      let res: { text: string; tokenIds: number[]; stats: GenerateStats }
      if (isPlamo3) {
        const prompt = buildPlamo3TranslationPrompt({
          input: cleanTerm,
          sourceLang: options.sourceLang ?? 'en',
          targetLang: options.targetLang,
          context: useContext ? options.context : undefined,
        })
        res = await engine.generate(prompt, genOpts)
      } else {
        const messages = buildInstructTermMessages(
          cleanTerm,
          options.targetLang,
          useContext ? options.context : undefined,
        )
        res = await engine.generate(messages, genOpts)
      }
      const rawText = res.text || streamedText
      const out = extractCleanTermTranslation(rawText)
      return { out, rawText, stats: res.stats }
    } finally {
      options.signal?.removeEventListener('abort', onAbort)
    }
  }

  const firstAttempt = await runOnce(hasContext)

  // If context was provided on a general instruct model (e.g. Bonsai) and it accidentally translated
  // the full context sentence or failed target script validation, automatically retry without context!
  if (
    hasContext &&
    !isPlamo3 &&
    (!isValidTargetLanguageTranslation(firstAttempt.out, cleanTerm, options.targetLang) ||
      looksLikeInvalidContextTermOutput(cleanTerm, options.context, firstAttempt.out, options.targetLang))
  ) {
    const fallbackAttempt = await runOnce(false)
    if (isValidTargetLanguageTranslation(fallbackAttempt.out, cleanTerm, options.targetLang)) {
      return {
        translation: fallbackAttempt.out,
        rawText: fallbackAttempt.rawText,
        usedContext: false,
        stats: fallbackAttempt.stats,
      }
    }
  }

  return {
    translation: firstAttempt.out,
    rawText: firstAttempt.rawText,
    usedContext: hasContext,
    stats: firstAttempt.stats,
  }
}

/**
 * Tier 2 Helper: Translate a sentence, paragraph, or multi-line document (`text`).
 * Supports `context`, `glossary`, `form` (`'polite' | 'plain'`), `history`, and token streaming across both
 * dedicated translation models (`plamo3`) and general instruct models (`qwen3` / `qwen35`).
 */
export async function translateText(engine: HachiueEngine, options: TranslateTextOptions): Promise<TranslationResult> {
  const cleanText = options.text.trim()
  if (!cleanText) {
    throw new Error('hachiue: translateText requires non-empty text')
  }

  const maxNewTokens = options.maxNewTokens ?? 512
  const temperature = options.temperature ?? 0
  const isPlamo3 = engine.arch.archName === 'plamo3'

  const genOpts: GenerateOptions = {
    maxNewTokens,
    temperature,
    repetitionPenalty: options.repetitionPenalty,
    signal: options.signal,
    onToken: options.onToken,
  }

  let res: { text: string; tokenIds: number[]; stats: GenerateStats }
  if (isPlamo3) {
    const prompt = buildPlamo3TranslationPrompt({
      input: cleanText,
      sourceLang: options.sourceLang,
      targetLang: options.targetLang,
      context: options.context,
      glossary: options.glossary,
      form: options.form,
      history: options.history,
    })
    res = await engine.generate(prompt, genOpts)
  } else {
    const messages = buildInstructTextMessages(options)
    res = await engine.generate(messages, genOpts)
  }

  const translation = extractCleanTextTranslation(res.text)
  return {
    translation,
    rawText: res.text,
    usedContext: Boolean(options.context?.trim()),
    stats: res.stats,
  }
}
