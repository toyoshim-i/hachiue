import { consumeHeaderPrefix, defaultFetchRange, fetchGgufHeader } from './gguf.js'
import type { ParsedGgufHeader, StorageOptions } from './types.js'

const DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024 // 16 MiB
const IDB_NAME = 'hachiue-model-cache-v1'
const IDB_STORE = 'chunks'

interface CacheMeta {
  url: string
  totalSize: number
  chunkBytes: number
  completedChunks: number[]
  updatedAt: number
}

function hashUrl(url: string): string {
  let h1 = 0xdeadbeef ^ url.length
  let h2 = 0x41c6ce57 ^ url.length
  for (let i = 0; i < url.length; i++) {
    const ch = url.charCodeAt(i)
    h1 = Math.imul(h1 ^ ch, 2654435761)
    h2 = Math.imul(h2 ^ ch, 1597334677)
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909)
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909)
  return (h2 >>> 0).toString(16).padStart(8, '0') + (h1 >>> 0).toString(16).padStart(8, '0')
}

function openIdb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE)
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function idbGet<T>(key: string, existingDb?: IDBDatabase | null): Promise<T | undefined> {
  if (typeof indexedDB === 'undefined') return undefined
  const db = existingDb ?? (await openIdb())
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly')
    const req = tx.objectStore(IDB_STORE).get(key)
    req.onsuccess = () => {
      if (!existingDb) db.close()
      resolve(req.result as T | undefined)
    }
    req.onerror = () => {
      if (!existingDb) db.close()
      reject(req.error)
    }
  })
}

async function idbHasKey(key: string, existingDb?: IDBDatabase | null): Promise<boolean> {
  if (typeof indexedDB === 'undefined') return false
  const db = existingDb ?? (await openIdb())
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly')
    const req = tx.objectStore(IDB_STORE).count(key)
    req.onsuccess = () => {
      if (!existingDb) db.close()
      resolve(req.result > 0)
    }
    req.onerror = () => {
      if (!existingDb) db.close()
      reject(req.error)
    }
  })
}

async function idbPut(key: string, val: unknown, existingDb?: IDBDatabase | null): Promise<void> {
  if (typeof indexedDB === 'undefined') return
  const db = existingDb ?? (await openIdb())
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite')
    tx.objectStore(IDB_STORE).put(val, key)
    tx.oncomplete = () => {
      if (!existingDb) db.close()
      resolve()
    }
    tx.onerror = () => {
      if (!existingDb) db.close()
      reject(tx.error)
    }
  })
}

async function idbClearAll(): Promise<void> {
  if (typeof indexedDB === 'undefined') return
  const db = await openIdb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite')
    tx.objectStore(IDB_STORE).clear()
    tx.oncomplete = () => {
      db.close()
      resolve()
    }
    tx.onerror = () => {
      db.close()
      reject(tx.error)
    }
  })
}

async function getOpfsRoot(): Promise<FileSystemDirectoryHandle | null> {
  try {
    if (typeof navigator !== 'undefined' && navigator.storage?.getDirectory) {
      return await navigator.storage.getDirectory()
    }
  } catch {
    // Private browsing or unsupported environment
  }
  return null
}

/**
 * Read a byte range `[start, endInclusive]` from OPFS or IndexedDB cache if all covering chunks are cached.
 * Returns `null` if any covering chunk is missing from cache.
 */
export async function readCachedRange(
  url: string,
  start: number,
  endInclusive: number,
): Promise<Uint8Array | null> {
  const key = hashUrl(url)
  const opfs = await getOpfsRoot()
  if (opfs) {
    try {
      const metaHandle = await opfs.getFileHandle(`${key}.meta.json`)
      const metaText = await (await metaHandle.getFile()).text()
      if (metaText) {
        const meta = JSON.parse(metaText) as CacheMeta
        const clampedEnd = Math.min(endInclusive, meta.totalSize - 1)
        if (clampedEnd >= start) {
          const c0 = Math.floor(start / meta.chunkBytes)
          const c1 = Math.floor(clampedEnd / meta.chunkBytes)
          const set = new Set(meta.completedChunks)
          let allPresent = true
          for (let c = c0; c <= c1; c++) {
            if (!set.has(c)) {
              allPresent = false
              break
            }
          }
          if (allPresent) {
            const dataHandle = await opfs.getFileHandle(`${key}.gguf.bin`)
            const file = await dataHandle.getFile()
            if (file.size > clampedEnd) {
              const slice = file.slice(start, clampedEnd + 1)
              return new Uint8Array(await slice.arrayBuffer())
            }
          }
        }
      }
    } catch {
      // Fall through to IDB check
    }
  }
  try {
    const db = typeof indexedDB !== 'undefined' ? await openIdb() : null
    if (!db) return null
    try {
      const meta = await idbGet<CacheMeta>(`meta:${key}`, db)
      if (!meta || meta.completedChunks.length === 0) return null
      const clampedEnd = Math.min(endInclusive, meta.totalSize - 1)
      if (clampedEnd < start) return null
      const c0 = Math.floor(start / meta.chunkBytes)
      const c1 = Math.floor(clampedEnd / meta.chunkBytes)
      const set = new Set(meta.completedChunks)
      for (let c = c0; c <= c1; c++) {
        if (!set.has(c)) return null
      }
      const out = new Uint8Array(clampedEnd - start + 1)
      let written = 0
      for (let c = c0; c <= c1; c++) {
        const chunk = await idbGet<Uint8Array>(`chunk:${key}:${c}`, db)
        if (!chunk) return null
        const chunkStart = c * meta.chunkBytes
        const s0 = Math.max(0, start - chunkStart)
        const s1 = Math.min(chunk.byteLength, clampedEnd + 1 - chunkStart)
        const sub = chunk.subarray(s0, s1)
        out.set(sub, written)
        written += sub.byteLength
      }
      return out
    } finally {
      db.close()
    }
  } catch {
    return null
  }
}

/**
 * Return the cached byte size of a model URL if completely cached in OPFS or IndexedDB, or 0 otherwise.
 */
export async function getCachedModelSize(url: string, expectedTotalBytes?: number): Promise<number> {
  const key = hashUrl(url)
  const opfs = await getOpfsRoot()
  if (opfs) {
    try {
      const metaHandle = await opfs.getFileHandle(`${key}.meta.json`)
      const metaFile = await metaHandle.getFile()
      const meta = JSON.parse(await metaFile.text()) as CacheMeta
      if (expectedTotalBytes && meta.totalSize !== expectedTotalBytes) return 0
      const totalChunks = Math.ceil(meta.totalSize / meta.chunkBytes)
      return meta.completedChunks.length === totalChunks ? meta.totalSize : 0
    } catch {
      // Fall through to IDB check
    }
  }
  try {
    const meta = await idbGet<CacheMeta>(`meta:${key}`)
    if (!meta) return 0
    if (expectedTotalBytes && meta.totalSize !== expectedTotalBytes) return 0
    const totalChunks = Math.ceil(meta.totalSize / meta.chunkBytes)
    return meta.completedChunks.length === totalChunks ? meta.totalSize : 0
  } catch {
    return 0
  }
}

/**
 * Check if a model URL is completely cached in OPFS or IndexedDB.
 */
export async function isModelCached(url: string, expectedTotalBytes?: number): Promise<boolean> {
  return (await getCachedModelSize(url, expectedTotalBytes)) > 0
}

/**
 * Clear all cached models from OPFS and IndexedDB.
 */
export async function clearModelCache(): Promise<void> {
  const opfs = await getOpfsRoot()
  if (opfs) {
    try {
      // @ts-expect-error FileSystemDirectoryHandle keys() is supported in modern browsers
      for await (const name of opfs.keys()) {
        if (typeof name === 'string' && (name.endsWith('.gguf.bin') || name.endsWith('.meta.json'))) {
          await opfs.removeEntry(name).catch(() => {})
        }
      }
    } catch {
      // Ignore iteration errors
    }
  }
  await idbClearAll().catch(() => {})
}

/**
 * Stream a byte range `[startOffset, totalSize)` in bounded chunks (default 16 MiB),
 * reading from OPFS / IndexedDB cache when available and persisting newly fetched HTTP Range chunks
 * before invoking `onChunk(chunkBytes, chunkFileOffset)`.
 *
 * Because `onChunk` is awaited before moving to the next chunk, the caller can flush GPU staging
 * buffers (`await device.queue.onSubmittedWorkDone()`) and guarantee peak memory stays <= 1 chunk!
 */
export async function streamModelChunks(
  url: string,
  startOffset: number,
  totalSize: number,
  onChunk?: ((chunk: Uint8Array, fileOffset: number) => Promise<void>) | null,
  options: StorageOptions = {},
): Promise<void> {
  const chunkBytes = options.chunkBytes ?? DEFAULT_CHUNK_BYTES
  const useCache = options.useCache ?? true
  const fetchRange = options.fetchRange ?? defaultFetchRange
  const key = hashUrl(url)

  // Request durable storage on mobile browsers best-effort
  if (useCache && typeof navigator !== 'undefined') {
    navigator.storage?.persist?.().catch(() => {})
  }

  const firstChunkIndex = Math.floor(startOffset / chunkBytes)
  const totalChunks = Math.ceil(totalSize / chunkBytes)

  let opfsDir: FileSystemDirectoryHandle | null = null
  let opfsDataFile: File | null = null
  let idbConn: IDBDatabase | null = null
  let meta: CacheMeta = {
    url,
    totalSize,
    chunkBytes,
    completedChunks: [],
    updatedAt: Date.now(),
  }

  if (useCache) {
    opfsDir = await getOpfsRoot()
    let opfsMeta: CacheMeta | null = null
    if (opfsDir) {
      try {
        const metaHandle = await opfsDir.getFileHandle(`${key}.meta.json`)
        const existingText = await (await metaHandle.getFile()).text()
        if (existingText) {
          const parsed = JSON.parse(existingText) as CacheMeta
          if (parsed.totalSize === totalSize && parsed.chunkBytes === chunkBytes) {
            opfsMeta = parsed
          }
        }
      } catch {
        // No existing OPFS meta file yet
      }
    }

    let idbMeta: CacheMeta | null = null
    try {
      if (typeof indexedDB !== 'undefined') {
        idbConn = await openIdb()
        const existing = await idbGet<CacheMeta>(`meta:${key}`, idbConn)
        if (existing && existing.totalSize === totalSize && existing.chunkBytes === chunkBytes) {
          idbMeta = existing
        }
      }
    } catch {
      // Ignore IDB errors
    }

    if (idbMeta && (!opfsMeta || idbMeta.completedChunks.length > opfsMeta.completedChunks.length)) {
      // IndexedDB has the fuller cache (e.g. iOS WebKit fell back to IDB on a previous run)
      meta = idbMeta
      opfsDir = null
    } else if (opfsDir) {
      try {
        const dataHandle = await opfsDir.getFileHandle(`${key}.gguf.bin`, { create: true })
        if (typeof dataHandle.createWritable === 'function') {
          opfsDataFile = await dataHandle.getFile()
          if (opfsMeta) meta = opfsMeta
        } else {
          opfsDir = null
          if (idbMeta) meta = idbMeta
        }
      } catch {
        opfsDir = null
        if (idbMeta) meta = idbMeta
      }
    }
  }

  const completedSet = new Set<number>(meta.completedChunks)

  let dirtyMeta = false
  const saveMeta = async (): Promise<void> => {
    if (!dirtyMeta) return
    dirtyMeta = false
    meta.completedChunks = Array.from(completedSet)
    meta.updatedAt = Date.now()
    if (opfsDir) {
      try {
        const mh = await opfsDir.getFileHandle(`${key}.meta.json`, { create: true })
        const mw = await mh.createWritable()
        await mw.write(JSON.stringify(meta))
        await mw.close()
        return
      } catch {
        // Ignore
      }
    }
    await idbPut(`meta:${key}`, meta, idbConn).catch(() => {})
  }

  interface AcquiredChunk {
    ci: number
    chunkStart: number
    chunkEndExclusive: number
    data: Uint8Array
    fromCache: boolean
  }

  // Track highest byte position reported so out-of-order/pipelined progress never moves backward
  let maxReportedLoaded = 0
  const totalPayloadBytes = totalSize - startOffset
  const reportProgress = (loaded: number, cached: boolean): void => {
    if (loaded > maxReportedLoaded) {
      maxReportedLoaded = loaded
    }
    options.onProgress?.({
      phase: 'weights',
      loadedBytes: Math.min(maxReportedLoaded, totalPayloadBytes),
      totalBytes: totalPayloadBytes,
      cached,
    })
  }

  const acquireChunk = async (ci: number): Promise<AcquiredChunk> => {
    if (options.signal?.aborted) {
      throw new DOMException('Model loading aborted', 'AbortError')
    }
    const chunkStart = ci * chunkBytes
    const chunkEndExclusive = Math.min(totalSize, chunkStart + chunkBytes)
    const expectedLen = chunkEndExclusive - chunkStart

    let data: Uint8Array | null = null
    let fromCache = false

    if (useCache && completedSet.has(ci)) {
      if (opfsDir && opfsDataFile && opfsDataFile.size >= chunkEndExclusive) {
        if (!onChunk) {
          data = new Uint8Array(0)
          fromCache = true
        } else {
          try {
            const sliceBlob = opfsDataFile.slice(chunkStart, chunkEndExclusive)
            data = new Uint8Array(await sliceBlob.arrayBuffer())
            fromCache = true
          } catch {
            data = null
          }
        }
      }
      if (!data) {
        try {
          if (!onChunk) {
            if (await idbHasKey(`chunk:${key}:${ci}`, idbConn)) {
              data = new Uint8Array(0)
              fromCache = true
            }
          } else {
            const cachedChunk = await idbGet<Uint8Array>(`chunk:${key}:${ci}`, idbConn)
            if (cachedChunk && cachedChunk.byteLength === expectedLen) {
              data = cachedChunk
              fromCache = true
            }
          }
        } catch {
          data = null
        }
      }
    }

    const headerPrefix = ci === 0 ? consumeHeaderPrefix(url) : null

    if (!data) {
      const baseLoaded = Math.max(0, chunkStart - startOffset)
      if (headerPrefix && headerPrefix.byteLength > 0 && headerPrefix.byteLength <= expectedLen) {
        if (headerPrefix.byteLength === expectedLen) {
          data = headerPrefix
        } else {
          const tailStart = headerPrefix.byteLength
          const tail = await fetchRange(url, tailStart, chunkEndExclusive - 1, options.signal, (bytesInTail) => {
            const curLoaded = Math.max(0, tailStart + bytesInTail - startOffset)
            if (curLoaded > baseLoaded) {
              reportProgress(curLoaded, false)
            }
          })
          const combined = new Uint8Array(expectedLen)
          combined.set(headerPrefix, 0)
          combined.set(tail, tailStart)
          data = combined
        }
      } else {
        data = await fetchRange(url, chunkStart, chunkEndExclusive - 1, options.signal, (bytesInChunk) => {
          const curLoaded = Math.max(0, chunkStart + bytesInChunk - startOffset)
          if (curLoaded > baseLoaded) {
            reportProgress(curLoaded, false)
          }
        })
      }
      if (data.byteLength !== expectedLen) {
        throw new Error(`hachiue: chunk ${ci} returned ${data.byteLength} bytes, expected ${expectedLen}`)
      }
    }

    return { ci, chunkStart, chunkEndExclusive, data, fromCache }
  }

  const persistChunk = async (ci: number, chunkStart: number, data: Uint8Array): Promise<void> => {
    if (!useCache) return
    if (opfsDir) {
      let writable: FileSystemWritableFileStream | null = null
      try {
        const dataHandle = await opfsDir.getFileHandle(`${key}.gguf.bin`, { create: true })
        writable = await dataHandle.createWritable({ keepExistingData: true })
        await writable.write({ type: 'write', position: chunkStart, data: data as Uint8Array<ArrayBuffer> })
        await writable.close()
        writable = null
        completedSet.add(ci)
        dirtyMeta = true
        if ((ci & 3) === 0 || ci === totalChunks - 1) await saveMeta()
        return
      } catch {
        if (writable) await writable.close().catch(() => {})
        await opfsDir.removeEntry(`${key}.gguf.bin`).catch(() => {})
        await opfsDir.removeEntry(`${key}.meta.json`).catch(() => {})
        opfsDir = null
      }
    }
    if (!opfsDir) {
      try {
        if (!idbConn && typeof indexedDB !== 'undefined') {
          idbConn = await openIdb()
        }
        await idbPut(`chunk:${key}:${ci}`, data, idbConn)
        completedSet.add(ci)
        dirtyMeta = true
        if ((ci & 3) === 0 || ci === totalChunks - 1) await saveMeta()
      } catch {
        // Non-fatal if storage quota exceeded
      }
    }
  }

  try {
    // Double-buffered pipeline: prefetch chunk (ci + 1) while chunk (ci) persists to OPFS/IDB and uploads to WebGPU.
    // Peak memory is strictly bounded to at most 2 chunks (32 MiB).
    let nextPromise: Promise<AcquiredChunk> | null =
      firstChunkIndex < totalChunks ? acquireChunk(firstChunkIndex) : null

    for (let ci = firstChunkIndex; ci < totalChunks; ci++) {
      const cur = (await nextPromise)!
      // Immediately start acquiring chunk (ci + 1) in the second buffer slot
      nextPromise = ci + 1 < totalChunks ? acquireChunk(ci + 1) : null
      // Prevent unhandled rejection if cur processing throws before nextPromise is awaited
      nextPromise?.catch(() => {})

      const sliceOffset = Math.max(0, startOffset - cur.chunkStart)
      const sub = onChunk && sliceOffset > 0 ? cur.data.subarray(sliceOffset) : cur.data

      // Overlap local cache write and WebGPU staging upload for chunk (ci)
      await Promise.all([
        cur.fromCache ? Promise.resolve() : persistChunk(ci, cur.chunkStart, cur.data),
        onChunk
          ? onChunk(sub, cur.chunkStart + sliceOffset)
          : cur.fromCache
            ? Promise.resolve()
            : new Promise<void>((r) => setTimeout(r, 0)),
      ])

      // Release current chunk buffer before moving to (ci + 1)
      ;(cur as { data: Uint8Array | null }).data = null

      reportProgress(cur.chunkEndExclusive - startOffset, cur.fromCache)
    }
  } finally {
    await saveMeta().catch(() => {})
    if (idbConn) {
      idbConn.close()
    }
  }
}

/**
 * Prefetch a GGUF model into the local OPFS / IndexedDB chunk cache in the background
 * without initializing WebGPU or allocating GPU buffers.
 * Safe to run concurrently while another `HachiueEngine` instance is actively generating.
 */
export async function prefetchModel(url: string, options: StorageOptions = {}): Promise<ParsedGgufHeader> {
  const header = await fetchGgufHeader(url, options)
  await streamModelChunks(url, 0, header.totalFileSize, null, { ...options, useCache: true })
  return header
}
