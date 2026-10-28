import { defaultFetchRange } from './gguf.js'
import type { StorageOptions } from './types.js'

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
 * Check if a model URL is completely cached in OPFS or IndexedDB.
 */
export async function isModelCached(url: string, expectedTotalBytes?: number): Promise<boolean> {
  const key = hashUrl(url)
  const opfs = await getOpfsRoot()
  if (opfs) {
    try {
      const metaHandle = await opfs.getFileHandle(`${key}.meta.json`)
      const metaFile = await metaHandle.getFile()
      const meta = JSON.parse(await metaFile.text()) as CacheMeta
      if (expectedTotalBytes && meta.totalSize !== expectedTotalBytes) return false
      const totalChunks = Math.ceil(meta.totalSize / meta.chunkBytes)
      return meta.completedChunks.length === totalChunks
    } catch {
      // Fall through to IDB check
    }
  }
  try {
    const meta = await idbGet<CacheMeta>(`meta:${key}`)
    if (!meta) return false
    if (expectedTotalBytes && meta.totalSize !== expectedTotalBytes) return false
    const totalChunks = Math.ceil(meta.totalSize / meta.chunkBytes)
    return meta.completedChunks.length === totalChunks
  } catch {
    return false
  }
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
  onChunk: (chunk: Uint8Array, fileOffset: number) => Promise<void>,
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
    if (opfsDir) {
      try {
        const metaHandle = await opfsDir.getFileHandle(`${key}.meta.json`, { create: true })
        const existingText = await (await metaHandle.getFile()).text()
        if (existingText) {
          const parsed = JSON.parse(existingText) as CacheMeta
          if (parsed.totalSize === totalSize && parsed.chunkBytes === chunkBytes) {
            meta = parsed
          }
        }
        const dataHandle = await opfsDir.getFileHandle(`${key}.gguf.bin`, { create: true })
        if (typeof dataHandle.createWritable === 'function') {
          opfsDataFile = await dataHandle.getFile()
        } else {
          // Safari main-thread FileSystemFileHandle without createWritable -> use IndexedDB chunk fallback
          opfsDir = null
        }
      } catch {
        opfsDir = null
      }
    }
    if (!opfsDir) {
      try {
        if (typeof indexedDB !== 'undefined') {
          idbConn = await openIdb()
        }
        const existing = await idbGet<CacheMeta>(`meta:${key}`, idbConn)
        if (existing && existing.totalSize === totalSize && existing.chunkBytes === chunkBytes) {
          meta = existing
        }
      } catch {
        // Ignore IDB errors
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
        try {
          const sliceBlob = opfsDataFile.slice(chunkStart, chunkEndExclusive)
          data = new Uint8Array(await sliceBlob.arrayBuffer())
          fromCache = true
        } catch {
          data = null
        }
      } else if (!opfsDir) {
        try {
          const cachedChunk = await idbGet<Uint8Array>(`chunk:${key}:${ci}`, idbConn)
          if (cachedChunk && cachedChunk.byteLength === expectedLen) {
            data = cachedChunk
            fromCache = true
          }
        } catch {
          data = null
        }
      }
    }

    if (!data) {
      const baseLoaded = Math.max(0, chunkStart - startOffset)
      data = await fetchRange(url, chunkStart, chunkEndExclusive - 1, options.signal, (bytesInChunk) => {
        const curLoaded = Math.max(0, chunkStart + bytesInChunk - startOffset)
        if (curLoaded > baseLoaded) {
          reportProgress(curLoaded, false)
        }
      })
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
      const sub = sliceOffset > 0 ? cur.data.subarray(sliceOffset) : cur.data

      // Overlap local cache write and WebGPU staging upload for chunk (ci)
      await Promise.all([
        cur.fromCache ? Promise.resolve() : persistChunk(ci, cur.chunkStart, cur.data),
        onChunk(sub, cur.chunkStart + sliceOffset),
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
