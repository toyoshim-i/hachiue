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

async function idbGet<T>(key: string): Promise<T | undefined> {
  if (typeof indexedDB === 'undefined') return undefined
  const db = await openIdb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly')
    const req = tx.objectStore(IDB_STORE).get(key)
    req.onsuccess = () => {
      db.close()
      resolve(req.result as T | undefined)
    }
    req.onerror = () => {
      db.close()
      reject(req.error)
    }
  })
}

async function idbPut(key: string, val: unknown): Promise<void> {
  if (typeof indexedDB === 'undefined') return
  const db = await openIdb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite')
    tx.objectStore(IDB_STORE).put(val, key)
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
  let opfsWritable: FileSystemWritableFileStream | null = null
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
        const existing = await idbGet<CacheMeta>(`meta:${key}`)
        if (existing && existing.totalSize === totalSize && existing.chunkBytes === chunkBytes) {
          meta = existing
        }
      } catch {
        // Ignore IDB errors
      }
    }
  }

  const completedSet = new Set<number>(meta.completedChunks)

  const saveMeta = async (): Promise<void> => {
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
    await idbPut(`meta:${key}`, meta).catch(() => {})
  }

  try {
    for (let ci = firstChunkIndex; ci < totalChunks; ci++) {
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
            const cachedChunk = await idbGet<Uint8Array>(`chunk:${key}:${ci}`)
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
        data = await fetchRange(url, chunkStart, chunkEndExclusive - 1, options.signal)
        if (data.byteLength !== expectedLen) {
          throw new Error(`hachiue: chunk ${ci} returned ${data.byteLength} bytes, expected ${expectedLen}`)
        }
        if (useCache) {
          if (opfsDir) {
            try {
              const dataHandle = await opfsDir.getFileHandle(`${key}.gguf.bin`, { create: true })
              opfsWritable = await dataHandle.createWritable({ keepExistingData: true })
              await opfsWritable.write({ type: 'write', position: chunkStart, data: data as Uint8Array<ArrayBuffer> })
              await opfsWritable.close()
              opfsWritable = null
              completedSet.add(ci)
              await saveMeta()
            } catch {
              opfsDir = null
            }
          }
          if (!opfsDir) {
            try {
              await idbPut(`chunk:${key}:${ci}`, data)
              completedSet.add(ci)
              await saveMeta()
            } catch {
              // Non-fatal if storage quota exceeded
            }
          }
        }
      }

      const sliceOffset = Math.max(0, startOffset - chunkStart)
      const sub = sliceOffset > 0 ? data.subarray(sliceOffset) : data
      await onChunk(sub, chunkStart + sliceOffset)

      options.onProgress?.({
        phase: 'weights',
        loadedBytes: chunkEndExclusive - startOffset,
        totalBytes: totalSize - startOffset,
        cached: fromCache,
      })
    }
  } finally {
    if (opfsWritable) {
      await opfsWritable.close().catch(() => {})
    }
  }
}
