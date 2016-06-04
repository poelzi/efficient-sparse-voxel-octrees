/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ClusteredFile.hpp"

#define FW_USE_ZLIB 0
#if FW_USE_ZLIB
#   include "3rdparty/zlib/zlib.h"
#endif

using namespace FW;

//------------------------------------------------------------------------

ClusteredFile::ClusteredFile(const String& fileName, File::Mode mode, int clusterSize, bool disableCache)
:   m_file                  (fileName, mode, disableCache),
    m_clusterSize           (clusterSize),
    m_defaultCompression    ((Compression)DefaultCompression),
    m_dirty                 (false),

    m_firstCached           (NULL),
    m_lastCached            (NULL),
    m_cacheUsed             (0),
    m_cacheSize             (DefaultCacheSize),
    m_asyncBytesPending     (0)
{
    FW_ASSERT(clusterSize > 0);

    switch (mode)
    {
    case File::Read:
        if (!readMasterChunk())
            clearInternal();
        break;

    case File::Create:
        clear();
        break;

    case File::Modify:
        if (!readMasterChunk())
            clear();
        break;

    default:
        FW_ASSERT(false);
        break;
    }
}

//------------------------------------------------------------------------

ClusteredFile::~ClusteredFile(void)
{
    flush();
    clearInternal();
}

//------------------------------------------------------------------------

F32 ClusteredFile::getFragmentsPerChunk(void) const
{
    int chunks = 0;
    int fragments = 0;
    for (int i = 0; i < getNumGroups(); i++)
    {
        for (int j = 0; j < getNumIDs(i); j++)
        {
            if (!exists(i, j))
                continue;

            chunks++;
            int cluster = get(i, j)->firstCluster;
            int prev = -1;
            while (cluster != FW_S32_MAX)
            {
                if (cluster != prev + 1)
                    fragments++;
                prev = cluster;
                cluster = m_clusters[prev].next;
            }
        }
    }
    return (F32)fragments / (F32)max(chunks, 1);
}

//------------------------------------------------------------------------

void ClusteredFile::setCompression(Compression compression)
{
    FW_ASSERT(compression >= 0 && compression < Compression_Max);

    if (m_defaultCompression != compression)
    {
        m_defaultCompression = compression;
        if (m_file.getMode() != File::Read)
            m_dirty = true;
    }
}

//------------------------------------------------------------------------

void ClusteredFile::clear(void)
{
    if (!checkWritable())
        return;

    clearInternal();
    m_clusters.add().next = FW_S32_MAX;
    m_file.seek(0);
    Array<U8> cluster(NULL, m_clusterSize);
    m_file.write(cluster.getPtr(), m_clusterSize);

    Chunk* chunk            = createChunk(GroupID_Private, PrivateChunkID_Master);
    chunk->firstCluster     = 0;
    chunk->compression      = Compression_None;
    chunk->compressedSize   = 1;
    chunk->uncompressedSize = 1;

    m_dirty = true;
}

//------------------------------------------------------------------------

void ClusteredFile::append(ClusteredFile& other)
{
    if (!checkWritable() || &other == this)
        return;

    for (int i = GroupID_Default; i < other.getNumGroups(); i++)
    {
        for (int j = 0; j < other.getNumIDs(i); j++)
        {
            if (!other.exists(i, j))
                continue;

            if (exists(i, j))
                removeChunk(get(i, j), true);

            cacheCopy(createChunk(i, j), other, other.get(i, j));
            cacheEvict();
            other.cacheEvict();
            m_dirty = true;
        }
    }
}

//------------------------------------------------------------------------

void ClusteredFile::flush(bool clearCache)
{
    if (m_dirty)
    {
        FW_ASSERT(m_file.getMode() != File::Read);
        writeMasterChunk();
        m_file.flush();
        m_dirty = false;
    }

    while (m_asyncOps.getSize())
    {
        asyncWait(m_asyncOps.getLast());
        asyncFinish();
    }

    cacheEvict();
    while (clearCache && m_firstCached)
        cacheEvict(m_firstCached);
}

//------------------------------------------------------------------------

int ClusteredFile::getNumIDs(int groupID) const
{
    FW_ASSERT(groupID >= 0);
    return (groupID >= m_groups.getSize()) ? 0 : m_groups[groupID]->chunks.getSize();
}

//------------------------------------------------------------------------

int ClusteredFile::getFreeID(int groupID) const
{
    FW_ASSERT(groupID >= 0);

    if (groupID >= m_groups.getSize())
        return 0;

    const Group* g = m_groups[groupID];
    return (g->firstFree) ? g->firstFree->id : g->chunks.getSize();
}

//------------------------------------------------------------------------

bool ClusteredFile::exists(int groupID, int chunkID) const
{
    FW_ASSERT(groupID >= 0 && chunkID >= 0);

    if (groupID >= m_groups.getSize())
        return false;
    return (chunkID < m_groups[groupID]->chunks.getSize() && get(groupID, chunkID)->firstCluster != -1);
}

//------------------------------------------------------------------------

int ClusteredFile::getSize(int groupID, int chunkID)
{
    FW_ASSERT(groupID >= 0 && chunkID >= 0);

    if (!exists(groupID, chunkID))
    {
        setError("Chunk %d/%d does not exist!", groupID, chunkID);
        return 0;
    }

    if (groupID == GroupID_Private && chunkID == PrivateChunkID_Master)
        flush();
    return get(groupID, chunkID)->uncompressedSize;
}

//------------------------------------------------------------------------

int ClusteredFile::getSizeOnDisk(int groupID, int chunkID)
{
    int size = 0;
    if (getSize(groupID, chunkID))
        size = get(groupID, chunkID)->compressedSize;
    return ((max(size, 1) + m_clusterSize - 1) / m_clusterSize) * m_clusterSize;
}

//------------------------------------------------------------------------

void ClusteredFile::read(int groupID, int chunkID, void* data, int size)
{
    FW_ASSERT(groupID >= 0 && chunkID >= 0);
    FW_ASSERT(size >= 0);
    FW_ASSERT(data || !size);

    size = min(size, getSize(groupID, chunkID));
    if (!size)
        return;

    memcpy(data, cacheRead(get(groupID, chunkID), size), size);
    cacheEvict();
}

//------------------------------------------------------------------------

void ClusteredFile::readPrefetch(int groupID, int chunkID, int size)
{
    FW_ASSERT(groupID >= 0 && chunkID >= 0);
    FW_ASSERT(size >= 0);

    size = min(size, getSize(groupID, chunkID));
    if (!size)
        return;

    cacheReadPrefetch(get(groupID, chunkID), size);
    cacheEvict();
}

//------------------------------------------------------------------------

bool ClusteredFile::readIsReady(int groupID, int chunkID, int size)
{
    FW_ASSERT(groupID >= 0 && chunkID >= 0);
    FW_ASSERT(size >= 0);

    size = min(size, getSize(groupID, chunkID));
    if (!size)
        return true;

    bool ready = cacheReadIsReady(get(groupID, chunkID), size);
    cacheEvict();
    return ready;
}

//------------------------------------------------------------------------

void ClusteredFile::write(int groupID, int chunkID, const void* data, int size)
{
    FW_ASSERT(groupID >= 0 && chunkID >= 0);
    FW_ASSERT(size >= 0);
    FW_ASSERT(data || !size);

    if (!checkWritable())
        return;

    if (groupID == GroupID_Private)
    {
        setError("Tried to overwrite a private chunk!");
        return;
    }

    if (exists(groupID, chunkID))
        removeChunk(get(groupID, chunkID), true);

    Chunk* chunk = createChunk(groupID, chunkID);
    chunk->compression = m_defaultCompression;
    cacheWrite(chunk, data, size);
    cacheEvict();
    m_dirty = true;
}

//------------------------------------------------------------------------

void ClusteredFile::remove(int groupID, int chunkID)
{
    FW_ASSERT(groupID >= 0 && chunkID >= 0);

    if (!checkWritable())
        return;

    if (groupID == GroupID_Private)
    {
        setError("Tried to remove a private chunk!");
        return;
    }

    if (!exists(groupID, chunkID))
        return;

    removeChunk(get(groupID, chunkID), true);
    m_dirty = true;
}

//------------------------------------------------------------------------

void ClusteredFile::copy(int dstGroupID, int dstChunkID, ClusteredFile& srcFile, int srcGroupID, int srcChunkID)
{
    FW_ASSERT(dstGroupID >= 0 && dstChunkID >= 0);
    FW_ASSERT(srcGroupID >= 0 && srcChunkID >= 0);

    if (!checkWritable())
        return;

    if (dstGroupID == GroupID_Private)
    {
        setError("Tried to overwrite a private chunk!");
        return;
    }

    if (!srcFile.exists(srcGroupID, srcChunkID))
    {
        setError("Chunk %d/%d does not exist!", srcGroupID, srcChunkID);
        return;
    }

    if (&srcFile == this && srcGroupID == dstGroupID && srcChunkID == dstChunkID)
        return;

    if (exists(dstGroupID, dstChunkID))
        removeChunk(get(dstGroupID, dstChunkID), true);

    cacheCopy(createChunk(dstGroupID, dstChunkID), srcFile, srcFile.get(srcGroupID, srcChunkID));
    cacheEvict();
    srcFile.cacheEvict();
    m_dirty = true;
}

//------------------------------------------------------------------------

void ClusteredFile::move(int dstGroupID, int dstChunkID, int srcGroupID, int srcChunkID)
{
    FW_ASSERT(dstGroupID >= 0 && dstChunkID >= 0);
    FW_ASSERT(srcGroupID >= 0 && srcChunkID >= 0);

    if (!checkWritable())
        return;

    if (!exists(srcGroupID, srcChunkID))
    {
        setError("Chunk %d/%d does not exist!", srcGroupID, srcChunkID);
        return;
    }

    if (dstGroupID == GroupID_Private)
    {
        setError("Tried to overwrite a private chunk!");
        return;
    }

    if (srcGroupID == GroupID_Private)
    {
        setError("Tried to move a private chunk!");
        return;
    }

    if (srcGroupID == dstGroupID && srcChunkID == dstChunkID)
        return;

    if (exists(dstGroupID, dstChunkID))
        removeChunk(get(dstGroupID, dstChunkID), true);

    Chunk* dst              = createChunk(dstGroupID, dstChunkID);
    Chunk* src              = get(srcGroupID, srcChunkID);
    dst->firstCluster       = src->firstCluster;
    dst->compression        = src->compression;
    dst->compressedSize     = src->compressedSize;
    dst->uncompressedSize   = src->uncompressedSize;

    if (src->cachedData.size)
    {
        dst->cachedData = src->cachedData;
        dst->cachedDataCompressed = src->cachedDataCompressed;
        initBuffer(src->cachedData);
        removeChunkFromList(src, m_firstCached, m_lastCached);
        addChunkToList(dst, m_firstCached, m_lastCached);
    }

    if (src->asyncOp)
    {
        if (src->asyncOp->dataOwner)
            src->asyncOp->dataOwner = dst;
        if (src->asyncOp->readTarget)
            src->asyncOp->readTarget = dst;
        dst->asyncOp = src->asyncOp;
        src->asyncOp = NULL;
    }

    removeChunk(src, false);
    m_dirty = true;
}

//------------------------------------------------------------------------

void ClusteredFile::initBuffer(Buffer& buffer)
{
    buffer.size = 0;
    buffer.base = NULL;
    buffer.ptr  = NULL;
}

//------------------------------------------------------------------------

void ClusteredFile::allocBuffer(Buffer& buffer, int size)
{
    pushMemOwner("ClusteredFile buffers");
    buffer.size  = max(size, 1) + m_clusterSize - 1;
    buffer.size -= buffer.size % m_clusterSize;
    buffer.base  = (U8*)malloc(buffer.size + m_clusterSize - 1);
    buffer.ptr   = buffer.base + m_clusterSize - 1;
    buffer.ptr  -= (UPTR)buffer.ptr % (UPTR)m_clusterSize;
    popMemOwner();
}

//------------------------------------------------------------------------

void ClusteredFile::freeBuffer(Buffer& buffer)
{
    free(buffer.base);
    initBuffer(buffer);
}

//------------------------------------------------------------------------

void ClusteredFile::clearInternal(void)
{
    m_freeClusters.reset();
    m_clusters.reset();
    for (int i = 0; i < m_groups.getSize(); i++)
    {
        Group* g = m_groups[i];
        for (int j = 0; j < g->chunks.getSize(); j++)
        {
            cacheEvict(g->chunks[j]);
            delete g->chunks[j];
        }
        delete g;
    }
    m_groups.reset();
}

//------------------------------------------------------------------------

bool ClusteredFile::readMasterChunk(void)
{
    if (!m_file.getSize())
        return false;

    m_file.seek(0);
    BufferedInputStream in(m_file);

    // MasterHeader.

    char formatID[9];
    in.readFully(formatID, 8);
    formatID[8] = '\0';
    if (String(formatID) != "Clusters")
        setError("Not a clustered file!");

    S32 formatVersion;
    in >> formatVersion;
    if (formatVersion != 2)
        setError("Unsupported clustered file version!");

    S32 numClusters, clusterSize, numChunks, defaultCompression;
    in >> numClusters >> clusterSize >> numChunks >> defaultCompression;
    if (numClusters < 0 || clusterSize <= 0 || numChunks < 0 || defaultCompression < 0 || defaultCompression >= Compression_Max)
        setError("Corrupt master header!");

    // Array of ChunkInfo.

    for (int i = 0; i < numChunks && !hasError(); i++)
    {
        S32 group, id, firstCluster, compression, compressedSize, uncompressedSize;
        in >> group >> id >> firstCluster >> compression >> compressedSize >> uncompressedSize;
        if (group < 0 ||
            id < 0 ||
            firstCluster < -1 || firstCluster >= numClusters ||
            compression < 0 || compression >= Compression_Max ||
            compressedSize < 0 ||
            uncompressedSize < 0 ||
            (compression == Compression_None && compressedSize != uncompressedSize) ||
            exists(group, id))
        {
            setError("Corrupt chunk info!");
        }
        else
        {
            Chunk* chunk            = createChunk(group, id);
            chunk->firstCluster     = firstCluster;
            chunk->compression      = (Compression)compression;
            chunk->compressedSize   = compressedSize;
            chunk->uncompressedSize = uncompressedSize;
        }
    }

    // Array of ClusterInfo.

    if (!hasError())
    {
        m_clusters.reset(numClusters);
        for (int i = 0; i < numClusters; i++)
        {
            in >> m_clusters[i].next;
            if (m_clusters[i].next == -1)
                m_freeClusters.add(i, i);
        }
    }

    // Check that MasterChunk is valid.

    if (!exists(GroupID_Private, PrivateChunkID_Master))
        setError("No master chunk!");
    else
    {
        const Chunk* master = get(GroupID_Private, PrivateChunkID_Master);
        if (master->firstCluster != 0 || master->compression != Compression_None || master->uncompressedSize != (7 + numChunks * 6 + numClusters) * (int)sizeof(S32))
            setError("Corrupt master chunk!");

        int masterClusters = (master->uncompressedSize + clusterSize - 1) / clusterSize;
        for (int i = 0; i < masterClusters; i++)
            if (i >= numClusters || m_clusters[i].next != ((i < masterClusters - 1) ? i + 1 : FW_S32_MAX))
                setError("Corrupt master chunk!");
    }

    // Handle errors.

    if (hasError())
        return false;

    // Set rest of the members.

    m_clusterSize = clusterSize;
    m_defaultCompression = (Compression)defaultCompression;
    return true;
}

//------------------------------------------------------------------------

void ClusteredFile::writeMasterChunk(void)
{
    // Get the previous MasterChunk.

    FW_ASSERT(exists(GroupID_Private, PrivateChunkID_Master));
    Chunk* master = get(GroupID_Private, PrivateChunkID_Master);
    FW_ASSERT(master->firstCluster == 0);
    int masterClusters = (master->uncompressedSize + m_clusterSize - 1) / m_clusterSize;
    FW_ASSERT(masterClusters <= m_clusters.getSize());

    // Calculate current MasterChunk size.

    int numChunks = 0;
    for (int i = 0; i < m_groups.getSize(); i++)
        for (int j = 0; j < m_groups[i]->chunks.getSize(); j++)
            if (get(i, j)->firstCluster != -1)
                numChunks++;
    master->uncompressedSize = (7 + numChunks * 6 + m_clusters.getSize()) * sizeof(S32);

    // MasterChunk has grown => allocate additional clusters.

    if (master->uncompressedSize > masterClusters * m_clusterSize)
    {
        Array<Backlink> backlinks;
        gatherBacklinks(backlinks);
        for (; master->uncompressedSize > masterClusters * m_clusterSize; masterClusters++)
        {
            // Free cluster => allocate.

            if (masterClusters < m_clusters.getSize() && m_clusters[masterClusters].next == -1)
            {
                m_freeClusters.remove(masterClusters);
                continue;
            }

            // Allocate a new cluster.

            int free;
            if (!m_freeClusters.isEmpty())
                free = m_freeClusters.removeMin();
            else
            {
                free = m_clusters.getSize();
                m_clusters.add();
                master->uncompressedSize += sizeof(S32);
                backlinks.add();
            }

            // Move the offending cluster.

            Array<U8> clusterData(NULL, m_clusterSize);

            if (free == masterClusters)
                memset(clusterData.getPtr(), 0, m_clusterSize);
            else
            {
                FW_ASSERT(m_clusters[masterClusters].next >= 0);
                m_file.seek((S64)masterClusters * m_clusterSize);
                m_file.readFully(clusterData.getPtr(), m_clusterSize);

                const Backlink& bl = backlinks[masterClusters];
                if (bl.cluster == -1)
                    bl.chunk->firstCluster = free;
                else
                    m_clusters[bl.cluster].next = free;
                backlinks[free] = backlinks[masterClusters];

                int next = m_clusters[masterClusters].next;
                m_clusters[free].next = next;
                if (next != FW_S32_MAX)
                    backlinks[next].cluster = free;
            }

            m_file.seek((S64)free * m_clusterSize);
            m_file.write(clusterData.getPtr(), m_clusterSize);
        }
    }

    // MasterChunk has shrunk => free unused clusters.

    while (master->uncompressedSize <= (masterClusters - 1) * m_clusterSize)
    {
        masterClusters--;
        m_clusters[masterClusters].next = -1;
        m_freeClusters.add(masterClusters, masterClusters);
    }

    // Update MasterChunk entries.

    master->compressedSize = master->uncompressedSize;
    for (int i = 0; i < masterClusters - 1; i++)
        m_clusters[i].next = i + 1;
    m_clusters[masterClusters - 1].next = FW_S32_MAX;

    // MasterHeader.

    m_file.seek(0);
    BufferedOutputStream out(m_file);
    out.write("Clusters", 8);
    out << (S32)2 << m_clusters.getSize() << m_clusterSize << (S32)numChunks << (S32)m_defaultCompression;

    // Array of ChunkInfo.

    for (int i = 0; i < m_groups.getSize(); i++)
    {
        for (int j = 0; j < m_groups[i]->chunks.getSize(); j++)
        {
            const Chunk* chunk = get(i, j);
            if (chunk->firstCluster != -1)
                out << (S32)i << (S32)j << chunk->firstCluster << (S32)chunk->compression << chunk->compressedSize << chunk->uncompressedSize;
        }
    }

    // Array of ClusterInfo.

    for (int i = 0; i < m_clusters.getSize(); i++)
        out << m_clusters[i].next;

    // Flush MasterChunk.

    out.flush();
    FW_ASSERT(hasError() || m_file.getOffset() == master->uncompressedSize);
}

//------------------------------------------------------------------------

void ClusteredFile::gatherBacklinks(Array<Backlink>& backlinks)
{
    backlinks.reset(m_clusters.getSize());
    for (int i = 0; i < m_groups.getSize(); i++)
    {
        for (int j = 0; j < m_groups[i]->chunks.getSize(); j++)
        {
            int prev = -1;
            Chunk* chunk = get(i, j);
            int cluster = chunk->firstCluster;
            while (cluster != FW_S32_MAX)
            {
                backlinks[cluster].cluster  = prev;
                backlinks[cluster].chunk    = (prev == -1) ? chunk : NULL;
                prev                        = cluster;
                cluster                     = m_clusters[prev].next;
            }
        }
    }
}

//------------------------------------------------------------------------

ClusteredFile::Chunk* ClusteredFile::createChunk(int groupID, int chunkID)
{
    FW_ASSERT(!exists(groupID, chunkID));

    while (m_groups.getSize() <= groupID)
    {
        Group* g        = m_groups.add(new Group);
        g->id           = m_groups.getSize() - 1;
        g->firstFree    = NULL;
        g->lastFree     = NULL;
    }

    Group* g = m_groups[groupID];
    while (g->chunks.getSize() <= chunkID)
    {
        Chunk* c            = g->chunks.add(new Chunk);

        c->id               = g->chunks.getSize() - 1;
        c->group            = g;
        c->prev             = NULL;
        c->next             = NULL;

        c->firstCluster     = 0;
        c->compression      = Compression_None;
        c->compressedSize   = 0;
        c->uncompressedSize = 0;

        initBuffer(c->cachedData);
        c->cachedDataCompressed = false;
        c->asyncOp          = NULL;

        addChunkToList(c, g->firstFree, g->lastFree);
    }

    Chunk* c = g->chunks[chunkID];
    removeChunkFromList(c, g->firstFree, g->lastFree);
    return c;
}

//------------------------------------------------------------------------

void ClusteredFile::removeChunk(Chunk* c, bool freeClusters)
{
    FW_ASSERT(c);
    FW_ASSERT(c->firstCluster >= 0);

    cacheEvict(c);

    if (freeClusters)
    {
        S32 cluster = c->firstCluster;
        do
        {
            S32 prev = cluster;
            cluster = m_clusters[prev].next;
            m_clusters[prev].next = -1;
            m_freeClusters.add(prev, prev);
        }
        while (cluster != FW_S32_MAX);
    }
    c->firstCluster = -1;

    Group* g = c->group;
    addChunkToList(c, g->firstFree, g->lastFree);
    while (g->chunks.getSize() && g->chunks.getLast()->firstCluster == -1)
    {
        Chunk* last = g->chunks.removeLast();
        removeChunkFromList(last, g->firstFree, g->lastFree);
        delete last;
    }
}

//------------------------------------------------------------------------

void ClusteredFile::addChunkToList(Chunk* c, Chunk*& first, Chunk*& last)
{
    FW_ASSERT(c);

    c->prev = last;
    if (!c->prev)
        first = c;
    else
        c->prev->next = c;

    c->next = NULL;
    last = c;
}

//------------------------------------------------------------------------

void ClusteredFile::removeChunkFromList(Chunk* c, Chunk*& first, Chunk*& last)
{
    FW_ASSERT(c);

    if (!c->prev)
        first = c->next;
    else
        c->prev->next = c->next;

    if (!c->next)
        last = c->prev;
    else
        c->next->prev = c->prev;
}

//------------------------------------------------------------------------

const U8* ClusteredFile::cacheRead(Chunk* c, int size, bool needUncompressed)
{
    FW_ASSERT(c);

    // Prefetch and wait for the AsyncOp.

    if (!cacheReadPrefetch(c, size))
    {
        asyncWait(c->asyncOp);
        asyncFinish();
    }

    // Compressed => decompress.

    FW_ASSERT(c->cachedData.size);
    if (needUncompressed && c->cachedDataCompressed)
    {
        Buffer tmp;
        allocBuffer(tmp, c->uncompressedSize);
        decompress(tmp.ptr, c->uncompressedSize, c->cachedData.ptr, c->compressedSize, c->compression);

        m_cacheUsed += tmp.size - c->cachedData.size;
        freeBuffer(c->cachedData);
        c->cachedDataCompressed = false;
    }
    return c->cachedData.ptr;
}

//------------------------------------------------------------------------

bool ClusteredFile::cacheReadPrefetch(Chunk* c, int size)
{
    FW_ASSERT(c);
    FW_ASSERT(size >= 0 && size <= c->uncompressedSize);

    // Already cached => move to the end of the cached chunk list (most recently used).

    if (cacheReadIsReady(c, size))
    {
        removeChunkFromList(c, m_firstCached, m_lastCached);
        addChunkToList(c, m_firstCached, m_lastCached);
        return true;
    }

    // Already loading => done.

    if (c->asyncOp && c->asyncOp->readTarget &&
        (c->asyncOp->data.size >= size || c->compression != Compression_None))
    {
        return false;
    }

    // Evict old data - the chunk may have been read partially.

    cacheEvict(c);

    // Create AsyncOp.

    c->asyncOp = new AsyncOp;
    allocBuffer(c->asyncOp->data, size);
    c->asyncOp->dataOwner = NULL;
    c->asyncOp->readTarget = c;

    // Start async reads.

    FW_ASSERT(c->firstCluster >= 0 && c->firstCluster < m_clusters.getSize());
    int startCluster    = c->firstCluster;
    int startOfs        = 0;
    int currCluster     = startCluster;
    int currOfs         = 0;

    while (currOfs < c->asyncOp->data.size)
    {
        FW_ASSERT(currCluster != FW_S32_MAX);
        int prevCluster = currCluster;
        currCluster = m_clusters[prevCluster].next;
        currOfs += m_clusterSize;

        if (currOfs == c->asyncOp->data.size ||
            currCluster != prevCluster + 1 ||
            currOfs - startOfs + m_clusterSize > File::MaxBytesPerSysCall)
        {
            asyncStartRange(c->asyncOp, startOfs, startCluster, prevCluster - startCluster + 1, false);
            startCluster = currCluster;
            startOfs = currOfs;
        }
    }

    m_asyncOps.add(c->asyncOp);
    return false;
}

//------------------------------------------------------------------------

bool ClusteredFile::cacheReadIsReady(Chunk* c, int size)
{
    FW_ASSERT(c);
    FW_ASSERT(size >= 0 && size <= c->uncompressedSize);

    asyncFinish();
    return (c->cachedData.size >= size ||
        (c->cachedData.size && c->compression != Compression_None));
}

//------------------------------------------------------------------------

void ClusteredFile::cacheWrite(Chunk* c, const void* data, int size, int compressedSize)
{
    FW_ASSERT(c);

    // Replace cached data.

    cacheEvict(c);
    FW_ASSERT(!c->cachedData.size);

    if (compressedSize == -1)
    {
        c->uncompressedSize     = size;
        c->compressedSize       = size; // fixed below
        c->cachedDataCompressed = false;
        allocBuffer(c->cachedData, size);
        memcpy(c->cachedData.ptr, data, size);
    }
    else
    {
        c->uncompressedSize     = size;
        c->compressedSize       = compressedSize;
        c->cachedDataCompressed = true;
        allocBuffer(c->cachedData, compressedSize);
        memcpy(c->cachedData.ptr, data, compressedSize);
    }

    m_cacheUsed += c->cachedData.size;
    addChunkToList(c, m_firstCached, m_lastCached);

    // Create AsyncOp.

    AsyncOp* op = new AsyncOp;
    op->readTarget = NULL;

    if (c->compression == Compression_None || c->cachedDataCompressed)
    {
        op->data = c->cachedData;
        op->dataOwner = c;
        c->asyncOp = op;
    }
    else
    {
        Array<U8> tmp;
        compress(tmp, c->cachedData.ptr, c->uncompressedSize, c->compression);
        c->compressedSize = tmp.getSize();

        allocBuffer(op->data, tmp.getSize());
        memcpy(op->data.ptr, tmp.getPtr(), tmp.getSize());
        op->dataOwner = NULL;
    }

    // Start async writes.

    bool    grow            = (op->data.size > m_freeClusters.numItems() * m_clusterSize);
    int     startCluster    = -1;
    int     startOfs        = 0;
    int     currCluster     = -1;
    int     currOfs         = 0;

    while (currCluster != FW_S32_MAX)
    {
        // Advance offset.

        int prevCluster = currCluster;
        if (prevCluster != -1)
            currOfs += m_clusterSize;

        // Allocate cluster.

        if (currOfs == op->data.size && prevCluster != -1)
            currCluster = FW_S32_MAX;
        else if (!grow)
            currCluster = m_freeClusters.removeMin();
        else
        {
            currCluster = m_clusters.getSize();
            m_clusters.add();
        }

        // First cluster => handle as a special case.

        if (prevCluster == -1)
        {
            startCluster = currCluster;
            c->firstCluster = currCluster;
            continue;
        }

        // Link to the previous cluster.
        // Seek => write preceding clusters.

        m_clusters[prevCluster].next = currCluster;
        if (currCluster != prevCluster + 1 ||
            currOfs - startOfs + m_clusterSize > File::MaxBytesPerSysCall)
        {
            asyncStartRange(op, startOfs, startCluster, prevCluster - startCluster + 1, true);
            startCluster = currCluster;
            startOfs = currOfs;
        }
    }

    m_asyncOps.add(op);
}

//------------------------------------------------------------------------

void ClusteredFile::cacheCopy(Chunk* dst, ClusteredFile& srcFile, Chunk* src)
{
    FW_ASSERT(dst && src);
    FW_ASSERT(dst != src);

    const U8* data = srcFile.cacheRead(src, src->uncompressedSize, (dst->compression != src->compression));
    if (src->cachedDataCompressed)
        cacheWrite(dst, data, src->uncompressedSize, src->compressedSize);
    else
        cacheWrite(dst, data, src->uncompressedSize);
}

//------------------------------------------------------------------------

void ClusteredFile::cacheEvict(Chunk* c)
{
    FW_ASSERT(c);

    // Remove from cache.

    if (c->cachedData.size)
    {
        m_cacheUsed -= c->cachedData.size;
        removeChunkFromList(c, m_firstCached, m_lastCached);
    }

    // Detach pending AsyncOp.

    if (c->asyncOp)
    {
        if (c->asyncOp->dataOwner)
            initBuffer(c->cachedData);

        c->asyncOp->readTarget  = NULL;
        c->asyncOp->dataOwner   = NULL;
        c->asyncOp              = NULL;
    }

    // Delete data.

    freeBuffer(c->cachedData);
}

//------------------------------------------------------------------------

void ClusteredFile::cacheEvict(void)
{
    asyncFinish();
    while (m_cacheUsed > m_cacheSize)
        cacheEvict(m_firstCached);
}

//------------------------------------------------------------------------

void ClusteredFile::asyncStartRange(AsyncOp* op, int dataOfs, int firstCluster, int numClusters, bool isWrite)
{
    FW_ASSERT(op);
    AsyncRange& range = op->ranges.add();

    range.firstCluster  = firstCluster;
    range.numClusters   = numClusters;
    range.isWrite       = isWrite;

    for (int i = 0; i < numClusters; i++)
    {
        Cluster& c = m_clusters[firstCluster + i];
        if (isWrite)
        {
            while (c.pendingReads || c.pendingWrites)
                asyncStall();
            c.pendingWrites++;
        }
        else
        {
            while (c.pendingWrites)
                asyncStall();
            c.pendingReads++;
        }
    }

    m_file.seek((S64)firstCluster * m_clusterSize);
    if (isWrite)
        range.fileOp = m_file.writeAsync(op->data.ptr + dataOfs, numClusters * m_clusterSize);
    else
        range.fileOp = m_file.readAsync(op->data.ptr + dataOfs, numClusters * m_clusterSize);
    m_asyncBytesPending += numClusters * m_clusterSize;
}

//------------------------------------------------------------------------

void ClusteredFile::asyncEndRange(AsyncRange& range)
{
    for (int i = 0; i < range.numClusters; i++)
    {
        Cluster& c = m_clusters[range.firstCluster + i];
        if (range.isWrite)
            c.pendingWrites--;
        else
            c.pendingReads--;
    }
    m_asyncBytesPending -= range.numClusters * m_clusterSize;
    delete range.fileOp;
}

//------------------------------------------------------------------------

void ClusteredFile::asyncWait(AsyncOp* op)
{
    FW_ASSERT(op);
    for (int i = 0; i < op->ranges.getSize(); i++)
        op->ranges[i].fileOp->wait();
}

//------------------------------------------------------------------------

void ClusteredFile::asyncFinish(void)
{
    for (int i = 0; i < m_asyncOps.getSize(); i++)
    {
        // Poll file ops and delete finished ones.

        AsyncOp* op = m_asyncOps[i];
        while (op->ranges.getSize() && op->ranges.getLast().fileOp->isDone())
            asyncEndRange(op->ranges.removeLast());

        // Unfinished ops still exist => skip.

        if (op->ranges.getSize())
            continue;

        // Reading to cache => copy data to the target.

        Chunk* c = op->readTarget;
        if (c)
        {
            c->asyncOp = NULL;
            cacheEvict(c);

            c->cachedData = op->data;
            c->cachedDataCompressed = (c->compression != Compression_None);
            m_cacheUsed += c->cachedData.size;
            initBuffer(op->data);

            addChunkToList(c, m_firstCached, m_lastCached);
        }

        // Release the data array.

        if (op->dataOwner)
            op->dataOwner->asyncOp = NULL;
        else
            freeBuffer(op->data);

        // Delete the op.

        m_asyncOps.removeSwap(i);
        delete op;
        i--;
    }
}

//------------------------------------------------------------------------

void ClusteredFile::asyncStall(void)
{
    asyncFinish();
    if (!m_asyncOps.getSize())
        return;

    asyncWait(m_asyncOps.getFirst());
    asyncFinish();
}

//------------------------------------------------------------------------

void ClusteredFile::compress(Array<U8>& compressed, const void* data, int uncompressedSize, Compression compression)
{
    int level;
    switch (compression)
    {
    case Compression_None:
        compressed.reset(uncompressedSize);
        memcpy(compressed.getPtr(), data, uncompressedSize);
        return;

    case Compression_ZLibLow:
        level = 1;
        break;

    case Compression_ZLibMedium:
        level = 6;
        break;

    case Compression_ZLibHigh:
        level = 9;
        break;

    default:
        FW_ASSERT(false);
        return;
    }

#if FW_USE_ZLIB
    uLongf destLen = uncompressedSize + uncompressedSize / 1000 + 14; // "must be at least 0.1% larger than sourceLen plus 12 bytes"
    compressed.reset(destLen);
    if (compress2((Bytef*)compressed.getPtr(), &destLen, (const Bytef*)data, uncompressedSize, level) != Z_OK)
        fail("ZLib compress2() failed!");
    compressed.resize(destLen);
#else
    fail("ClusteredFile: ZLib compression not supported!");
#endif
}

//------------------------------------------------------------------------

void ClusteredFile::decompress(void* data, int uncompressedSize, const void* compressed, int compressedSize, Compression compression)
{
    switch (compression)
    {
    case Compression_None:
        memcpy(data, compressed, compressedSize);
        return;

    case Compression_ZLibLow:
    case Compression_ZLibMedium:
    case Compression_ZLibHigh:
        break;

    default:
        FW_ASSERT(false);
        return;
    }

#if FW_USE_ZLIB
    uLongf destLen = uncompressedSize;
    if (uncompress((Bytef*)data, &destLen, (const Bytef*)compressed, compressedSize) != Z_OK)
        fail("ZLib uncompress() failed!");
    if ((int)destLen != uncompressedSize)
        fail("ZLib uncompress() produced wrong amount of data!");
#else
    fail("ClusteredFile: ZLib compression not supported!");
    FW_UNREF(uncompressedSize);
#endif
}

//------------------------------------------------------------------------
