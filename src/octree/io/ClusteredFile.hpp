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

#pragma once
#include "io/File.hpp"
#include "base/BinaryHeap.hpp"

namespace FW
{
//------------------------------------------------------------------------

class ClusteredFile
{
public:
    enum Compression
    {
        Compression_None = 0,
        Compression_ZLibLow,
        Compression_ZLibMedium,
        Compression_ZLibHigh,

        Compression_Max,
    };

    enum
    {
        DefaultCompression  = Compression_None,
        DefaultClusterSize  = 4096,
        DefaultCacheSize    = 128 << 20
    };

    enum GroupID
    {
        GroupID_Private = 0,
        GroupID_Default
    };

    enum PrivateChunkID
    {
        PrivateChunkID_Master = 0
    };

private:
    struct Chunk;
    struct AsyncOp;

    struct Cluster
    {
        S32             next;               // FW_S32_MAX if none, -1 if free
        S32             pendingReads;
        S32             pendingWrites;

        Cluster(void) : next(-1), pendingReads(0), pendingWrites(0) {}
    };

    struct Buffer
    {
        S32             size;
        U8*             base;
        U8*             ptr;
    };

    struct Group
    {
        S32             id;
        Chunk*          firstFree;          // NULL if none
        Chunk*          lastFree;           // NULL if none
        Array<Chunk*>   chunks;
    };

    struct Chunk
    {
        S32             id;
        Group*          group;
        Chunk*          prev;               // free or cached, NULL if none
        Chunk*          next;               // free or cached, NULL if none

        S32             firstCluster;       // -1 if the chunk doesn't exist
        Compression     compression;
        S32             compressedSize;
        S32             uncompressedSize;

        Buffer          cachedData;
        bool            cachedDataCompressed;
        AsyncOp*        asyncOp;
    };

    struct AsyncRange
    {
        S32             firstCluster;
        S32             numClusters;
        bool            isWrite;
        File::AsyncOp*  fileOp;
    };

    struct AsyncOp
    {
        Buffer          data;
        Chunk*          dataOwner;          // (!dataOwner || data == dataOwner->cachedData)
        Chunk*          readTarget;
        Array<AsyncRange> ranges;
    };

    struct Backlink
    {
        S32             cluster;            // -1 if none
        Chunk*          chunk;              // NULL if none
    };

public:
                        ClusteredFile       (const String& fileName, File::Mode mode, int clusterSize = DefaultClusterSize, bool disableCache = false);
                        ~ClusteredFile      (void);

    const String&       getName             (void) const                            { return m_file.getName(); }
    File::Mode          getMode             (void) const                            { return m_file.getMode(); }
    S64                 getFileSize         (void)                                  { return m_file.getSize(); }
    F32                 getFragmentsPerChunk(void) const;
    bool                checkWritable       (void) const                            { return m_file.checkWritable(); }

    int                 getClusterSize      (void) const                            { return m_clusterSize; }
    void                setCompression      (Compression compression);
    Compression         getCompression      (void) const                            { return m_defaultCompression; }
    S64                 getCacheSize        (void) const                            { return m_cacheSize; }
    void                setCacheSize        (S64 size)                              { FW_ASSERT(size >= 0); m_cacheSize = size; cacheEvict(); }
    S64                 getAsyncBytesPending(void) const                            { return m_asyncBytesPending; }

    void                clear               (void);
    void                append              (ClusteredFile& other);
    void                set                 (ClusteredFile& other)                  { if (&other != this) { clear(); append(other); } }
    void                flush               (bool clearCache = true);

    int                 getNumGroups        (void) const                            { return m_groups.getSize(); }
    int                 getNumIDs           (int groupID) const;
    int                 getFreeID           (int groupID) const;
    bool                exists              (int groupID, int chunkID) const;
    int                 getSize             (int groupID, int chunkID);
    int                 getSizeOnDisk       (int groupID, int chunkID);

    void                read                (int groupID, int chunkID, void* data, int size);
    template <class T> void read            (int groupID, int chunkID, Array<T>& data)     { data.reset((getSize(groupID, chunkID) + sizeof(T) - 1) / sizeof(T)); read(groupID, chunkID, data.getPtr(), data.getNumBytes()); }
    void                readPrefetch        (int groupID, int chunkID, int size);
    bool                readIsReady         (int groupID, int chunkID, int size);

    void                write               (int groupID, int chunkID, const void* data, int size);
    template <class T> void write           (int groupID, int chunkID, const Array<T>& data) { write(groupID, chunkID, data.getPtr(), data.getNumBytes()); }

    void                remove              (int groupID, int chunkID);
    void                copy                (int dstGroupID, int dstChunkID, ClusteredFile& srcFile, int srcGroupID, int srcChunkID);
    void                copy                (int dstGroupID, int dstChunkID, int srcGroupID, int srcChunkID) { copy(dstGroupID, dstChunkID, *this, srcGroupID, srcChunkID); }
    void                move                (int dstGroupID, int dstChunkID, int srcGroupID, int srcChunkID);

    ClusteredFile&      operator=           (ClusteredFile& other)                  { set(other); return *this; }
    ClusteredFile&      operator+=          (ClusteredFile& other)                  { append(other); return *this; }

private:
    Chunk*              get                 (int groupID, int chunkID) const        { return m_groups[groupID]->chunks[chunkID]; }

    void                initBuffer          (Buffer& buffer);
    void                allocBuffer         (Buffer& buffer, int size);
    void                freeBuffer          (Buffer& buffer);

    void                clearInternal       (void);
    bool                readMasterChunk     (void);
    void                writeMasterChunk    (void);
    void                gatherBacklinks     (Array<Backlink>& backlinks);

    Chunk*              createChunk         (int groupID, int chunkID);
    void                removeChunk         (Chunk* c, bool freeClusters);
    void                addChunkToList      (Chunk* c, Chunk*& first, Chunk*& last);
    void                removeChunkFromList (Chunk* c, Chunk*& first, Chunk*& last);

    const U8*           cacheRead           (Chunk* c, int size, bool needUncompressed = true);
    bool                cacheReadPrefetch   (Chunk* c, int size);
    bool                cacheReadIsReady    (Chunk* c, int size);
    void                cacheWrite          (Chunk* c, const void* data, int size, int compressedSize = -1);
    void                cacheCopy           (Chunk* dst, ClusteredFile& srcFile, Chunk* src);
    void                cacheEvict          (Chunk* c);
    void                cacheEvict          (void);

    void                asyncStartRange     (AsyncOp* op, int dataOfs, int firstCluster, int numClusters, bool isWrite);
    void                asyncEndRange       (AsyncRange& range);
    void                asyncWait           (AsyncOp* op);
    void                asyncFinish         (void);
    void                asyncStall          (void);

    static void         compress            (Array<U8>& compressed, const void* data, int uncompressedSize, Compression compression);
    static void         decompress          (void* data, int uncompressedSize, const void* compressed, int compressedSize, Compression compression);

private:
                        ClusteredFile       (const ClusteredFile&); // forbidden

private:
    File                m_file;
    S32                 m_clusterSize;
    Compression         m_defaultCompression;
    BinaryHeap<S32>     m_freeClusters;
    Array<Cluster>      m_clusters;
    Array<Group*>       m_groups;
    bool                m_dirty;

    Chunk*              m_firstCached;      // least recently used, NULL if none
    Chunk*              m_lastCached;       // most recently used, NULL if none
    S64                 m_cacheUsed;
    S64                 m_cacheSize;
    Array<AsyncOp*>     m_asyncOps;
    S64                 m_asyncBytesPending;
};

//------------------------------------------------------------------------
/*

Clustered file format v2
------------------------

- the basic unit of data is 32-bit little-endian int
- the entire file is an array of equally-sized clusters
- clusters form linked lists, each one storing a chunk
- chunks are identified by groupID and chunkID, both ranging from 0 to num-1
- groupID 0 is private, and cannot contain user chunks
- MasterChunk (groupID 0, chunkID 0) starts at the first cluster and is linear

MasterChunk
    0       7       struct  MasterHeader
    ?       n*6     struct  array of ChunkInfo (MasterHeader.numChunks)
    7       n*1     struct  array of ClusterInfo (MasterHeader.numClusters)
    ?

MasterHeader
    0       2       bytes   formatID (must be "Clusters")
    2       1       int     formatVersion (must be 2)
    3       1       int     numClusters
    4       1       int     clusterSize (bytes)
    5       1       int     numChunks
    6       1       int     defaultCompression (see ClusteredFile::Compression)
    7

ChunkInfo
    0       1       int     groupID
    1       1       int     chunkID
    2       1       int     firstCluster
    3       1       int     compression (see ClusteredFile::Compression)
    4       1       int     compressedSize
    5       1       int     uncompressedSize
    6

ClusterInfo
    0       1       bits    next: next cluster in the chunk (FW_S32_MAX if none, -1 if free)
    1

*/
//------------------------------------------------------------------------
}
