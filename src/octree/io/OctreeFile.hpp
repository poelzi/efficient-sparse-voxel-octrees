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
#include "ClusteredFile.hpp"
#include "AttachIO.hpp"

namespace FW
{
//------------------------------------------------------------------------

class MeshBase;
class OctreeSlice;

//------------------------------------------------------------------------

class OctreeFile
{
public:
    enum
    {
        UnitScale               = 23,
        MaxPrefetchSlices       = 256,
        MaxPrefetchBytesTotal   = 64 << 20,
    };

    enum SliceState
    {
        SliceState_Unused = 0,  // Does not exist.
        SliceState_Unbuilt,     // Contains BuildDataAttach, but no nodes.
        SliceState_Complete,    // Contains nodes.

        SliceState_Max
    };

    struct Object
    {
        Mat4f           objectToWorld;
        Mat4f           octreeToObject;
        S32             rootSlice;
        Array<AttachIO::AttachType> runtimeAttachTypes;
    };

private:
    enum GroupID
    {
        GroupID_Static = ClusteredFile::GroupID_Default,
        GroupID_Slices,
        GroupID_Meshes
    };

    enum StaticChunkID
    {
        StaticChunkID_Octree = 0
    };

    struct ObjectInfo
    {
        Object          object;
        MeshBase*       mesh;
        bool            meshValid;
    };

public:
                        OctreeFile          (const String& fileName, File::Mode mode, int clusterSize = ClusteredFile::DefaultClusterSize);
                        ~OctreeFile         (void);

    const String&       getName             (void) const            { return m_file.getName(); }
    File::Mode          getMode             (void) const            { return m_file.getMode(); }
    S64                 getFileSize         (void)                  { return m_file.getFileSize(); }
    bool                checkWritable       (void) const            { return m_file.checkWritable(); }

    void                setCompression      (ClusteredFile::Compression compression) { m_file.setCompression(compression); }
    ClusteredFile::Compression getCompression(void) const           { return m_file.getCompression(); }
    S64                 getCacheSize        (void) const            { return m_file.getCacheSize(); }
    void                setCacheSize        (S64 size)              { m_file.setCacheSize(size); }
    S64                 getAsyncBytesPending(void) const            { return m_file.getAsyncBytesPending(); }

    void                clear               (void);
    void                clearSlices         (void);
    void                clearSlices         (int objID);
    void                set                 (OctreeFile& other, int maxLevels = UnitScale, bool includeMeshes = true, bool enablePrints = false);
    void                flush               (bool clearCache = true);

    int                 addObject           (void);
    int                 getNumObjects       (void) const            { return m_objects.getSize(); }
    const Object&       getObject           (int objID) const       { return m_objects[objID].object; }
    void                setObject           (int objID, const Object& obj);
    MeshBase*           getMesh             (int objID);
    MeshBase*           getMeshCopy         (int objID);
    void                setMesh             (int objID, MeshBase* mesh);

    int                 getNumSliceIDs      (void) const            { return m_file.getNumIDs(GroupID_Slices); }
    int                 getFreeSliceID      (void) const            { return m_file.getFreeID(GroupID_Slices); }
    bool                hasSlice            (int sliceID) const     { return m_file.exists(GroupID_Slices, sliceID); }
    int                 getSliceSize        (int sliceID)           { return m_file.getSize(GroupID_Slices, sliceID); }
    SliceState          getSliceState       (int sliceID) const;

    void                readSlice           (int sliceID, OctreeSlice& slice);
    void                readSlicePrefetch   (int sliceID);
    bool                readSliceIsReady    (int sliceID);

    void                writeSlice          (const OctreeSlice& slice);
    void                removeSlice         (int sliceID);

    void                printStats          (void);

    OctreeFile&         operator=           (OctreeFile& other)     { set(other); return *this; }

private:
    void                clearInternal       (void);
    void                setSliceState       (int sliceID, SliceState state);

    bool                readOctreeChunk     (void);
    void                writeOctreeChunk    (void);

private:
                        OctreeFile          (const OctreeFile&); // forbidden

private:
    ClusteredFile       m_file;
    bool                m_octreeChunkDirty;
    Array<ObjectInfo>   m_objects;
    Array<U32>          m_sliceState; // 16 values per dword
};

//------------------------------------------------------------------------

class OctreeSlice // all offsets and sizes are dword-based
{
public:
    enum SliceInfo
    {
        SliceInfo_ID = 0,
        SliceInfo_State,
        SliceInfo_CubePos,
        SliceInfo_CubeScale = 5,
        SliceInfo_NodeScale,
        SliceInfo_NumChildEntries,
        SliceInfo_ChildEntryPtr,
        SliceInfo_NumAttach,
        SliceInfo_AttachInfoPtr,
        SliceInfo_NumNodes,
        SliceInfo_NodeSplitPtr,
        SliceInfo_NumSplitNodes,
        SliceInfo_NodeValidMaskPtr,

        SliceInfo_End
    };

    enum ChildEntry
    {
        ChildEntry_NoChild  = -1,
        ChildEntry_Split    = -2
    };

    enum AttachInfo
    {
        AttachInfo_Type,
        AttachInfo_Ptr,
        AttachInfo_Size,

        AttachInfo_End
    };

public:
                        OctreeSlice         (void)                  {}
                        OctreeSlice         (OctreeSlice& other)    : m_data(other.m_data) {}
                        ~OctreeSlice        (void)                  {}

    const Array<S32>&   getData             (void) const            { return m_data; }
    Array<S32>&         getData             (void)                  { return m_data; }
    int                 getSize             (void) const            { return m_data.getSize(); }
    S32                 get                 (int idx) const         { return m_data[idx]; }
    S32&                get                 (int idx)               { return m_data[idx]; }
    void                set                 (int idx, S32 value)    { m_data.set(idx, value); }

    void                init                (int numChildEntries, int maxAttach, int numNodes, int numSplitNodes);

    int                 getID               (void) const            { return get(SliceInfo_ID); }
    void                setID               (int id)                { set(SliceInfo_ID, id); }
    OctreeFile::SliceState getState         (void) const            { return (OctreeFile::SliceState)get(SliceInfo_State); }
    void                setState            (OctreeFile::SliceState state) { set(SliceInfo_State, state); }
    Vec3i               getCubePos          (void) const            { int i = SliceInfo_CubePos; return Vec3i(get(i), get(i + 1), get(i + 2)); }
    void                setCubePos          (const Vec3i& v)        { int i = SliceInfo_CubePos; set(i, v.x); set(i + 1, v.y); set(i + 2, v.z); }
    int                 getCubeScale        (void) const            { return get(SliceInfo_CubeScale); }
    void                setCubeScale        (int v)                 { set(SliceInfo_CubeScale, v); }
    int                 getNodeScale        (void) const            { return get(SliceInfo_NodeScale); }
    void                setNodeScale        (int v)                 { set(SliceInfo_NodeScale, v); }

    int                 getNumChildEntries  (void) const            { return get(SliceInfo_NumChildEntries); }
    int                 getChildEntryOfs    (void) const            { return get(SliceInfo_ChildEntryPtr); }
    const S32*          getChildEntryPtr    (int idx = 0) const     { FW_ASSERT(idx >= 0 && idx <= getNumChildEntries()); return m_data.getPtr(getChildEntryOfs() + idx); }
    S32*                getChildEntryPtr    (int idx = 0)           { FW_ASSERT(idx >= 0 && idx <= getNumChildEntries()); return m_data.getPtr(getChildEntryOfs() + idx); }
    int                 getChildEntry       (int idx) const         { FW_ASSERT(idx < getNumChildEntries()); return *getChildEntryPtr(idx); }
    void                setChildEntry       (int idx, int v)        { FW_ASSERT(idx < getNumChildEntries()); *getChildEntryPtr(idx) = v; }

    int                 getNumAttach        (void) const            { return get(SliceInfo_NumAttach); }
    int                 getAttachInfoOfs    (int attachIdx) const   { FW_ASSERT(attachIdx >= 0 && attachIdx < getNumAttach()); return get(SliceInfo_AttachInfoPtr) + attachIdx * AttachInfo_End; }
    AttachIO::AttachType getAttachType      (int attachIdx) const   { return (AttachIO::AttachType)get(getAttachInfoOfs(attachIdx) + AttachInfo_Type); }
    int                 getAttachOfs        (int attachIdx) const   { return get(getAttachInfoOfs(attachIdx) + AttachInfo_Ptr); }
    int                 getAttachSize       (int attachIdx) const   { return get(getAttachInfoOfs(attachIdx) + AttachInfo_Size); }
    void                startAttach         (AttachIO::AttachType type);
    void                endAttach           (void);

    int                 getNumNodes         (void) const            { return get(SliceInfo_NumNodes); }
    int                 getNodeSplitOfs     (void) const            { return get(SliceInfo_NodeSplitPtr); }
    const U32*          getNodeSplitPtr     (void) const            { return (const U32*)m_data.getPtr(getNodeSplitOfs()); }
    U32*                getNodeSplitPtr     (void)                  { return (U32*)m_data.getPtr(getNodeSplitOfs()); }
    bool                isNodeSplit         (int nodeIdx) const;
    void                setNodeSplit        (int nodeIdx, bool isSplit);

    int                 getNumSplitNodes    (void) const            { return get(SliceInfo_NumSplitNodes); }
    int                 getNodeValidMaskOfs (void) const            { return get(SliceInfo_NodeValidMaskPtr); }
    const U8*           getNodeValidMaskPtr (int splitNodeIdx = 0) const { FW_ASSERT(splitNodeIdx >= 0 && splitNodeIdx <= getNumSplitNodes()); return (const U8*)m_data.getPtr(getNodeValidMaskOfs()) + splitNodeIdx; }
    U8*                 getNodeValidMaskPtr (int splitNodeIdx = 0)  { FW_ASSERT(splitNodeIdx >= 0 && splitNodeIdx <= getNumSplitNodes()); return (U8*)m_data.getPtr(getNodeValidMaskOfs()) + splitNodeIdx; }
    U8                  getNodeValidMask    (int splitNodeIdx) const { FW_ASSERT(splitNodeIdx < getNumSplitNodes()); return *getNodeValidMaskPtr(splitNodeIdx); }
    void                setNodeValidMask    (int splitNodeIdx, U32 validMask) { FW_ASSERT(splitNodeIdx < getNumSplitNodes()); *getNodeValidMaskPtr(splitNodeIdx) = (U8)validMask; }
    bool                hasNodeChild        (int splitNodeIdx, int childIdx) const { FW_ASSERT(childIdx >= 0 && childIdx < 8); return ((getNodeValidMask(splitNodeIdx) & (1 << childIdx)) != 0); }

    OctreeSlice&        operator=           (const OctreeSlice& other) { m_data.set(other.m_data); return *this; }
    S32                 operator[]          (int idx) const         { return get(idx); }
    S32&                operator[]          (int idx)               { return get(idx); }

private:
    Array<S32>          m_data;
};

//------------------------------------------------------------------------
/*

Octree file format v1
---------------------

- based on "Clustered file format v1"
- stores meshes in "Binary mesh file format v1"
- the basic unit of data is 32-bit little-endian int
- all pointers are dword-based (as opposed to byte-based)

Group       ID
    1       0       struct  OctreeChunk
    2       sliceID struct  Slice
    3       objID   file    BinaryMesh

OctreeChunk
    0       5       struct  OctreeHeader
    5       n*?     struct  array of Object (OctreeHeader.numObjects)
    ?       n*1     struct  array of SliceStatePacked ((OctreeHeader.numSlices + 15) / 16)
    ?

OctreeHeader
    0       2       bytes   formatID (must be "Octree  ")
    2       1       int     formatVersion (must be 1)
    3       1       int     numObjects
    4       1       int     numSlices
    5

Object
    0       34      struct  ObjectInfo
    34      n*1     int     array of attachment types (ObjectInfo.numRuntimeAttachTypes, see AttachIO::AttachType)
    ?

ObjectInfo
    0       16      float   objectToWorld: 4x4 matrix (column major)
    16      16      float   octreeToObject: 4x4 matrix (column major)
    32      1       int     rootSlice: slice id (-1 if none)
    33      1       int     numRuntimeAttachTypes
    34

SliceStatePacked (covers 16 slices)
    0.0     .2      bits    state of the 1st slice (see OctreeFile::SliceState)
    0.2     .2      bits    state of the 2nd slice (see OctreeFile::SliceState)
    ...
    1

Slice
    0       15      struct  SliceInfo
    15      n*1     struct  array of SliceChildEntry (SliceInfo.numChildEntries)
    ?       n*3     struct  array of SliceAttachInfo (SliceInfo.numAttach)
    ?       n*1     struct  array of SliceNodeSplitPacked ((SliceInfo.numNodes + 31) / 32)
    ?       n*1     struct  array of SliceNodeValidMaskPacked ((SliceInfo.numSplitNodes + 3) / 4)
    ?       n*?     struct  pool of SliceXxxAttach
    ?

SliceInfo
    0       1       int     id: slice id
    1       1       int     state (see OctreeFile::SliceState)
    2       3       uint    cubePos: position of the slice cube (0.23 fixed point, [0,0,0] for the root slice)
    5       1       int     cubeScale: log2 size of the cube (23 for the root slice)
    6       1       int     nodeScale: log2 size of nodes (23 for the root slice)
    7       1       int     numChildEntries: number of SliceChildEntry
    8       1       int     childEntryPtr: pointer to array of SliceChildEntry, relative to the Slice
    9       1       int     numAttach: number of attachments
    10      1       int     attachInfoPtr: pointer to array of SliceAttachInfo, relative to the Slice
    11      1       int     numNodes
    12      1       int     nodeSplitPtr: pointer to array of SliceNodeSplitPacked, relative to the Slice
    13      1       int     numSplitNodes
    14      1       int     nodeValidMaskPtr: pointer to array of SliceNodeValidMaskPacked, relative to the Slice
    15

SliceChildEntry (depth-first order)
    0       1       int     child slice ID, -1 if no child, -2 if split (see OctreeSlice::ChildEntry)
    1

SliceAttachInfo
    0       1       int     type (see AttachIO::AttachType)
    1       1       int     ptr: pointer to SliceXxxAttach, relative to the Slice
    2       1       int     size: size of the SliceXxxAttach
    3

SliceNodeSplitPacked (covers 32 nodes)
    0.0     .1      bits    split0: whether the 1st node is split
    0.1     .1      bits    split1: whether the 2nd node is split
    ...
    1

SliceNodeValidMaskPacked (covers 4 split nodes)
    0.0     .8      bits    validMask0: whether each child of the 1st node exists
    0.8     .8      bits    validMask1: whether each child of the 2nd node exists
    ...
    1

Attachments
    See AttachIO.h

*/
//------------------------------------------------------------------------
}
