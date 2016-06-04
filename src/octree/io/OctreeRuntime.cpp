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

#include "OctreeRuntime.hpp"
#include "3d/Mesh.hpp"
#include "base/BinaryHeap.hpp"
#include "../Util.hpp"

using namespace FW;

//------------------------------------------------------------------------

OctreeRuntime::OctreeRuntime(MemoryManager::Mode mode)
:   m_mem                   (mode, PageBytes),
    m_numSlicesLoaded       (0),
    m_numNodesLoaded        (0),
    m_numNodeChildrenLoaded (0),
    m_loadSliceID           (-1)
{
}

//------------------------------------------------------------------------

OctreeRuntime::~OctreeRuntime(void)
{
    clear();
}

//------------------------------------------------------------------------

void OctreeRuntime::clear(void)
{
    m_mem.clear();

    for (int i = 0; i < m_objects.getSize(); i++)
    {
        delete m_objects[i]->attachIO;
        delete m_objects[i];
    }
    m_objects.clear();

    for (int i = 0; i < m_slices.getSize(); i++)
        delete m_slices[i];
    m_slices.clear();

    m_numSlicesLoaded       = 0;
    m_numNodesLoaded        = 0;
    m_numNodeChildrenLoaded = 0;
    m_loadSliceID           = -1;
}

//------------------------------------------------------------------------

bool OctreeRuntime::addObject(int objectID, int rootSliceID, const Array<AttachIO::AttachType>& runtimeAttachTypes)
{
    FW_ASSERT(objectID >= 0);

    // Ensure that the struct exists.

    while (objectID >= m_objects.getSize())
        m_objects.add(NULL);

    // Object exists => Check consistency.

    if (m_objects[objectID])
    {
        FW_ASSERT(m_objects[objectID]->rootSliceID == rootSliceID);
        return true;
    }

    // Create the object.

    Object* obj             = new Object;
    m_objects[objectID]     = obj;

    obj->rootSliceID        = rootSliceID;
    obj->attachIO           = new AttachIO(runtimeAttachTypes);
    obj->nodeAlign          = obj->attachIO->getNodeAlign();
    FW_ASSERT(obj->nodeAlign >= 2);
    FW_ASSERT((obj->nodeAlign & (obj->nodeAlign - 1)) == 0);

    obj->trunksPerPage      = (PageSize - obj->nodeAlign) / 32;
    obj->pagesPerTrunkBlock = (TrunksPerBlock + obj->trunksPerPage - 1) / obj->trunksPerPage;
    obj->trunkBlockInfoOfs  = obj->pagesPerTrunkBlock * PageSize;
    obj->trunkBlockSize     = obj->attachIO->layoutTrunkBlock(obj->trunksPerPage, obj->pagesPerTrunkBlock, obj->trunkBlockInfoOfs);

    obj->firstFreeTrunk     = -1;

    FW_ASSERT(obj->trunksPerPage > 0);
    FW_ASSERT(obj->nodeAlign/*PageHeader*/ + obj->trunksPerPage * (16/*Nodes*/ + 16/*PageHeader.farPtrs*/) <= PageSize);

    // Setup the root slice.

    Slice* slice = getOrCreateSlice(rootSliceID);
    FW_ASSERT(!slice->isReached);

    slice->trunkID = allocateTrunk(objectID);
    if (slice->trunkID == -1)
    {
        removeObject(objectID);
        return false;
    }

    slice->isReached    = true;
    slice->objectID     = objectID;
    slice->subtrunk     = 0;
    slice->cubePos      = Vec3i(0);
    slice->cubeScale    = OctreeFile::UnitScale;
    slice->nodeScale    = OctreeFile::UnitScale;
    return true;
}

//------------------------------------------------------------------------

void OctreeRuntime::removeObject(int objectID)
{
    Object* obj = m_objects[objectID];
    if (!obj)
        return;

    // Free slices.

    Array<S32> stack(obj->rootSliceID);
    while (stack.getSize())
    {
        Slice* slice = m_slices[stack.removeLast()];
        if (!slice->isReached)
            continue;

        slice->isReached = false;
        slice->isUnbuilt = false;
        slice->isLoaded = false;
        slice->numChildrenLoaded = 0;

        for (int i = 0; i < slice->blocks.getSize(); i++)
        {
            SliceBlock& sb = slice->blocks[i];
            freeBlock(sb.block);
            if (sb.childEntry >= 0)
                stack.add(sb.childEntry);
        }
    }

    // Free trunk blocks.

    for (int i = 0; i < obj->trunkBlocks.getSize(); i++)
        freeBlock(obj->trunkBlocks[i]);

    // Destroy object.

    delete obj->attachIO;
    delete obj;
    m_objects[objectID] = NULL;
}

//------------------------------------------------------------------------

bool OctreeRuntime::setSliceState(int sliceID, OctreeFile::SliceState state)
{
    FW_ASSERT(state >= 0 && state < OctreeFile::SliceState_Max);
    Slice* slice = getOrCreateSlice(sliceID);

    // Does not exist, already loaded, or not reached => not loadable.

    if (state == OctreeFile::SliceState_Unused || slice->isLoaded || !slice->isReached)
        return false;

    // Loadable if built.

    slice->isUnbuilt = (state == OctreeFile::SliceState_Unbuilt);
    return (!slice->isUnbuilt);
}

//------------------------------------------------------------------------

S32 OctreeRuntime::setSliceToLoad(const OctreeSlice& sliceData)
{
    struct StackEntry
    {
        const S32*  oldNode;
        S32         numLevels;
    };

    // No data => ignore.

    m_loadSliceID = -1;
    m_loadBuffer.reset();

    if (!sliceData.getSize())
        return 0;

    // Set state.

    int sliceID = sliceData.getID();
    if (!setSliceState(sliceID, sliceData.getState()))
        return 0;

    // Get slice info.

    Slice* slice = m_slices[sliceID];
    FW_ASSERT(slice->cubePos == sliceData.getCubePos());
    FW_ASSERT(slice->cubeScale == sliceData.getCubeScale());
    FW_ASSERT(slice->nodeScale == sliceData.getNodeScale());
    m_loadBlocks.resize(sliceData.getNumChildEntries());

    for (int i = 0; i < sliceData.getNumChildEntries(); i++)
    {
        LoadBlock& lb       = m_loadBlocks[i];
        lb.childEntry       = sliceData.getChildEntry(i);
        lb.data.clear();
        lb.rootNodeBlock    = NULL;
        lb.rootNode         = NULL;

        lb.splitTrunkID     = -1;
        lb.block.ofs        = -1;
        lb.block.size       = -1;
    }

    // Count nodes and their children.

    slice->numNodes = sliceData.getNumSplitNodes();
    slice->numNodeChildren = 0;
    for (int i = 0; i < slice->numNodes; i++)
        slice->numNodeChildren += popc8(sliceData.getNodeValidMask(i));

    // Download old block.

    Array<StackEntry> stack(NULL, 1);
    stack[0].oldNode = NULL;
    stack[0].numLevels = slice->cubeScale - slice->nodeScale + 1;

    if (slice->parentSliceID != -1)
    {
        const Block& block = m_slices[slice->parentSliceID]->blocks[slice->indexInParent].block;
        FW_ASSERT(block.ofs != -1);
        m_loadBuffer.reset(NULL, block.size * sizeof(S32), Buffer::Hint_None, PageBytes);
        m_loadBuffer.setRange(0, m_mem.getBuffer(), block.ofs * sizeof(S32), block.size * sizeof(S32));
        stack[0].oldNode = (const S32*)m_loadBuffer.getPtr() + m_objects[slice->objectID]->nodeAlign;
    }

    // Build new blocks.

    int nodeIdx = 0;
    int splitNodeIdx = 0;
    S32 memDelta = 0;

    if (slice->parentSliceID != -1)
        memDelta -= (S32)m_slices[slice->parentSliceID]->blocks[slice->indexInParent].block.size;

    m_objects[slice->objectID]->attachIO->beginSliceImport(&sliceData);

    for (int i = 0; i < m_loadBlocks.getSize(); i++)
    {
        LoadBlock& lb = m_loadBlocks[i];
        StackEntry curr = stack.removeLast();

        // Split => push children to stack.

        if (lb.childEntry == OctreeSlice::ChildEntry_Split)
        {
            lb.rootNodeBlock = (const S32*)m_loadBuffer.getPtr();
            lb.rootNode = curr.oldNode;

            for (int j = 7; j >= 0; j--)
            {
                StackEntry& child = stack.add();
                child.oldNode = NULL;
                if (curr.oldNode && isNodeChildNode(curr.oldNode, j))
                    child.oldNode = getNodeChild(curr.oldNode, j);
                child.numLevels = curr.numLevels - 1;
            }
            continue;
        }

        // Does not have any nodes => skip.

        if (!curr.oldNode && slice->parentSliceID != -1)
            continue;

        // Build block.

        gatherImportNodes(&sliceData, &nodeIdx, &splitNodeIdx, curr.oldNode, curr.numLevels);
        int blockInfoOfs = layoutImportNodes(slice->objectID);
        buildBlock(lb.data, blockInfoOfs, m_loadSliceID, i, slice->objectID);
        lb.rootNodeBlock = lb.data.getPtr();
        lb.rootNode = lb.rootNodeBlock + m_objects[slice->objectID]->nodeAlign;
        memDelta += (lb.data.getSize() + PageSize - 1) & -PageSize;
    }

    FW_ASSERT(nodeIdx == sliceData.getNumNodes());
    FW_ASSERT(splitNodeIdx == sliceData.getNumSplitNodes());

    // Let loadSlice() do the rest.

    m_loadSliceID = sliceID;
    return memDelta * sizeof(S32);
}

//------------------------------------------------------------------------

bool OctreeRuntime::loadSlice(void)
{
    struct StackEntry
    {
        Vec3i   cubePos;
        S32     cubeScale;
        S32     trunkID;
        S32     subtrunk;
    };

    if (m_loadSliceID == -1)
        return true;

    Slice* slice = m_slices[m_loadSliceID];
    Object* obj = m_objects[slice->objectID];
    bool ok = true;
    FW_ASSERT(obj);

    // Allocate trunks and blocks.

    for (int i = 0; i < m_loadBlocks.getSize() && ok; i++)
    {
        LoadBlock& lb = m_loadBlocks[i];
        if (!lb.rootNode)
            continue;

        if (lb.childEntry == OctreeSlice::ChildEntry_Split)
        {
            lb.splitTrunkID = allocateTrunk(slice->objectID);
            ok = (lb.splitTrunkID != -1);
        }
        else
        {
            lb.block = allocateBlock(lb.data.getSize());
            ok = (lb.block.ofs != -1);
        }
    }

    // Out of memory => fail.

    if (!ok)
    {
        for (int i = 0; i < m_loadBlocks.getSize(); i++)
        {
            LoadBlock& lb = m_loadBlocks[i];
            if (lb.splitTrunkID != -1)
                freeTrunk(slice->objectID, lb.splitTrunkID);
            lb.splitTrunkID = -1;
            freeBlock(lb.block);
        }
        return false;
    }

    // Free the old block.

    if (slice->parentSliceID != -1)
        freeBlock(m_slices[slice->parentSliceID]->blocks[slice->indexInParent].block);

    // Update children.

    slice->blocks.resize(m_loadBlocks.getSize());
    Array<StackEntry> stack(NULL, 1);

    stack[0].cubePos    = slice->cubePos;
    stack[0].cubeScale  = slice->cubeScale;
    stack[0].trunkID    = slice->trunkID;
    stack[0].subtrunk   = slice->subtrunk;

    for (int i = 0; i < m_loadBlocks.getSize(); i++)
    {
        StackEntry curr = stack.removeLast();
        const LoadBlock& lb = m_loadBlocks[i];
        SliceBlock& sb = slice->blocks[i];

        // Copy LoadBlock to SliceBlock.

        sb.childEntry   = lb.childEntry;
        sb.splitTrunkID = lb.splitTrunkID;
        sb.block        = lb.block;

        // Does not have any nodes => skip.

        if (!lb.rootNode)
            continue;

        // Setup trunk node, except for the child pointer.

        Trunk& trunk = obj->trunks[curr.trunkID];
        trunk.validMask[curr.subtrunk] = getNodeValidMask(lb.rootNode);
        trunk.nonLeafMask[curr.subtrunk] = getNodeNonLeafMask(lb.rootNode);

        obj->attachIO->copyNodeToSubtrunk(
            m_mem.getBuffer(), trunk.blockOfs, trunk.nodeOfs + curr.subtrunk * 2,
            trunk.pageInBlock, trunk.trunkInPage, curr.subtrunk,
            lb.rootNodeBlock, lb.rootNode);

        // Split => set trunk child pointer and push children to the stack.

        if (lb.childEntry == OctreeSlice::ChildEntry_Split)
        {
            trunk.childOfs[curr.subtrunk] = obj->trunks[lb.splitTrunkID].nodeOfs;
            updateTrunkNode(trunk, curr.subtrunk);

            for (int j = 7; j >= 0; j--)
            {
                StackEntry& child   = stack.add();
                child.cubePos       = getCubeChildPos(curr.cubePos, curr.cubeScale, j);
                child.cubeScale     = curr.cubeScale - 1;
                child.trunkID       = lb.splitTrunkID;
                child.subtrunk      = numNodeChildNodesBefore(lb.rootNode, j);
            }
            continue;
        }

        // Not split => set trunk child pointer and upload block.

        trunk.childOfs[curr.subtrunk] = lb.block.ofs + (getNodeChildren(lb.rootNode) - lb.rootNodeBlock);
        updateTrunkNode(trunk, curr.subtrunk);
        upload(lb.block.ofs, lb.data.getPtr(), lb.block.size);

        // Does not correspond to a child slice => done.

        if (lb.childEntry == OctreeSlice::ChildEntry_NoChild)
            continue;

        // Setup child slice.

        Slice* c = getOrCreateSlice(lb.childEntry);
        FW_ASSERT(!c->isReached && !c->isLoaded);

        c->isReached     = true;
        c->objectID      = slice->objectID;
        c->parentSliceID = m_loadSliceID;
        c->indexInParent = i;
        c->trunkID       = curr.trunkID;
        c->subtrunk      = curr.subtrunk;

        c->cubePos       = curr.cubePos;
        c->cubeScale     = curr.cubeScale;
        c->nodeScale     = slice->nodeScale - 1;
    }

    FW_ASSERT(!stack.getSize());

    // Update rest of the state.

    slice->isLoaded = true;
    if (slice->parentSliceID != -1)
        m_slices[slice->parentSliceID]->numChildrenLoaded++;

    m_numSlicesLoaded++;
    m_numNodesLoaded += slice->numNodes;
    m_numNodeChildrenLoaded += slice->numNodeChildren;

    m_loadSliceID = -1;
    m_loadBlocks.clear();
    m_loadBuffer.reset();
    return true;
}

//------------------------------------------------------------------------

void OctreeRuntime::unloadSlice(int sliceID)
{
    if (sliceID == -1 || !m_slices[sliceID]->isLoaded)
        return;

    Array<S32> stack(sliceID);
    while (stack.getSize())
    {
        int id = stack.removeLast();
        Slice* slice = m_slices[id];

        if (!slice->numChildrenLoaded)
        {
            unloadSliceInternal(id);
            continue;
        }

        stack.add(id);
        for (int i = 0; i < slice->blocks.getSize(); i++)
        {
            int cid = slice->blocks[i].childEntry;
            if (cid >= 0 && m_slices[cid]->isLoaded)
                stack.add(cid);
        }
    }
}

//------------------------------------------------------------------------

const S32* OctreeRuntime::getRootNodeCPU(int objectID)
{
    FW_ASSERT(m_mem.getMode() == MemoryManager::Mode_CPU);
    S64 ofs = getRootNodeOfs(objectID);
    if (ofs == -1)
        return NULL;
    return (const S32*)m_mem.getBuffer().getPtr(ofs * sizeof(S32));
}

//------------------------------------------------------------------------

CUdeviceptr OctreeRuntime::getRootNodeCuda(int objectID)
{
    FW_ASSERT(m_mem.getMode() == MemoryManager::Mode_Cuda);
    S64 ofs = getRootNodeOfs(objectID);
    if (ofs == -1)
        return NULL;
    return m_mem.getBuffer().getCudaPtr(ofs * sizeof(S32));
}

//------------------------------------------------------------------------

void OctreeRuntime::findSlices(
    Array<FindResult>&  results,
    FindMode            mode,
    int                 objectID,
    const Vec3f&        areaLo,
    const Vec3f&        areaHi,
    int                 maxLevels,
    int                 maxResults)
{
    FW_ASSERT(maxLevels >= 0);
    FW_ASSERT(maxResults >= 0);

    // Object not loaded => ignore.

    results.clear();
    if (!hasObject(objectID) || !maxResults)
        return;

    // Calculate temporary values.

    S32 scaleLimit = OctreeFile::UnitScale + 1 - maxLevels;
    Vec3f areaLoScaled = areaLo * exp2(OctreeFile::UnitScale);
    Vec3f areaHiScaled = areaHi * exp2(OctreeFile::UnitScale);

    // Traverse slices.

    Array<S32> stack(m_objects[objectID]->rootSliceID);
    while (stack.getSize())
    {
        int sliceID = stack.removeLast();
        const Slice* slice = m_slices[sliceID];
        if (slice->nodeScale < scaleLimit)
            continue;

        // Push children to the stack.

        for (int i = 0; i < slice->blocks.getSize(); i++)
            if (slice->blocks[i].childEntry >= 0)
                stack.add(slice->blocks[i].childEntry);

        // Not a valid result for the query => skip.

        bool valid;
        switch (mode)
        {
        case FindMode_Load:             valid = (!slice->isLoaded && !slice->isUnbuilt); break;
        case FindMode_Build:            valid = (!slice->isLoaded && slice->isUnbuilt); break;
        case FindMode_LoadOrBuild:      valid = (!slice->isLoaded); break;
        case FindMode_Unload:           valid = (slice->isLoaded && !slice->numChildrenLoaded); break;
        case FindMode_UnloadDeepest:    valid = (slice->isLoaded && !slice->numChildrenLoaded); break;
        default:                        FW_ASSERT(false); return;
        }
        if (!valid)
            continue;

        // Calculate temporary values.

        F32 cubeSize = exp2(slice->cubeScale);
        Vec3f cubeLo = Vec3f(slice->cubePos);
        Vec3f cubeHi = cubeLo + cubeSize;

        F32 dist = cubeSize * 0.5f + sqrt(
            sqr(max(areaLoScaled.x - cubeHi.x, cubeLo.x - areaHiScaled.x, 0.0f)) +
            sqr(max(areaLoScaled.y - cubeHi.y, cubeLo.y - areaHiScaled.y, 0.0f)) +
            sqr(max(areaLoScaled.z - cubeHi.z, cubeLo.z - areaHiScaled.z, 0.0f)));

        // Evaluate score.

        F32 score;
        switch (mode)
        {
        case FindMode_Load:             score = exp2(slice->nodeScale) / dist; break;
        case FindMode_Build:            score = exp2(slice->nodeScale) / dist; break;
        case FindMode_LoadOrBuild:      score = exp2(slice->nodeScale) / dist; break;
        case FindMode_Unload:           score = -exp2(slice->nodeScale) / dist; break;
        case FindMode_UnloadDeepest:    score = (F32)-slice->nodeScale; break;
        default:                        FW_ASSERT(false); return;
        }

        // Not good enough => skip.

        if (results.getSize() == maxResults && score <= results.getLast().score)
            continue;

        // Add result.

        int slot = results.getSize();
        results.add();
        while (slot > 0 && score > results[slot - 1].score)
        {
            results[slot] = results[slot - 1];
            slot--;
        }

        results[slot].sliceID = sliceID;
        results[slot].score = score;
        if (results.getSize() > maxResults)
            results.removeLast();
    }

    // Unload => negate scores.

    if (mode == FindMode_Unload || mode == FindMode_UnloadDeepest)
        for (int i = 0; i < results.getSize(); i++)
            results[i].score = -results[i].score;
}

//------------------------------------------------------------------------

OctreeRuntime::FindResult OctreeRuntime::findSlice(
    FindMode        mode,
    int             objectID,
    const Vec3f&    areaLo,
    const Vec3f&    areaHi,
    int             maxLevels)
{
    Array<FindResult> results;
    findSlices(results, mode, objectID, areaLo, areaHi, maxLevels, 1);
    if (results.getSize())
        return results[0];

    FindResult res;
    res.sliceID = -1;
    res.score = -FW_F32_MAX;
    return res;
}

//------------------------------------------------------------------------

String OctreeRuntime::getStats(void) const
{
    S64 total = m_mem.getTotalBytes();
    S64 used = total - m_mem.getFreeBytes();

    return sprintf("OctreeRuntime: slices %d, megs %.0f, used %.0f%%, bytes/voxel %.2f",
        m_numSlicesLoaded,
        (F64)used * exp2(-20),
        (F64)used / (F64)total * 100.0,
        (F64)used / (F64)max(m_numNodeChildrenLoaded, (S64)1));
}

//------------------------------------------------------------------------

OctreeRuntime::Block OctreeRuntime::allocateBlock(S64 size)
{
    FW_ASSERT(size >= 0);

    // Allocate.

    Block block;
    block.ofs = m_mem.alloc(size << 2, &m_reloc) >> 2;
    block.size = size;

    // No relocations => done.

    if (m_reloc.isEmpty())
        return block;

    // Fix trunk blocks and trunks.

    for (int i = 0; i < m_objects.getSize(); i++)
    {
        Object* obj = m_objects[i];
        if (!obj)
            continue;

        for (int j = 0; j < obj->trunkBlocks.getSize(); j++)
            obj->trunkBlocks[j].ofs += m_reloc.getDWordDelta(obj->trunkBlocks[j].ofs);

        for (int j = 0; j < obj->trunks.getSize(); j++)
        {
            Trunk& t = obj->trunks[j];
            S64 tdelta = m_reloc.getDWordDelta(t.blockOfs);
            t.blockOfs += tdelta;
            t.nodeOfs += tdelta;

            for (int k = 0; k < 8; k++)
            {
                S64 cdelta = m_reloc.getDWordDelta(t.childOfs[k]);
                t.childOfs[k] += cdelta;
                if (tdelta != cdelta)
                    updateTrunkNode(t, k);
            }
        }
    }

    // Fix blocks.

    for (int i = 0; i < m_slices.getSize(); i++)
    {
        Array<SliceBlock>& blocks = m_slices[i]->blocks;
        for (int j = 0; j < blocks.getSize(); j++)
            blocks[j].block.ofs += m_reloc.getDWordDelta(blocks[j].block.ofs);
    }

    for (int i = 0; i < m_loadBlocks.getSize(); i++)
        m_loadBlocks[i].block.ofs += m_reloc.getDWordDelta(m_loadBlocks[i].block.ofs);
    return block;
}

//------------------------------------------------------------------------

void OctreeRuntime::freeBlock(Block& block)
{
    if (block.ofs != -1)
        m_mem.free(block.ofs << 2, block.size << 2);

    block.ofs = -1;
    block.size = -1;
}

//------------------------------------------------------------------------

int OctreeRuntime::allocateTrunk(int objectID)
{
    Object* obj = m_objects[objectID];
    FW_ASSERT(obj);

    // There are free trunks => allocate one.

    if (obj->firstFreeTrunk != -1)
    {
        int trunkID = obj->firstFreeTrunk;
        obj->firstFreeTrunk = obj->trunks[trunkID].nextFreeTrunk;
        return trunkID;
    }

    // Allocate and upload the block.

    Block block = allocateBlock(obj->trunkBlockSize);
    if (block.ofs == -1)
        return -1;

    Array<S32> blockData(NULL, obj->trunkBlockSize);
    initTrunkBlock(blockData.getPtr(), objectID);
    upload(block.ofs, blockData.getPtr(), block.size);
    obj->trunkBlocks.add(block);

    // Create the trunks.

    int firstTrunkID = obj->trunks.getSize();
    obj->trunks.add(NULL, obj->pagesPerTrunkBlock * obj->trunksPerPage);

    for (int page = obj->pagesPerTrunkBlock - 1; page >= 0; page--)
    {
        for (int trunkIdx = obj->trunksPerPage - 1; trunkIdx >= 0; trunkIdx--)
        {
            int trunkID         = firstTrunkID + trunkIdx + page * obj->trunksPerPage;
            Trunk& trunk        = obj->trunks[trunkID];
            trunk.nextFreeTrunk = obj->firstFreeTrunk;
            obj->firstFreeTrunk = trunkID;

            trunk.blockOfs      = block.ofs;
            trunk.pageInBlock   = page;
            trunk.trunkInPage   = trunkIdx;
            trunk.nodeOfs       = block.ofs + page * PageSize + obj->nodeAlign + trunkIdx * 32;

            for (int i = 0; i < 8; i++)
            {
                trunk.validMask[i] = 0x00;
                trunk.nonLeafMask[i] = 0x00;
            }
        }
    }

    // Allocate a trunk now that we are guaranteed to have free ones.

    int trunkID = obj->firstFreeTrunk;
    obj->firstFreeTrunk = obj->trunks[trunkID].nextFreeTrunk;
    return trunkID;
}

//------------------------------------------------------------------------

void OctreeRuntime::freeTrunk(int objectID, int trunkID)
{
    Object* obj = m_objects[objectID];
    FW_ASSERT(obj);

    obj->trunks[trunkID].nextFreeTrunk = obj->firstFreeTrunk;
    obj->firstFreeTrunk = trunkID;
}

//------------------------------------------------------------------------

void OctreeRuntime::initTrunkBlock(S32* blockData, int objectID)
{
    const Object* obj = m_objects[objectID];
    FW_ASSERT(blockData && obj);
    memset(blockData, 0, obj->trunkBlockSize * sizeof(S32));

    blockData[obj->trunkBlockInfoOfs + BlockInfo_SliceID]       = -1;
    blockData[obj->trunkBlockInfoOfs + BlockInfo_IndexInSlice]  = -1;
    blockData[obj->trunkBlockInfoOfs + BlockInfo_BlockPtr]      = -obj->trunkBlockInfoOfs;
    blockData[obj->trunkBlockInfoOfs + BlockInfo_NumAttach]     = 0;

    for (int i = 0; i < obj->pagesPerTrunkBlock; i++)
        blockData[i * PageSize] = obj->trunkBlockInfoOfs - i * PageSize;

    obj->attachIO->initTrunkBlock(blockData);
}

//------------------------------------------------------------------------

void OctreeRuntime::updateTrunkNode(const Trunk& trunk, int sub)
{
    S64 subOfs = trunk.nodeOfs + sub * 2;
    S32 childRel = (S32)(trunk.childOfs[sub] - subOfs);
    FW_ASSERT((childRel & 1) == 0);

    if ((childRel & ~0xFFFE) != 0)
    {
        upload(subOfs + 16, childRel >> 1);
        childRel = 0x11;
    }
    upload(subOfs, (childRel << 16) | (trunk.validMask[sub] << 8) | trunk.nonLeafMask[sub]);
}

//------------------------------------------------------------------------

void OctreeRuntime::unloadSliceInternal(int sliceID)
{
    Slice* slice = m_slices[sliceID];
    Object* obj = m_objects[slice->objectID];
    FW_ASSERT(slice->isLoaded);
    FW_ASSERT(!slice->numChildrenLoaded);
    FW_ASSERT(obj);

    // Allocate temporary buffer to store the old data.

    Array<S32> blockOfs(NULL, slice->blocks.getSize());
    int numTrunkBlocks = slice->blocks.getSize() / (obj->trunksPerPage * obj->pagesPerTrunkBlock) + 1;
    int trunkBlockSize = (obj->trunkBlockSize + PageSize - 1) & -PageSize;
    int bufferSize = numTrunkBlocks * trunkBlockSize;

    for (int i = 0; i < slice->blocks.getSize(); i++)
    {
        blockOfs[i] = bufferSize;
        if (slice->blocks[i].block.ofs != -1)
            bufferSize += ((S32)slice->blocks[i].block.size + PageSize - 1) & -PageSize;
    }

    Buffer buffer(NULL, bufferSize * sizeof(S32), Buffer::Hint_None, PageBytes);
    S32* bufferData = (S32*)buffer.getMutablePtr();

    // Initialize trunk blocks, and download/free leaf blocks.

    for (int i = 0; i < numTrunkBlocks; i++)
        initTrunkBlock(bufferData + i * trunkBlockSize, slice->objectID);

    for (int i = 0; i < slice->blocks.getSize(); i++)
    {
        Block& block = slice->blocks[i].block;
        if (block.ofs == -1)
            continue;

        buffer.setRange(blockOfs[i] * sizeof(S32), m_mem.getBuffer(), block.ofs * sizeof(S32), block.size * sizeof(S32));
        freeBlock(block);
    }

    // Copy root subtrunk to the buffer.

    const Trunk& trunk = obj->trunks[slice->trunkID];
    bufferData[slice->subtrunk * 2 + obj->nodeAlign] = 0x110000 | (trunk.validMask[slice->subtrunk] << 8) | trunk.nonLeafMask[slice->subtrunk];
    obj->attachIO->copyTrunkToTrunk(buffer, 0, obj->nodeAlign, 0, 0,
        m_mem.getBuffer(), trunk.blockOfs, trunk.nodeOfs, trunk.pageInBlock, trunk.trunkInPage);

    // Copy split trunks to the buffer.

    Array<S32> stack(slice->subtrunk * 2 + obj->nodeAlign);
    int currBlock = 0;
    int currPage = 0;
    int currTrunk = 0;

    for (int i = 0; i < slice->blocks.getSize(); i++)
    {
        const SliceBlock& sb = slice->blocks[i];
        int dstSubOfs = stack.removeLast();

        // Set child pointer assuming leaf block.
        // No split trunk => done.

        bufferData[dstSubOfs + 16] = (blockOfs[i] + obj->nodeAlign * 2 - dstSubOfs) >> 1;
        if (sb.splitTrunkID == -1)
            continue;

        // Allocate trunk and set child pointer.

        currTrunk++;
        if (currTrunk == obj->trunksPerPage)
        {
            currTrunk = 0;
            currPage++;
            if (currPage == obj->pagesPerTrunkBlock)
            {
                currPage = 0;
                currBlock++;
            }
        }

        int dstBlockOfs = currBlock * trunkBlockSize;
        int dstTrunkOfs = dstBlockOfs + currPage * PageSize + currTrunk * 32 + obj->nodeAlign;
        bufferData[dstSubOfs + 16] = (dstTrunkOfs - dstSubOfs) >> 1;

        // Copy and free the trunk.

        const Trunk& split = obj->trunks[sb.splitTrunkID];
        for (int j = 0; j < 8; j++)
            bufferData[dstTrunkOfs + j * 2] = 0x110000 | (split.validMask[j] << 8) | split.nonLeafMask[j];
        obj->attachIO->copyTrunkToTrunk(buffer, dstBlockOfs, dstTrunkOfs, currPage, currTrunk,
            m_mem.getBuffer(), split.blockOfs, split.nodeOfs, split.pageInBlock, split.trunkInPage);
        freeTrunk(slice->objectID, sb.splitTrunkID);

        // Push children to the stack.

        for (int j = 7; j >= 0; j--)
            stack.add(dstTrunkOfs + popc8(bufferData[dstSubOfs] & ((1 << j) - 1)) * 2);
    }

    // Build the new block.

    if (slice->parentSliceID != -1)
    {
        Block& block = m_slices[slice->parentSliceID]->blocks[slice->indexInParent].block;
        gatherImportNodes(NULL, NULL, NULL, bufferData + slice->subtrunk * 2 + obj->nodeAlign, slice->cubeScale - slice->nodeScale);
        int blockInfoOfs = layoutImportNodes(slice->objectID);
        obj->attachIO->beginSliceImport(NULL);
        buildBlock(m_unloadBlock, blockInfoOfs, slice->parentSliceID, slice->indexInParent, slice->objectID);

        // Allocate.

        block = allocateBlock(m_unloadBlock.getSize());
        if (block.ofs == -1)
            fail("OctreeRuntime::unloadSlice() ran out of memory!"); // should not happen

        // Update trunk node.

        Trunk& trunk = obj->trunks[slice->trunkID];
        trunk.nonLeafMask[slice->subtrunk] = m_importNodes[0].nonLeafMask;
        trunk.childOfs[slice->subtrunk] = block.ofs + obj->nodeAlign * 2;
        updateTrunkNode(trunk, slice->subtrunk);

        // Upload.

        upload(block.ofs, m_unloadBlock.getPtr(), block.size);
    }

    // Tear down child slices.

    for (int i = 0; i < slice->blocks.getSize(); i++)
    {
        S32 cid = slice->blocks[i].childEntry;
        if (cid < 0)
            continue;

        m_slices[cid]->isReached = false;
        m_slices[cid]->isUnbuilt = false;
    }
    slice->blocks.clear();

    // Update rest of the state.

    slice->isLoaded = false;
    if (slice->parentSliceID != -1)
        m_slices[slice->parentSliceID]->numChildrenLoaded--;

    m_numSlicesLoaded--;
    m_numNodesLoaded -= slice->numNodes;
    m_numNodeChildrenLoaded -= slice->numNodeChildren;
}

//------------------------------------------------------------------------

OctreeRuntime::Slice* OctreeRuntime::getOrCreateSlice(S32 sliceID)
{
    FW_ASSERT(sliceID >= 0);
    while (sliceID >= m_slices.getSize())
    {
        Slice* slice = m_slices.add(new Slice);

        slice->isReached     = false;
        slice->isUnbuilt     = false;
        slice->isLoaded      = false;
        slice->numChildrenLoaded = 0;

        slice->objectID      = -1;
        slice->parentSliceID = -1;
        slice->indexInParent = 0;
        slice->trunkID       = -1;
        slice->subtrunk      = -1;

        slice->cubePos       = Vec3i(0);
        slice->cubeScale     = OctreeFile::UnitScale;
        slice->nodeScale     = OctreeFile::UnitScale;
    }
    return m_slices[sliceID];
}

//------------------------------------------------------------------------

S64 OctreeRuntime::getRootNodeOfs(int objectID)
{
    if (!hasObject(objectID))
        return -1;

    const Object* obj = m_objects[objectID];
    const Slice* slice = m_slices[obj->rootSliceID];
    if (!slice->isLoaded || slice->parentSliceID != -1)
        return -1;

    return obj->trunks[slice->trunkID].nodeOfs + slice->subtrunk * 2;
}

//------------------------------------------------------------------------

void OctreeRuntime::gatherImportNodes(
    const OctreeSlice*  sliceData,
    int*                sliceNodeIdx,
    int*                sliceSplitNodeIdx,
    const S32*          oldRootNode,
    int                 numLevels)
{
    struct StackEntry
    {
        S32         level;
        const S32*  oldNode;
        S32         importNodeIdx;   // -1 if none
    };

    // Initialize traversal.

    m_importNodes.clear();

    Array<StackEntry> stack(NULL, 1);
    stack[0].level          = -1;
    stack[0].oldNode        = NULL;
    stack[0].importNodeIdx  = -1;

    // Traverse nodes in the old tree.

    while (stack.getSize())
    {
        StackEntry curr = stack.removeLast();
        U32 oldNodeData = (curr.oldNode) ? *curr.oldNode : 0x0101;
        int firstChild  = m_importNodes.getSize();
        U32 nonLeafMask = 0;

        // Leaf in loadSlice() => get children from the slice.

        if (sliceData && curr.level == numLevels - 2)
        {
            for (int i = 0; i < 8; i++)
            {
                if ((oldNodeData & (0x0100 << i)) == 0)
                    continue;

                if (!sliceData->isNodeSplit((*sliceNodeIdx)++))
                    continue;

                AttachIO::ImportNode& child = m_importNodes.add();
                child.srcInRuntime = NULL;
                child.srcInSlice   = (*sliceSplitNodeIdx)++;
                child.validMask    = sliceData->getNodeValidMask(child.srcInSlice);
                child.nonLeafMask  = 0x00;
                child.firstChild   = m_importNodes.getSize() - 1;
                nonLeafMask |= 1 << i;
            }
        }

        // Otherwise, unless leaf in unloadSlice() => get children from the old tree.

        else if (curr.level < numLevels - 1)
        {
            nonLeafMask = oldNodeData & 0xFF;
            int ofs = (oldNodeData >> 17) * 2;
            const S32* oldChildren = oldRootNode;
            if (curr.oldNode)
                oldChildren = curr.oldNode + (((oldNodeData & 0x10000) == 0) ? ofs : curr.oldNode[ofs] * 2);

            for (int i = 7; i >= 0; i--)
            {
                U32 cmask = 1 << i;
                if ((nonLeafMask & cmask) == 0)
                    continue;

                StackEntry& top     = stack.add();
                int childIdx        = popc8(nonLeafMask & (cmask - 1));
                top.level           = curr.level + 1;
                top.oldNode         = oldChildren + childIdx * 2;
                top.importNodeIdx   = firstChild + childIdx;
                m_importNodes.add();
            }
        }

        // Fill in ImportNode.

        if (curr.importNodeIdx != -1)
        {
            AttachIO::ImportNode& node = m_importNodes[curr.importNodeIdx];
            node.srcInRuntime = curr.oldNode;
            node.srcInSlice   = -1;
            node.validMask    = (U8)(oldNodeData >> 8);
            node.nonLeafMask  = (U8)nonLeafMask;
            node.firstChild   = (nonLeafMask) ? firstChild : curr.importNodeIdx;
        }

        // Fill in numParentChildren and idxInParent.

        if (nonLeafMask)
        {
            int num = popc8(nonLeafMask);
            for (int i = 0; i < num; i++)
            {
                AttachIO::ImportNode& node = m_importNodes[firstChild + i];
                node.numParentChildren = (U8)num;
                node.idxInParent = (U8)i;
            }
        }
    }
}

//------------------------------------------------------------------------

int OctreeRuntime::layoutImportNodes(int objectID)
{
    // Determine node offsets, ignoring PageHeaders and
    // assuming that FarPtrs follow Nodes immediately.

    int numNodes = m_importNodes.getSize();
    int align = m_objects[objectID]->nodeAlign;

    FW_ASSERT(m_importNodes.getSize());
    m_importNodes.add();
    m_importNodes[numNodes].ofsInBlock = 0;

    for (int i = numNodes - 1; i >= 0; i--)
    {
        AttachIO::ImportNode& node = m_importNodes[i];
        node.ofsInBlock = m_importNodes[i + 1].ofsInBlock; // to account for nonLeafMask=0x00
        if (node.idxInParent == node.numParentChildren - 1)
            node.ofsInBlock -= align - 2; // to account for possible alignment

        // Approximate distance between the node and its first child.

        int diff = m_importNodes[node.firstChild].ofsInBlock - node.ofsInBlock;

        // Account for the difference between currOfs and the final offset.

        diff += 18; // 8 FarPtrs and the Node itself

        // Account for possible PageHeaders and PagePaddings.
        //    ceil(diff / (PageSize - 32)) * 32
        // <= (ceil(diff / PageSize) + ceil(diff * 64 / PageSize^2)) * 32

        int t0 = (diff + (1 << PageSizeLog2) - 1) >> PageSizeLog2;
        int t1 = (diff + (1 << (PageSizeLog2 * 2 - 6)) - 1) >> (PageSizeLog2 * 2 - 6);
        diff += (t0 + t1) * 32;

        // Update the offset.

        node.ofsInBlock -= (diff <= 0xFFFF) ? 2 : 4;
    }

    // Determine final node offsets for each group of children.

    int currNodeOfs = 0;
    for (int i = 0; i < numNodes;)
    {
        int num = m_importNodes[i].numParentChildren;
        int needed = m_importNodes[i + num].ofsInBlock - m_importNodes[i].ofsInBlock - (align - 2); // Nodes and FarPtrs

        currNodeOfs = max(currNodeOfs, ((currNodeOfs + needed - 1) & -PageSize) + align);
        for (int j = 0; j < num; j++)
            m_importNodes[i + j].ofsInBlock = currNodeOfs + j * 2;

        currNodeOfs = (currNodeOfs + needed + align - 1) & -align;
        i += num;
    }

    m_importNodes.removeLast();
    FW_ASSERT(m_importNodes[0].ofsInBlock == align);
    FW_ASSERT(m_importNodes.getSize() < 2 || m_importNodes[1].ofsInBlock == align * 2);
    return currNodeOfs;
}

//------------------------------------------------------------------------

void OctreeRuntime::buildBlock(
    Array<S32>&         blockData,
    int                 blockInfoOfs,
    int                 sliceID,
    int                 indexInSlice,
    int                 objectID)
{
    // Fill in PageHeaders.

    blockData.resize(blockInfoOfs + BlockInfo_End);
    S32* ptr = blockData.getPtr();
    for (int i = 0; i < blockInfoOfs; i += PageSize)
        ptr[i] = blockInfoOfs - i;

    // Fill in Nodes and FarPtrs.

    int numNodes = m_importNodes.getSize();
    for (int i = 0; i < numNodes;)
    {
        int num = m_importNodes[i].numParentChildren;
        int nextI = i + num;
        int farPtrOfs = m_importNodes[i].ofsInBlock + num * 2;

        for (int j = i; j < nextI; j++)
        {
            const AttachIO::ImportNode& node = m_importNodes[j];
            int childPtr = m_importNodes[node.firstChild].ofsInBlock - node.ofsInBlock;
            FW_ASSERT(childPtr >= 0 && (childPtr & 1) == 0);

            if ((childPtr & ~0xFFFE) != 0)
            {
                ptr[farPtrOfs] = childPtr >> 1;
                childPtr = (farPtrOfs - node.ofsInBlock) | 1;
                farPtrOfs += 2;
            }
            ptr[node.ofsInBlock] = (childPtr << 16) | (node.validMask << 8) | node.nonLeafMask;
        }
        i = nextI;
    }

    // Fill in BlockInfo.

    ptr[blockInfoOfs + BlockInfo_SliceID]      = sliceID;
    ptr[blockInfoOfs + BlockInfo_IndexInSlice] = indexInSlice;
    ptr[blockInfoOfs + BlockInfo_BlockPtr]     = -blockInfoOfs;
    ptr[blockInfoOfs + BlockInfo_NumAttach]    = 0;

    // Import in attachments.

    m_objects[objectID]->attachIO->importNodes(blockData, m_importNodes);
}

//------------------------------------------------------------------------
