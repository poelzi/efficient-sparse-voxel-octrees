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

#include "BuilderBase.hpp"
#include "base/Timer.hpp"

using namespace FW;

//------------------------------------------------------------------------

void BuilderBase::ChildSlice::init(int idx, const Vec3i& cubePos, int cubeScale, int nodeScale, bool isSplit)
{
    this->idx       = idx;
    this->cubePos   = cubePos;
    this->cubeScale = cubeScale;
    this->nodeScale = nodeScale;
    this->isSplit   = isSplit;

    nodes.clear();
    buildData.clear();
    bitWriter = BitWriter(&buildData);
}

//------------------------------------------------------------------------

BuilderBase::ThreadState::ThreadState(void)
:   m_attachIO      (NULL),
    m_workIn        (0.0f),
    m_workOut       (0.0f)
{
}

//------------------------------------------------------------------------

BuilderBase::ThreadState::~ThreadState(void)
{
    delete m_attachIO;
}

//------------------------------------------------------------------------

void BuilderBase::ThreadState::runTask(Task& task)
{
    // Initialize state.

    FW_ASSERT(task.slice && task.buildData);
    OctreeSlice& slice = *task.slice;
    m_bitReader = BitReader(task.buildData);
    m_workIn = 0.0f;
    m_workOut = 0.0f;

    // Initialize AttachIO.

    if (!m_attachIO || m_attachIO->getRuntimeTypes() != task.attachTypes)
    {
        delete m_attachIO;
        m_attachIO = new AttachIO(task.attachTypes);
    }
    int maxSliceAttachments = m_attachIO->beginSliceExport();

    // Read parent slice and init child slices.

    initChildSlices(slice);

    pushMemOwner("Builder subclass state");
    beginParentSlice(slice.getCubePos(), slice.getCubeScale(), slice.getNodeScale(), task.objectID, task.numNodes);
    for (int i = 0; i < m_childSlices.getSize(); i++)
        if (!m_childSlices[i].isSplit)
            beginChildSlice(m_childSlices[i]);
    popMemOwner();

    // Process each parent node.

    int currChildSliceIdx = 0;
    m_nodeSplit.clear();
    m_nodeValidMask.clear();

    for (int nodeIdx = 0; nodeIdx < task.numNodes; nodeIdx++)
    {
        // Initialize parent node.

        m_attachIO->beginNodeExport();
        U8 validMask = 0x00;
        int shift = nodeIdx & 0x1F;
        if (!shift)
            m_nodeSplit.add(0);

        // Read parent node.

        pushMemOwner("Builder subclass state");
        Vec3i nodePos = readParentNode(slice.getCubePos(), slice.getCubeScale(), slice.getNodeScale());
        popMemOwner();

        // Not split?

        if (nodePos.x < 0)
        {
            m_attachIO->endNodeExport(true);
            continue;
        }

        // Find child slice.

        for (;;)
        {
            ChildSlice& cs = m_childSlices[currChildSliceIdx];
            if (!cs.isSplit && isPosInCube(nodePos, cs.cubePos, cs.cubeScale))
                break;
            currChildSliceIdx++;
        }
        ChildSlice& cs = m_childSlices[currChildSliceIdx];

        // Build child nodes.

        for (int childIdx = 0; childIdx < 8; childIdx++)
        {
            Vec3i childPos = getCubeChildPos(nodePos, slice.getNodeScale(), childIdx);
            pushMemOwner("Builder subclass state");
            bool exists = buildChildNode(cs, childPos);
            popMemOwner();
            if (!exists)
                continue;

            validMask |= 1 << childIdx;
            cs.nodes.add(childPos);
        }

        // Write parent node.

        m_nodeSplit.getLast() |= 1 << shift;
        m_nodeValidMask.add(validMask);
        m_attachIO->endNodeExport(false);
    }

    // Reconstruct complete parent slice.

    int sliceID = slice.getID();
    slice.init(m_childSlices.getSize(), maxSliceAttachments, task.numNodes, m_nodeValidMask.getSize());
    slice.setID(sliceID);
    slice.setState(OctreeFile::SliceState_Complete);
    slice.setCubePos(m_childSlices[0].cubePos);
    slice.setCubeScale(m_childSlices[0].cubeScale);
    slice.setNodeScale(m_childSlices[0].nodeScale + 1);
    memcpy(slice.getNodeSplitPtr(), m_nodeSplit.getPtr(), m_nodeSplit.getNumBytes());
    memcpy(slice.getNodeValidMaskPtr(), m_nodeValidMask.getPtr(), m_nodeValidMask.getNumBytes());

    // Construct unbuilt child slices.

    task.children.reset(m_childSlices.getSize());
    for (int i = 0; i < m_childSlices.getSize(); i++)
    {
        ChildSlice& cs = m_childSlices[i];
        if (cs.isSplit)
        {
            slice.setChildEntry(i, OctreeSlice::ChildEntry_Split);
            continue;
        }

        pushMemOwner("Builder subclass state");
        endChildSlice(cs);
        popMemOwner();
        if (!cs.nodes.getSize() || cs.nodeScale < 1)
            continue;

        OctreeSlice& childData = task.children[i];
        Array<S32> childEntries;
        selectSliceSplits(childEntries, cs);

        childData.init(childEntries.getSize(), 1, 0, 0);
        childData.setCubePos(cs.cubePos);
        childData.setCubeScale(cs.cubeScale);
        childData.setNodeScale(cs.nodeScale);
        memcpy(childData.getChildEntryPtr(), childEntries.getPtr(), childEntries.getNumBytes());

        childData.startAttach(AttachIO::BuildDataAttach);
        BitWriter bitWriter(&childData.getData());

        bitWriter.write(32, 1); // version
        bitWriter.write(32, task.objectID); // objectID
        bitWriter.write(32, cs.nodes.getSize()); // numNodes
        for (int j = 0; j < 8; j++)
            bitWriter.write(8, task.idString[j]); // subclassIDString[j]

        bitWriter.write(31, 0); // padding
        cs.writeBits(31, 0); // padding
        childData.getData().add(cs.buildData);
        childData.endAttach();
    }

    // Finish up.

    pushMemOwner("Builder subclass state");
    endParentSlice();
    popMemOwner();

    m_attachIO->endSliceExport(slice);

    task.workIn = m_workIn;
    task.workOut = m_workOut;
}

//------------------------------------------------------------------------

void BuilderBase::ThreadState::initChildSlices(const OctreeSlice& slice)
{
    m_childSlices.resize(slice.getNumChildEntries());
    Array<Vec4i> stack(Vec4i(slice.getCubePos(), slice.getCubeScale()));

    for (int i = 0; i < m_childSlices.getSize(); i++)
    {
        Vec4i cube = stack.removeLast();
        m_childSlices[i].init(
            i, cube.getXYZ(), cube.w, slice.getNodeScale() - 1,
            (slice.getChildEntry(i) == OctreeSlice::ChildEntry_Split));

        ChildSlice& cs = m_childSlices[i];
        if (cs.isSplit)
            for (int j = 7; j >= 0; j--)
                stack.add(Vec4i(getCubeChildPos(cube.getXYZ(), cube.w, j), cube.w - 1));
    }

    FW_ASSERT(!stack.getSize());
}

//------------------------------------------------------------------------

void BuilderBase::ThreadState::selectSliceSplits(Array<S32>& childEntries, const ChildSlice& cs)
{
    childEntries.clear();
    Array<Vec4i> stack(Vec4i(cs.cubePos, cs.cubeScale));

    while (stack.getSize())
    {
        Vec4i cube = stack.removeLast();
        bool split = false;

        // Count nodes in the cube and its children.

        int nodesInCube = 0;
        int nodesInChild[8];
        for (int i = 0; i < 8; i++)
            nodesInChild[i] = 0;

        for (int i = 0; i < cs.nodes.getSize(); i++)
        {
            if (!isPosInCube(cs.nodes[i], cube.getXYZ(), cube.w))
                continue;

            nodesInCube++;
            nodesInChild[getCubeChildIndex(cs.nodes[i], cube.w)]++;
        }

        // Split based on AvgNodesPerSlice.

        int numChildren = 0;
        for (int i = 0; i < 8; i++)
            if (nodesInChild[i])
                numChildren++;

        F32 n = (F32)numChildren;
        if (n < 2.0f || (F32)nodesInCube * 4.0f > (F32)AvgNodesPerSlice * n * log(n) / (n - 1.0f))
            split = true;

        // Split based on MaxNodesPerBlock.

        int nodesInLevel = 1;
        int nodesInBlock = 1;
        for (int i = cube.w - 1; i > cs.nodeScale; i--)
        {
            nodesInLevel = min(nodesInLevel * 8, AvgNodesPerSlice * 16 / 9 + 1);
            nodesInBlock += nodesInLevel;
        }

        nodesInBlock += nodesInCube; // cs.nodeScale
        nodesInBlock += nodesInCube * 8; // cs.nodeScale - 1
        if (nodesInBlock > MaxNodesPerBlock)
            split = true;

        // Split large nodes to increase parallelism.

        if (cube.w > OctreeFile::UnitScale - ForceSplitLevels)
            split = true;

        // Empty slice or too flat => do not split.

        if (!nodesInCube || cube.w - cs.nodeScale <= 1)
            split = false;

        // Not split => done.

        if (!split)
        {
            childEntries.add(OctreeSlice::ChildEntry_NoChild);
            continue;
        }

        // Split => push child cubes to the stack.

        childEntries.add(OctreeSlice::ChildEntry_Split);
        for (int i = 7; i >= 0; i--)
            stack.add(Vec4i(getCubeChildPos(cube.getXYZ(), cube.w, i), cube.w - 1));
    }
}

//------------------------------------------------------------------------

BuilderBase::BuilderBase(OctreeFile* file)
:   m_file          (file),
    m_maxThreads    (FW_S32_MAX),
    m_serialState   (NULL),
    m_numRunning    (0),
    m_abort         (false)
{
    FW_ASSERT(file);
}

//------------------------------------------------------------------------

BuilderBase::~BuilderBase(void)
{
    asyncAbort();
}

//------------------------------------------------------------------------

void BuilderBase::buildObject(int objectID, int numLevels, const Params& params, bool enablePrints)
{
    struct QueueEntry
    {
        S32 sliceID;
        F32 timeTotal;  // Before building the slice.
        F32 workTotal;  // Before building the slice.
    };

    FW_ASSERT(numLevels >= 0);

    // Create root slice.

    OctreeSlice rootSlice;
    if (!createRootSlice(rootSlice, objectID, params))
    {
        if (enablePrints)
            printf("%s: Tried to build a non-existent object!\n", getClassName().getPtr());
        return;
    }

    // No levels requested => done.

    if (!numLevels)
        return;

    // Print table header.

    if (enablePrints)
    {
        printf("%s: %-6s%-12s%-14s%-14s%-10s\n",
            getClassName().getPtr(), "Level", "Slices", "Elapsed", "Remaining", "Progress");
        printf("%s: %-6s%-12s%-14s%-14s%-10s\n",
            getClassName().getPtr(), "---", "---", "---", "---", "---");
    }

    // Initialize state.

    int     level       = 0;
    int     levelStart  = 0;
    int     levelEnd    = 1;
    Timer   timeLevel   (true);
    F32     workLevel   = 1.0f; // Total work on current level.
    F32     workDone    = 0.0f; // Work done on current level.
    F32     workNext    = 0.0f; // Total work on next level.
    Timer   timeTotal   (true);
    F32     workTotal   = 0.0f;

    Array<QueueEntry> queue;
    queue.add().sliceID = rootSlice.getID();

    // Build slices.

    for (int queueIdx = 0; queueIdx < queue.getSize(); queueIdx++)
    {
        F32 timeStart = timeTotal.getElapsed();
        queue[queueIdx].timeTotal = timeStart;
        queue[queueIdx].workTotal = workTotal;

        // Print level progress.

        if (enablePrints)
        {
            S32 sliceWindow = MaxAsyncBuildSlices * 2;
            F32 timeWindow = 30.0f;
            int idx = max(queueIdx - sliceWindow, 0);
            while (idx > 0 && queue[idx - 1].timeTotal >= timeStart - timeWindow)
                idx--;

            F32 remaining = -1.0f;
            if (queueIdx - idx >= sliceWindow)
            {
                F32 timePerWork = (timeStart - queue[idx].timeTotal) / (workTotal - queue[idx].workTotal);
                remaining = (workLevel - workDone) * timePerWork;
            }

            printf("%s: %-6s%-12s%-14s%-14s%-10s\r",
                getClassName().getPtr(),
                sprintf("%d/%d", level + 1, numLevels).getPtr(),
                sprintf("%d/%d", queueIdx - levelStart, levelEnd - levelStart).getPtr(),
                formatTime(timeLevel.getElapsed()).getPtr(),
                (remaining >= 0.0f) ? formatTime(remaining).getPtr() : "???",
                sprintf("%.0f%%", workDone / workLevel * 100.0f).getPtr());
        }

        // Start async reads and builds.

        S32 bytesTotal = 0;
        int endIdx = min(queueIdx + MaxPrefetchSlices, queue.getSize());

        for (int i = queueIdx; i < endIdx; i++)
        {
            int sliceID = queue[i].sliceID;
            int size = m_file->getSliceSize(sliceID);
            bytesTotal += size;
            if (bytesTotal > MaxPrefetchBytesTotal)
                break;

            if (!m_file->readSliceIsReady(sliceID))
                m_file->readSlicePrefetch(sliceID);
            else if (!asyncIsPending(sliceID) && asyncGetNumPending() < MaxAsyncBuildSlices)
            {
                OctreeSlice* slice = new OctreeSlice;
                m_file->readSlice(sliceID, *slice);
                asyncBuildSlice(slice);
            }
        }

        // Force-start the build of the current slice.

        int sliceID = queue[queueIdx].sliceID;
        if (!asyncIsPending(sliceID))
        {
            OctreeSlice* slice = new OctreeSlice;
            m_file->readSlice(sliceID, *slice);
            asyncBuildSlice(slice);
        }

        // Finish async build.

        F32 workIn, workOut;
        OctreeSlice* slice = asyncFinishSlice(true, sliceID, &workIn, &workOut);
        FW_ASSERT(slice);

        workDone += workIn;
        workNext += workOut;
        workTotal += workIn;

        if (level < numLevels - 1)
            for (int i = 0; i < slice->getNumChildEntries(); i++)
                if (slice->getChildEntry(i) >= 0)
                    queue.add().sliceID = slice->getChildEntry(i);

        delete slice;

        // Slices left on the current level => skip.

        if (queueIdx + 1 != levelEnd)
            continue;

        // Print level totals.

        if (enablePrints)
            printf("%s: %-6d%-12d%-14s%-14s%-10s\n",
                getClassName().getPtr(),
                level + 1,
                levelEnd - levelStart,
                formatTime(timeLevel.getElapsed()).getPtr(),
                "-",
                "100%");

        // Move to the next level.

        level++;
        levelStart  = levelEnd;
        levelEnd    = queue.getSize();
        workLevel   = workNext;
        workDone    = 0.0f;
        workNext    = 0.0f;
        timeLevel.start();
    }

    // Print totals.

    if (enablePrints)
    {
        printf("%s: %-6s%-12s%-14s%-14s%-10s\n",
            getClassName().getPtr(), "---", "---", "---", "---", "---");
        printf("%s: %-6s%-12d%-14s%-14s%-10s\n",
            getClassName().getPtr(),
            "Total",
            queue.getSize(),
            formatTime(timeTotal.getElapsed()).getPtr(),
            "-",
            "100%");
    }
}

//------------------------------------------------------------------------

bool BuilderBase::buildSlice(OctreeSlice* slice, F32* workIn, F32* workOut)
{
    if (!slice || !slice->getSize())
        return false;

    int sliceID = slice->getID();
    if (!asyncBuildSlice(slice))
        return false;

    asyncFinishSlice(true, sliceID, workIn, workOut);
    return true;
}

//------------------------------------------------------------------------

bool BuilderBase::asyncBuildSlice(OctreeSlice* slice)
{
    // No data => ignore.

    if (!slice || !slice->getSize())
        return false;

    // Already built or building => ignore.

    if (slice->getState() != OctreeFile::SliceState_Unbuilt || m_tasks.contains(slice->getID()))
        return false;

    // Find build data.
    // Not found => ignore.

    const S32* buildData = NULL;
    for (int i = 0; i < slice->getNumAttach(); i++)
        if (slice->getAttachType(i) == AttachIO::BuildDataAttach)
            buildData = slice->getData().getPtr(slice->getAttachOfs(i));

    if (!buildData)
        return false;

    // Read build data header.

    if (*buildData++ != 1) // version
        fail("BuilderBase: Unsupported build data version!");

    int objectID = *buildData++; // objectID
    if (objectID < 0 || objectID >= m_file->getNumObjects())
        return false;

    int numNodes = *buildData++; // numNodes
    if (numNodes <= 0)
        return false;

    String idString = getIDString();
    FW_ASSERT(idString.getLength() == 8);
    const char* idBytes = (const char*)buildData;
    for (int i = 0; i < 8; i++)
        if (idBytes[i] != idString[i]) // subclassIDString[i]
            return false;
    buildData += 2;

    // First time => setup threads.

    if (!m_serialState && !m_threads.getSize())
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        S32 numThreads = clamp((S32)si.dwNumberOfProcessors, 1, m_maxThreads);
        if (!supportsConcurrency())
            numThreads = 1;

        if (numThreads == 1)
            m_serialState = createThreadState(0);
        else
        {
            m_numRunning = numThreads;
            m_threads.reset(numThreads);
            for (int i = 0; i < numThreads; i++)
            {
                ThreadEntry& t  = m_threads[i];
                t.builder       = this;
                t.state         = createThreadState(i);
                t.thread        = new Thread;
                t.thread->start(threadFunc, &t);
            }
        }
    }

    // Create task.

    Task* task          = new Task;
    task->slice         = slice;
    task->idString      = idString;
    task->buildData     = buildData;
    task->objectID      = objectID;
    task->numNodes      = numNodes;
    task->attachTypes   = m_file->getObject(task->objectID).runtimeAttachTypes;

    prepareTask(*task);
    m_tasks.add(slice->getID(), task);

    // Queue task.

    if (m_serialState)
    {
        m_serialState->runTask(*task);
        m_finishedTasks.add(task);
    }
    else
    {
        m_monitor.enter();
        m_pendingTasks.add(task);
        m_monitor.notifyAll();
        m_monitor.leave();
    }
    return true;
}

//------------------------------------------------------------------------

OctreeSlice* BuilderBase::asyncFinishSlice(bool wait, int sliceID, F32* workIn, F32* workOut)
{
    // Nothing to finish => ignore.

    FW_ASSERT(sliceID >= -1);
    if ((sliceID == -1) ? !m_tasks.getSize() : !m_tasks.contains(sliceID))
        return NULL;

    // Get finished task.

    Task* task = NULL;
    m_monitor.enter();
    for (;;)
    {
        for (int i = 0; i < m_finishedTasks.getSize() && !task; i++)
            if (sliceID == -1 || m_finishedTasks[i]->slice->getID() == sliceID)
                task = m_finishedTasks.remove(i);
        if (task || !wait)
            break;
        m_monitor.wait();
    }
    m_monitor.leave();

    // No task => done.

    if (!task)
        return NULL;

    // Write resulting slices to the file.

    OctreeSlice* slice = task->slice;
    for (int i = 0; i < task->children.getSize(); i++)
    {
        OctreeSlice& child = task->children[i];
        if (!child.getSize())
            continue;

        child.setID(m_file->getFreeSliceID());
        child.setState(OctreeFile::SliceState_Unbuilt);
        slice->setChildEntry(i, child.getID());
        m_file->writeSlice(child);
    }

    m_file->writeSlice(*slice);

    // Output the amount of work.

    if (workIn)
        *workIn = task->workIn;
    if (workOut)
        *workOut = task->workOut;

    // Remove task.

    m_tasks.remove(slice->getID());
    delete task;
    return slice;
}

//------------------------------------------------------------------------

void BuilderBase::asyncAbort(void)
{
    m_monitor.enter();
    m_abort = true;
    m_monitor.notifyAll();
    while (m_numRunning)
        m_monitor.wait();
    m_abort = false;
    m_monitor.leave();

    for (int i = 0; i < m_threads.getSize(); i++)
    {
        delete m_threads[i].state;
        delete m_threads[i].thread;
    }

    m_threads.clear();
    delete m_serialState;
    m_serialState = NULL;

    for (int i = 0; i < m_pendingTasks.getSize(); i++)
    {
        delete m_pendingTasks[i]->slice;
        delete m_pendingTasks[i];
    }
    for (int i = 0; i < m_finishedTasks.getSize(); i++)
    {
        delete m_finishedTasks[i]->slice;
        delete m_finishedTasks[i];
    }

    m_tasks.clear();
    m_pendingTasks.clear();
    m_finishedTasks.clear();
}

//------------------------------------------------------------------------

bool BuilderBase::createRootSlice(OctreeSlice& slice, int objectID, const Params& params)
{
    FW_ASSERT(objectID >= 0 && objectID < m_file->getNumObjects());

    // Construct build data.

    ChildSlice cs;
    cs.init(0, Vec3i(0), OctreeFile::UnitScale, OctreeFile::UnitScale, false);
    Array<AttachIO::AttachType> runtimeAttachTypes;
    Mat4f octreeToObject;

    if (!createRootSlice(cs, runtimeAttachTypes, octreeToObject, objectID, params))
        return false;

    // Construct slice.

    slice.init(1, 1, 0, 0);
    slice.setID(m_file->getFreeSliceID());
    slice.setState(OctreeFile::SliceState_Unbuilt);
    slice.setCubePos(Vec3i(0));
    slice.setCubeScale(OctreeFile::UnitScale);
    slice.setNodeScale(OctreeFile::UnitScale);

    slice.startAttach(AttachIO::BuildDataAttach);
    BitWriter bitWriter(&slice.getData());
    String id = getIDString();
    FW_ASSERT(id.getLength() == 8);

    bitWriter.write(32, 1); // version
    bitWriter.write(32, objectID); // objectID
    bitWriter.write(32, 1); // numNodes
    for (int i = 0; i < 8; i++)
        bitWriter.write(8, id[i]); // subclassIDString[i]

    bitWriter.write(31, 0); // padding
    cs.writeBits(31, 0); // padding
    slice.getData().add(cs.buildData);
    slice.endAttach();

    // Update file.

    m_file->clearSlices(objectID);
    OctreeFile::Object fileObj = m_file->getObject(objectID);
    fileObj.octreeToObject = octreeToObject;
    fileObj.rootSlice = slice.getID();
    fileObj.runtimeAttachTypes = runtimeAttachTypes;
    m_file->setObject(objectID, fileObj);
    m_file->writeSlice(slice);
    return true;
}

//------------------------------------------------------------------------

void BuilderBase::threadFunc(void* param)
{
    pushMemOwner("BuilderBase state");
    ThreadEntry* entry = (ThreadEntry*)param;
    BuilderBase* b = entry->builder;
    FW_ASSERT(Thread::getCurrent() == entry->thread);

    entry->thread->setPriority(Thread::Priority_Min);
    b->m_monitor.enter();

    while (!b->m_abort)
    {
        if (!b->m_pendingTasks.getSize())
        {
            b->m_monitor.wait();
            continue;
        }

        Task* task = b->m_pendingTasks.remove(0);
        b->m_monitor.leave();

        entry->state->runTask(*task);

        b->m_monitor.enter();
        b->m_finishedTasks.add(task);
        b->m_monitor.notifyAll();
    }

    b->m_numRunning--;
    b->m_monitor.notifyAll();
    b->m_monitor.leave();
    popMemOwner();
}

//------------------------------------------------------------------------
