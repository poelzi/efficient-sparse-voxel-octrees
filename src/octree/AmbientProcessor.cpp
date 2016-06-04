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

#include "AmbientProcessor.hpp"
#include "Util.hpp"
#include "base/Hash.hpp"

using namespace FW;

//------------------------------------------------------------------------

AmbientProcessor::AmbientProcessor(OctreeFile* file, int objectID)
:   m_file          (file),
    m_objectID      (objectID),
    m_raysPerNode   (DefaultRaysPerNode),
    m_rayLength     (1.0f),
    m_flipNormals   (false),
    m_numWarps      (256)    // initial guess
{
    m_compiler.setSourceFile("src/octree/cuda/Ambient.cu");
    m_compiler.addOptions("-use_fast_math");
    m_compiler.include("src/framework");
    failIfError();

    FW_ASSERT(file);
    m_runtime = new OctreeRuntime(MemoryManager::Mode_Cuda);

    CudaModule::checkError("cuEventCreate", cuEventCreate(&m_kernelStartEvent, 0));
    CudaModule::checkError("cuEventCreate", cuEventCreate(&m_kernelEndEvent, 0));
}

//------------------------------------------------------------------------

AmbientProcessor::~AmbientProcessor(void)
{
    CudaModule::checkError("cuEventDestroy", cuEventDestroy(m_kernelStartEvent));
    CudaModule::checkError("cuEventDestroy", cuEventDestroy(m_kernelEndEvent));

    delete m_runtime;
}

//------------------------------------------------------------------------

void AmbientProcessor::run(void)
{
    Timer timer;
    timer.start();
    profileStart();
    profilePush("AmbientProcessor");

    // Select attachments.

    OctreeFile::Object object = m_file->getObject(m_objectID);
    Array<AttachIO::AttachType>  in  = object.runtimeAttachTypes;
    Array<AttachIO::AttachType>& out = object.runtimeAttachTypes;
    out.clear();
    for (int i = 0; i < AttachSlot_Max; i++)
        out.add(AttachIO::VoidAttach);

    // Assign imported attachments.

    for (int i = 0; i < in.getSize(); i++)
    {
        AttachIO::AttachType t = in[i];
        switch (t)
        {
        case AttachIO::ColorNormalPaletteAttach:    out[AttachSlot_Attribute] = t; break;
        case AttachIO::ColorNormalCornerAttach:     out[AttachSlot_Attribute] = t; break;
        case AttachIO::ColorNormalDXTAttach:        out[AttachSlot_Attribute] = t; break;
        case AttachIO::ContourAttach:               out[AttachSlot_Contour] = t; break;
        case AttachIO::AOAttach:                    out[AttachSlot_AO] = t; break;
        default:                                    break;
        }
    }

    // Set it.

    out[AttachSlot_AO] = AttachIO::AOAttach;
    m_file->setObject(m_objectID, object);

    // Start worrying about CUDA side.

    Array<AttachIO::AttachType> inputAttachTypes = out;
//  inputAttachTypes.resize(2); // discard everything except contours and colornormals
    m_runtime->addObject(m_objectID, object.rootSlice, inputAttachTypes);

    // Compile kernel.

    m_compiler.clearDefines();

    if (inputAttachTypes[AttachSlot_Contour] == AttachIO::ContourAttach)
        m_compiler.define("ENABLE_CONTOURS");

    switch (inputAttachTypes[AttachSlot_Attribute])
    {
    case AttachIO::ColorNormalPaletteAttach:    m_compiler.define("VOXELATTRIB_PALETTE"); break;
    case AttachIO::ColorNormalCornerAttach:     m_compiler.define("VOXELATTRIB_CORNER"); break;
    case AttachIO::ColorNormalDXTAttach:        m_compiler.define("VOXELATTRIB_DXT"); break;
    default:                                    fail("AmbientProcessor: Unsupported attribute attachment! (%d)", inputAttachTypes[AttachSlot_Attribute]); break;
    }

    if (m_flipNormals)
        m_compiler.define("FLIP_NORMALS");

    m_module = m_compiler.compile();
    failIfError();

    int nSlices = m_file->getNumSliceIDs();

    // init timers
    m_kernelTime = 0.0;
    m_requestsProcessed = 0;

    // load slices into runtime
    struct StackEntry
    {
        S32             sliceID;
        OctreeSlice*    slice;
        bool            pop;
    };
    Array<StackEntry> stack;
    stack.add().sliceID = object.rootSlice;

    profilePush("Load runtime");
    printf("Loading slices into runtime\n");
    int cnt = 0;
    while (stack.getSize())
    {
        printf("%d / %d ... \r", ++cnt, nSlices);

        S32 sliceID = stack.removeLast().sliceID; // no pops
        if (m_file->getSliceState(sliceID) != OctreeFile::SliceState_Complete)
            continue;

        OctreeSlice slice;
        profilePush("readSlice");
        m_file->readSlice(sliceID, slice);
        profilePop();

        int n = slice.getNumChildEntries();
        for (int i=0; i < n; i++)
        {
            int c = slice.getChildEntry(i);
            if (c == OctreeSlice::ChildEntry_Split)
                continue;
            if (c == OctreeSlice::ChildEntry_NoChild)
                continue;
            stack.add().sliceID = c;
        }

        for (int i = max(stack.getSize() - NumSlicesToPrefetch, 0); i < stack.getSize(); i++)
            m_file->readSlicePrefetch(stack[i].sliceID);

        m_runtime->setSliceToLoad(slice);
        if (!m_runtime->loadSlice())
            fail("out of memory");
    }
    profilePop();

    printf("\n%s\n", m_runtime->getStats().getPtr());

    // process slices

    printf("\nProcessing slices\n");
    profilePush("Process slices");
    StackEntry& rootEntry = stack.add();
    rootEntry.sliceID = object.rootSlice;
    rootEntry.pop = false;
    cnt = 0;

    // stack for tracking node positions
    Array<Array<NodeInfo> > positionStack;
    positionStack.reset(25);
    {
        NodeInfo& metaRoot = positionStack[24].add();
        metaRoot.pos.setZero();
        metaRoot.validMask = 1;
        metaRoot.lastInStrip = true;
        metaRoot.secondInPair = false;
    }

    // make sure the task lists are empty before starting
    m_sliceTasks.reset(0);
    m_sliceTaskTotal = 0;
    m_procTasks.reset(0);
    m_procTaskTotal = 0;

    Timer ambientTimer;
    ambientTimer.start();
    while (stack.getSize())
    {
        printf("%d / %d (%.2f/%.2f MRays/s)... \r",
            cnt,
            nSlices,
            m_kernelTime ? (0.000001 * m_raysPerNode * m_requestsProcessed / m_kernelTime) : 0,
            ambientTimer.getElapsed() ? (0.000001 * m_raysPerNode * m_requestsProcessed / ambientTimer.getElapsed()) : 0
            );

        StackEntry se = stack.removeLast();
        if (se.pop)
        {
            processSlice(se.slice, positionStack[se.slice->getNodeScale()]);
            continue;
        }

        S32 sliceID = se.sliceID;
        cnt++;
        if (m_file->getSliceState(sliceID) != OctreeFile::SliceState_Complete)
            continue;

        OctreeSlice* slice = new OctreeSlice;
        profilePush("readSlice");
        m_file->readSlice(sliceID, *slice);
        profilePop();

        // get position bits
        int scale = slice->getNodeScale();

        // track node positions
        int idx  = 0; // node index
        int sidx = 0; // split node index
        Array<NodeInfo>& sLevel  = positionStack[slice->getNodeScale()];
        Array<NodeInfo>& sParent = positionStack[slice->getNodeScale()+1];
        sLevel.reset(0);
        for (int i=0; i < sParent.getSize(); i++)
        {
            const NodeInfo& parent = sParent[i];
            Vec3i ppos = parent.pos;
            if (!isPosInCube(ppos, slice->getCubePos(), slice->getCubeScale()))
                continue;

            int ncnt = 0;
            int iidx = -1;
            for (int j=0; j < 8; j++)
            {
                if (!(parent.validMask & (1<<j)))
                    continue;
                if (!slice->isNodeSplit(idx++))
                    continue;

                iidx = sLevel.getSize();
                NodeInfo& info = sLevel.add();
                info.pos = ppos + base2ToVec(j) * (1 << scale);
                info.validMask = slice->getNodeValidMask(sidx);
                info.secondInPair = ((ncnt & 1) == 1);
                info.lastInStrip = false;
                sidx++;
                ncnt++;
            }
            if (iidx >= 0)
                sLevel[iidx].lastInStrip = true;
        }

        // add pop task for this slice before children
        StackEntry& popEntry = stack.add();
        popEntry.slice = slice;
        popEntry.pop   = true;

        // child slices
        int n = slice->getNumChildEntries();
        for (int i=0; i < n; i++)
        {
            int c = slice->getChildEntry(i);
            if (c == OctreeSlice::ChildEntry_Split)
                continue;
            if (c == OctreeSlice::ChildEntry_NoChild)
                continue;

            StackEntry& childEntry = stack.add();
            childEntry.sliceID = c;
            childEntry.pop     = false;
        }

        // prefetch in presence of pop entries
        int pcnt = 0;
        int spos = stack.getSize()-1;
        while (spos >= 0 && pcnt < NumSlicesToPrefetch)
        {
            if (!stack[spos].pop)
            {
                pcnt++;
                m_file->readSlicePrefetch(stack[spos].sliceID);
            }
            spos--;
        }
    }

    // flush task list
    initiateProcessing();
    finishProcessing();

    // done
    profilePop();
    printf("\ntime elapsed: %.2f s\n", timer.end());
    profilePop();
    profileEnd();
}

//------------------------------------------------------------------------

void AmbientProcessor::processSlice(OctreeSlice* slice, const Array<NodeInfo>& nodes)
{
    // find AO attachment if it already exists
    S32* attachData = 0;
    for (int i=0; i < slice->getNumAttach(); i++)
        if (slice->getAttachType(i) == AttachIO::AOAttach)
            attachData = slice->getData().getPtr(slice->getAttachOfs(i));

    if (!attachData)
    {
        // reconstruct slice
        OctreeSlice src = *slice; // take a copy
        if (nodes.getSize() != slice->getNumSplitNodes())
            fail("Node count mismatch");
        slice->init(src.getNumChildEntries(), src.getNumAttach()+1, src.getNumNodes(), src.getNumSplitNodes());
        slice->setID(src.getID());
        slice->setState(src.getState());
        slice->setCubePos(src.getCubePos());
        slice->setCubeScale(src.getCubeScale());
        slice->setNodeScale(src.getNodeScale());
        memcpy(slice->getChildEntryPtr(), src.getChildEntryPtr(), src.getNumChildEntries() * sizeof(S32));
        memcpy(slice->getNodeSplitPtr(), src.getNodeSplitPtr(), ((src.getNumNodes() + 31) >> 5) * 4);
        memcpy(slice->getNodeValidMaskPtr(), src.getNodeValidMaskPtr(), src.getNumSplitNodes());
        for (int i=0; i < src.getNumAttach(); i++)
        {
            slice->startAttach(src.getAttachType(i));
            slice->getData().add(src.getData().getPtr(src.getAttachOfs(i)), src.getAttachSize(i));
            slice->endAttach();
        }
        // create the new attachment
        S32 base = slice->getData().getSize();
        slice->startAttach(AttachIO::AOAttach);
        for (int i=0; i < slice->getNumSplitNodes(); i++)
            if (nodes[i].secondInPair || nodes[i].lastInStrip)
                slice->getData().add(0, 2); // all black blocks
        slice->endAttach();
        attachData = slice->getData().getPtr(base);
    }

    // count number of ao requests
    for (int i=0; i < slice->getNumSplitNodes(); i++)
        m_sliceTaskTotal += popc8(slice->getNodeValidMask(i));

    // add prepared slice to slice task list
    SliceTask* task = new SliceTask;
    task->slice = slice;
    task->nodes = nodes;
    task->attachData = attachData;
    m_sliceTasks.add(task);

    // check if we have enough AO rays
    if (m_sliceTaskTotal * m_raysPerNode >= MinRaysPerBatch)
        initiateProcessing();
}

void AmbientProcessor::initiateProcessing(void)
{
    // finish previous processing if any
    finishProcessing();

    // initialize ao request buffer
    int numRequests = m_sliceTaskTotal;

    // Perform these operations only for non-zero number of requests.
    if (numRequests)
    {
        m_requestBuffer.resizeDiscard(numRequests * sizeof(AmbientRequest));
        AmbientRequest* reqPtr = (AmbientRequest*)m_requestBuffer.getMutablePtr();
        for (int t=0; t < m_sliceTasks.getSize(); t++)
        {
            SliceTask* task = m_sliceTasks[t];
            OctreeSlice* slice = task->slice;
            const Array<NodeInfo>& nodes = task->nodes;

            U32 childHalfSize = 1 << (slice->getNodeScale() - 2);
            for (int i=0; i < slice->getNumSplitNodes(); i++)
            {
                U32 validMask = slice->getNodeValidMask(i);
                Vec3i base = nodes[i].pos + childHalfSize;

                // loop over children
                for (int j=0; j < 8; j++)
                {
                    if (!(validMask & (1<<j)))
                        continue;

                    AmbientRequest& req = *reqPtr++;
                    req.pos     = base + base2ToVec(j) * 2 * childHalfSize;
                    req.level   = slice->getNodeScale() - 1;
                }
            }
        }

        // Reset request buffer size.
        m_resultBuffer.resizeDiscard(numRequests * sizeof(AmbientResult));

        // thunder and lightning! invoke CUDA

        // Clear active warp buffer.
        m_activeWarps.resizeDiscard(m_numWarps * sizeof(S32));
        memset(m_activeWarps.getMutablePtr(), 0, (size_t)m_activeWarps.getSize());

        // Launch all requests.
        int maxRequests = max(MaxRaysPerBatch / m_raysPerNode, 1);
        for (int firstRequest = 0; firstRequest < numRequests; firstRequest += maxRequests)
        {
            CudaModule::sync(false); // keep watchdog awake and prevent race conditions

            // Set input.
            AmbientInput& in = getInput();
            in.numRequests   = min(numRequests - firstRequest, maxRequests);
            in.raysPerNode   = m_raysPerNode;
            in.rayLength     = m_rayLength;
            in.requestPtr    = m_requestBuffer.getCudaPtr(firstRequest * sizeof(AmbientRequest));
            in.resultPtr     = m_resultBuffer.getMutableCudaPtr(firstRequest * sizeof(AmbientResult));
            in.rootNode      = m_runtime->getRootNodeCuda(m_objectID);
            in.activeWarps   = m_activeWarps.getMutableCudaPtr();

            OctreeMatrices& om  = in.octreeMatrices;
            Mat4f octreeToWorld = m_file->getObject(m_objectID).objectToWorld * m_file->getObject(m_objectID).octreeToObject;
            om.octreeToWorld    = octreeToWorld * Mat4f::translate(Vec3f(-1.0f));
            om.worldToOctree    = invert(in.octreeMatrices.octreeToWorld);
            om.octreeToWorldN   = octreeToWorld.getXYZ().inverted().transposed();

            // Determine grid size.
            Vec2i blockSize = Vec2i(AMBK_BLOCK_WIDTH, AMBK_BLOCK_HEIGHT);
            int blockThreads = blockSize.x * blockSize.y;
            Vec2i gridSize = Vec2i((m_numWarps * 32 + blockThreads - 1) / blockThreads, 1);

            // Init warp counter.
            getWarpCounter() = 0;
            m_module->updateGlobals();

            // Launch kernel.
            if (firstRequest == 0)
            {
                CudaModule::sync(false); // for accurate timing
                CudaModule::checkError("cuEventRecord", cuEventRecord(m_kernelStartEvent, NULL));
            }
            m_module->getKernel("ambientKernel").setAsync().launch(gridSize * blockSize, blockSize); // async launch
        }

        CudaModule::checkError("cuEventRecord", cuEventRecord(m_kernelEndEvent, NULL));
    }

    // move tasks from current queue to processing queue
    m_procTasks = m_sliceTasks;
    m_procTaskTotal = m_sliceTaskTotal;
    m_sliceTasks.reset(0);
    m_sliceTaskTotal = 0;
}

void AmbientProcessor::finishProcessing(void)
{
    int numRequests = m_procTaskTotal;

    // Kernel was launched only if there were any requests.
    if (numRequests)
    {
        // Finish running previous kernel.
        profilePush("wait for CUDA");
        CudaModule::sync(true);
        profilePop();

        float kernelTime;
        CudaModule::checkError("cuEventElapsedTime", cuEventElapsedTime(&kernelTime, m_kernelStartEvent, m_kernelEndEvent));
        m_kernelTime += kernelTime * .001f;

        // Adjust the number of warps.
        const S32* activeWarps = (const S32*)m_activeWarps.getPtr();
        int numWarps = 0;
        for (int i = 0; i < m_numWarps; i++)
            if (activeWarps[i])
                numWarps++;
        if (numWarps * 2 > m_numWarps)
        {
            printf("warp count auto-detected: %d warps        \n", numWarps);
            m_numWarps = numWarps * 2;
        }

        // Update request counter.
        m_requestsProcessed += numRequests;

        // read node results and write to slices
        AmbientResult* resPtr = (AmbientResult*)m_resultBuffer.getPtr();
        for (int t=0; t < m_procTasks.getSize(); t++)
        {
            SliceTask* task = m_procTasks[t];
            OctreeSlice* slice = task->slice;
            S32* attachData = task->attachData;
            const Array<NodeInfo>& nodes = task->nodes;

            // Collect DXT blocks.
            float block[16];
            float bmin = 1.f;
            float bmax = 0.f;
            for (int j=0; j < 16; j++)
                block[j] = 0.f;
            for (int i=0; i < slice->getNumSplitNodes(); i++)
            {
                U32 validMask = slice->getNodeValidMask(i);
                int obase = nodes[i].secondInPair ? 8 : 0;
                for (int j=0; j < 8; j++)
                {
                    if (!(validMask & (1<<j)))
                        continue;

                    Vec3f res = (resPtr++)->ao;
                    float occ = res.x;

                    bmin = min(bmin, occ);
                    bmax = max(bmax, occ);
                    block[obase + j] = occ;
                }

                if (nodes[i].secondInPair || nodes[i].lastInStrip)
                {
                    // construct block
                    U8 imin = (U8)min(max(bmin * 255.f, 0.f), 255.f);
                    U8 imax = (U8)min(max(bmax * 255.f, 0.f), 255.f);
                    U64 b = imin | (imax << 8);
                    for (int j=0; j < 16; j++)
                    {
                        float val  = (block[j] - bmin) / (bmax - bmin) * 7.f;
                        int   ival = (int)(val + .5f);
                        ival = min(max(ival, 0), 7);
                        b |= ((U64)ival) << (16 + j*3);
                    }
                    // store it
                    *((U64*)attachData) = b;
                    attachData += 2;
                    // clear block data
                    for (int j=0; j < 16; j++)
                        block[j] = 0.f;
                    bmin = 1.f;
                    bmax = 0.f;
                }
            }

            // write to disk
            m_file->writeSlice(*slice);

            // delete task and slice
            delete task;
            delete slice;
        }
    }

    // all done
    m_procTasks.clear();
    m_procTaskTotal = 0;
}
