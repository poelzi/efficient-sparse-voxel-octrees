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
#include "base/Math.hpp"
#include "base/DLLImports.hpp"

namespace FW
{

//------------------------------------------------------------------------

#define RCK_TRACE_BLOCK_WIDTH   32
#define RCK_TRACE_BLOCK_HEIGHT  2

//------------------------------------------------------------------------

enum RenderFlags
{
    RenderFlags_CoarsePass              = 1 << 0,
    RenderFlags_UseCoarseData           = 1 << 1,
    RenderFlags_VisualizeIterations     = 1 << 2,
    RenderFlags_VisualizeRaycastLevel   = 1 << 3,
};

//------------------------------------------------------------------------

enum AttachSlot
{
    AttachSlot_Contour = 0,
    AttachSlot_Attribute,
    AttachSlot_AO,

    AttachSlot_Max
};

//------------------------------------------------------------------------
// PerfCounter_Instructions counts SASS instructions, excluding the
// following cases where dual issue is assumed:
//
//      - MOV between registers
//      - FMUL between registers
//      - FRCP of a register
//------------------------------------------------------------------------

#define PERF_COUNTER_LIST(X) \
    X(Instructions) \
    X(Iterations) \
    X(Intersect) \
    X(Push) \
    X(PushStore) \
    X(Advance) \
    X(Pop) \
    X(GlobalAccesses) \
    X(GlobalBytes) \
    X(GlobalTransactions) \
    X(LocalAccesses) \
    X(LocalBytes) \
    X(LocalTransactions)

enum PerfCounter
{
#define X(NAME) PerfCounter_ ## NAME,
    PERF_COUNTER_LIST(X)
#undef X
    PerfCounter_Max
};

//------------------------------------------------------------------------

struct OctreeMatrices
{
    Mat4f           viewportToCamera;
    Mat4f           cameraToOctree;
    Mat4f           octreeToWorld;
    F32             pixelInOctree;      // average size of a near-plane-pixel in the octree

    Mat4f           worldToOctree;
    Mat3f           octreeToWorldN;     // normal transformation matrix
    Vec3f           cameraPosition;     // camera position in world space
    Mat4f           octreeToViewport;
    Mat4f           viewportToOctreeN;  // matrix for transforming frustum planes
};

//------------------------------------------------------------------------

struct RenderInput
{
    Vec2i           frameSize;
    U32             flags;
    S32             batchSize;          // number of warps per batch
    S32             aaRays;             // number of AA rays per pixel for normal pass
    F32             maxVoxelSize;       // in pixels
    F32             brightness;
    S32             coarseSize;         // block size for coarse data
    Vec2i           coarseFrameSize;    // coarse data buffer size
    S32             numPrimaryRays;     // includes aa rays
    S32             totalWork;          // numPrimaryRays * numFrameRepeats
    CUdeviceptr     frame;
    CUdeviceptr     frameCoarse;        // contains tmin as float
    CUdeviceptr     aaSampleBuffer;     // individual aa samples
    CUdeviceptr     rootNode;
    CUdeviceptr     activeWarps;
    CUdeviceptr     perfCounters;
    OctreeMatrices  octreeMatrices;
};

//------------------------------------------------------------------------
}
