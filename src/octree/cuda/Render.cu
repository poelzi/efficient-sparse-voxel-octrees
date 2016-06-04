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

#include "Render.hpp"
#include "../io/OctreeRuntime.hpp"
#include "../io/AttachIO.hpp"

using namespace FW;

//------------------------------------------------------------------------
// Global variables.
//------------------------------------------------------------------------

__constant__ int4   c_input[(sizeof(RenderInput) + sizeof(int4) - 1) / sizeof(int4)];
__constant__ int4   c_blurLUT[BLUR_LUT_SIZE];
__device__ S32      g_warpCounter;

texture<U32, 1>     texIndexToPixel;
texture<U32, 1>     texIndexToPixelCoarse;
texture<F32, 1>     texFrameCoarseIn;
texture<uchar4, 1, cudaReadModeNormalizedFloat> texTempFrameIn;
texture<uchar4, 1, cudaReadModeNormalizedFloat> texAASamplesIn;

//------------------------------------------------------------------------
// Common helper functions.
//------------------------------------------------------------------------

__device__ inline const RenderInput& getInput(void)
{
    return *(const RenderInput*)c_input;
}

__device__ inline void updateCounter(PerfCounter counter, int amount = 1)
{
#ifdef ENABLE_PERF_COUNTERS
    int warpIdx = threadIdx.y + blockIdx.x * RCK_TRACE_BLOCK_HEIGHT;
    volatile S64* ptr = (S64*)getInput().perfCounters + (warpIdx * PerfCounter_Max + counter) * 33;
    ptr[threadIdx.x] += amount; // thread counter
    ptr[32] += amount; // warp counter
#endif
}

#ifdef ENABLE_PERF_COUNTERS
__device__ inline bool checkTransaction(int page)
{
    volatile __shared__ U32 buffer[RCK_TRACE_BLOCK_WIDTH * RCK_TRACE_BLOCK_HEIGHT];
    int fullIdx = threadIdx.x + threadIdx.y * RCK_TRACE_BLOCK_WIDTH;
    volatile U32* ptr = &buffer[fullIdx & -16];
    int idx = fullIdx & 15;

    // Clear buffer.

    for (int i = 0; i < 16; i++)
        ptr[i] = 0xFFFFFFFF;

    // Write address.

    ptr[idx] = page;

    // Check for duplicates.

    for (int i = 0; i < idx; i++)
        if (ptr[i] == page)
            return false;
    return true;
}
#endif

__device__ inline void updateCountersForGlobalAccess(int sizeLog2, S32* addr)
{
#ifdef ENABLE_PERF_COUNTERS
    updateCounter(PerfCounter_GlobalAccesses);
    updateCounter(PerfCounter_GlobalBytes, 1 << sizeLog2);
    if (checkTransaction((U32)addr >> ::min(sizeLog2 + 5, 7)))
        updateCounter(PerfCounter_GlobalTransactions);
#endif
}

__device__ inline void updateCountersForLocalAccess(int sizeLog2, int id)
{
#ifdef ENABLE_PERF_COUNTERS
    updateCounter(PerfCounter_LocalAccesses);
    updateCounter(PerfCounter_LocalBytes, 1 << sizeLog2);
    if (checkTransaction(id))
        updateCounter(PerfCounter_LocalTransactions);
#endif
}

//------------------------------------------------------------------------
// Utility routines.
//------------------------------------------------------------------------

#include "Util.inl"
#include "Raycast.inl"
#include "AttribLookup.inl"

//------------------------------------------------------------------------
// Private definitions.
//------------------------------------------------------------------------

#define BLUR_FACTOR 1.0f // Controls total amount of blurring. Larger than 1.0 causes everything to blur.

struct Aux // shared memory auxiliary storage
{
    U32*        framePtr;
#ifdef LARGE_RECONSTRUCTION_KERNEL
    U32*        aaSamplePtr;
#endif
#ifdef JITTER_LOD
    float       vSizeMultiplier;
#endif

    union
    {
        S32     fetchWorkTemp;

        Ray     ray;

        struct
        {
            U32 color;
            U32 alpha;
        }       aa;
    };
};

__constant__ float2 c_aa4table[4] =
{
    { 0.125f, 0.375f },
    { 0.375f, 0.875f },
    { 0.875f, 0.625f },
    { 0.625f, 0.125f }
};

//------------------------------------------------------------------------
// Ray generation.
//------------------------------------------------------------------------

__device__ Ray constructPrimaryRay(int ppos, int ridx, volatile Aux& aux)
{
    const RenderInput& input = getInput();
    float vsize = input.maxVoxelSize;
    int xsize = input.frameSize.x;

    // if coarse pass, make voxel large enough so that rays cannot accidentally get past it
    if (input.flags & RenderFlags_CoarsePass)
    {
        vsize = (float)input.coarseSize * 2.83f; // sqrt(8)
        xsize = input.coarseFrameSize.x;
    } else
    {
#ifdef JITTER_LOD
        // perturb randomly
        U32 a = ppos;
        U32 b = ridx;
        U32 c = 0x9e3779b9u;
        jenkinsMix(a, b, c);
        float f = (float)c / ((float)(1u << 31) * 2.f);
        f = .5f + .5f*f;
        aux.vSizeMultiplier = f;
        vsize *= f;
#endif
    }

    // find ray coordinates
    int pixely = ppos / xsize;
    int pixelx = ppos - (pixely * xsize);
    F32 fx = pixelx;
    F32 fy = pixely;

    if (input.flags & RenderFlags_CoarsePass)
    {
        fx *= (float)input.coarseSize;
        fy *= (float)input.coarseSize;
    } else
    {
        if (input.aaRays == 1)
        {
            fx += .5f; // center of pixel
            fy += .5f;
        } else if (input.aaRays == 4)
        {
            int aidx = (ridx & 3);
            fx += c_aa4table[aidx].x;
            fy += c_aa4table[aidx].y;
        }
    }

    F32 tmin = 0.f;
    if (getInput().flags & RenderFlags_UseCoarseData)
    {
        // fetch tmin
        int bx = pixelx / input.coarseSize;
        int by = pixely / input.coarseSize;
        int bidx = bx + by * input.coarseFrameSize.x;
        F32 tmin0 = tex1Dfetch(texFrameCoarseIn, bidx);
        F32 tmin1 = tex1Dfetch(texFrameCoarseIn, bidx+1);
        F32 tmin2 = tex1Dfetch(texFrameCoarseIn, bidx+input.coarseFrameSize.x);
        F32 tmin3 = tex1Dfetch(texFrameCoarseIn, bidx+input.coarseFrameSize.x+1);
        tmin = fminf(fminf(tmin0, tmin1), fminf(tmin2, tmin3));
        tmin = fminf(tmin, 0.9999f);
    }

    const Mat4f& vtc = input.octreeMatrices.viewportToCamera;
    const Mat4f& cto = input.octreeMatrices.cameraToOctree;

    float4 pos = make_float4(
        vtc.m00 * fx + vtc.m01 * fy + vtc.m03,
        vtc.m10 * fx + vtc.m11 * fy + vtc.m13,
        vtc.m20 * fx + vtc.m21 * fy + vtc.m23,
        vtc.m30 * fx + vtc.m31 * fy + vtc.m33);

    float3 near = make_float3(
        pos.x - vtc.m02,
        pos.y - vtc.m12,
        pos.z - vtc.m22);
    float near_sz = input.octreeMatrices.pixelInOctree * vsize;

    float3 diff = make_float3(
        vtc.m32 * pos.x - vtc.m02 * pos.w,
        vtc.m32 * pos.y - vtc.m12 * pos.w,
        vtc.m32 * pos.z - vtc.m22 * pos.w);
    float diff_sz = near_sz * vtc.m32;

    float a = 1.0f / (pos.w - vtc.m32);
    float b = 2.0f * a / fmaxf(pos.w + vtc.m32, 1.0e-8f);
    float c = tmin * b;

    Ray ray;
    ray.orig = near * a - diff * c;
    ray.dir  = diff * (c - b);
    ray.orig_sz = near_sz * a - diff_sz * c;
    ray.dir_sz  = diff_sz * (c - b);

    ray.orig = cto * ray.orig;
    ray.dir = make_float3(
        cto.m00 * ray.dir.x + cto.m01 * ray.dir.y + cto.m02 * ray.dir.z,
        cto.m10 * ray.dir.x + cto.m11 * ray.dir.y + cto.m12 * ray.dir.z,
        cto.m20 * ray.dir.x + cto.m21 * ray.dir.y + cto.m22 * ray.dir.z);
    return ray;
}

//------------------------------------------------------------------------
// Ray processing.
//------------------------------------------------------------------------

__device__ U32 processPrimaryRay(volatile Ray& ray, volatile F32& vSizeMultiplier)
{
    // Cast primary ray.

    CastResult castRes;
    CastStack stack;
    castRay(castRes, stack, ray);

    // Handle visualizations.

    if (getInput().flags & RenderFlags_VisualizeIterations)
    {
        F32 v = 255.0f * (F32)castRes.iter / 64.0f;
        return toABGR(make_float4(v, v, v, 0.0f));
    }
    else if (getInput().flags & RenderFlags_VisualizeRaycastLevel)
    {
        F32 v = 0.0f;
        if (castRes.t <= 1.0f)
            v = 255.0f - ((F32)CAST_STACK_DEPTH - (F32)castRes.stackPtr) * (255.0f / 18.0f);
        return toABGR(make_float4(v * 0.5f, v, v * 0.5f, 0.0f));
    }

    // Initialize light and incident vectors.

    float3 L = make_float3(0.3643f, 0.3535f, 0.8616f);
    float3 I = normalize(extractMat3f(getInput().octreeMatrices.octreeToWorld) * get(ray.dir));

    // No hit => sky.

    if (castRes.t > 1.0f)
    {
        float3 c;
        if (I.y >= 0.f)
        {
            float3 horz = { 179.0f, 205.0f, 253.0f };
            float3 zen  = { 77.0f,  102.0f, 179.0f };
            c = horz + (zen - horz) * I.y * I.y;
            c *= 2.5f;
        }
        else
        {
            float3 horz = { 192.0f, 154.0f, 102.0f };
            float3 zen  = { 128.0f, 102.0f, 77.0f };
            c = horz - (zen - horz) * I.y;
        }

        c *= fmaxf(L.y, 0.0f);
        float IL = dot(I, L);
        if (IL > 0.0f) 
            c += make_float3(255.0f, 179.0f, 102.0f) * powf(IL, 1000.0f); // sun

        return toABGR(make_float4(c.x, c.y, c.z, 0.0f));
    }

    // Get voxel color, normal, and ambient.

    float4 voxelColor;
    float3 voxelNormal;
    lookupVoxelColorNormal(voxelColor, voxelNormal, castRes, stack);

    F32 voxelAmbient = 1.0f;
#ifdef VOXELATTRIB_AO
    lookupVoxelAO(voxelAmbient, castRes, stack);
#endif

    // Calculate world-space normal and reflection vectors.

    float3 N  = normalize(getInput().octreeMatrices.octreeToWorldN * voxelNormal);
    float3 R  = (I - N * (dot(N, I) * 2.0f));
    F32    LN = dot(L, N);

    // Cast shadow ray.

    bool shadow = (LN <= 0.0f);
#ifdef ENABLE_SHADOWS
    if (!shadow)
    {
        Ray rayShad;
        rayShad.orig_sz = 0.0f;
        rayShad.dir_sz  = 0.0f;
        rayShad.orig    = castRes.pos + L * 0.0006f;
        rayShad.dir     = L * 3.0f;

        CastResult castResShad;
        CastStack  stackShad;
        castRay(castResShad, stackShad, rayShad);
        shadow = (castResShad.t <= 1.0f);
	}
#endif

    // Shade.

    float4 shadedColor = voxelColor * (voxelAmbient * (0.25f + LN * ((LN < 0.0f) ? 0.15f : (shadow) ? 0.25f : 1.0f)));
    if (!shadow)
        shadedColor += make_float4(32.f, 32.f, 32.f, 0.0f) * powf(fmaxf(dot(L, R), 0.0f), 18.0f); // specular
    shadedColor *= getInput().brightness;

    U32 color = toABGR(shadedColor);

    // Determine post-process filter radius.

    float vSize = (F32)(1 << castRes.stackPtr) / (F32)(1 << CAST_STACK_DEPTH);
    float pSize = ray.orig_sz + castRes.t * ray.dir_sz;
#ifdef JITTER_LOD
    vSize *= vSizeMultiplier;
#endif
    float blurRadius = ::max(vSize / pSize * getInput().maxVoxelSize, 1.0f);

    // Encode in the alpha channel.

    shadedColor.w = log2f(blurRadius) * 32.0f + 0.5f;
    return toABGR(shadedColor);
}

//------------------------------------------------------------------------
// Persistent threads.
//------------------------------------------------------------------------

__device__ void fetchWorkFirst(int& warp, int& batchCounter, int* warpCounter, int batchSize, volatile S32& sharedTemp)
{
#ifdef PERSISTENT_THREADS
    if (threadIdx.x == 0)
        sharedTemp = atomicAdd(warpCounter, batchSize);
    warp = sharedTemp;
    batchCounter = batchSize;
#else
    warp = threadIdx.y + blockIdx.x * RCK_TRACE_BLOCK_HEIGHT;
    batchCounter = 0;
#endif
}

__device__ void fetchWorkNext(int& warp, int& batchCounter, int* warpCounter, int batchSize, volatile S32& sharedTemp)
{
#ifdef PERSISTENT_THREADS
    batchCounter--;
    if (batchCounter > 0)
        warp++;
    else
    {
        if (threadIdx.x == 0)
            sharedTemp = atomicAdd(warpCounter, batchSize);
        batchCounter = batchSize;
        warp = sharedTemp;
    }
#else
    warp = 0x03FFFFFF;
#endif
}

//------------------------------------------------------------------------
// Rendering kernel.
//------------------------------------------------------------------------

#ifdef KERNEL_RENDER

extern "C" __global__ void kernel(void)
{
    const RenderInput& input = getInput();
    __shared__ Aux auxbuf[RCK_TRACE_BLOCK_WIDTH * RCK_TRACE_BLOCK_HEIGHT];
    volatile Aux& aux0 = auxbuf[RCK_TRACE_BLOCK_WIDTH * threadIdx.y];
    volatile Aux& aux  = auxbuf[threadIdx.x + RCK_TRACE_BLOCK_WIDTH * threadIdx.y];

    // fetch first warp of work
    int warp, batchCounter;
    fetchWorkFirst(warp, batchCounter, &g_warpCounter, input.batchSize, aux0.fetchWorkTemp);
    if (warp * 32 >= input.totalWork)
        return; // terminate before starting at all

#ifdef PERSISTENT_THREADS
    // notice that work is being done in this warp slot
    ((S32*)input.activeWarps)[threadIdx.y + blockIdx.x * RCK_TRACE_BLOCK_HEIGHT] = 1;
#endif

    // main warp loop
    for (;;)
    {
        // ray index
        int ridx = warp * 32 + threadIdx.x;
        if (ridx >= input.totalWork)
            return; // terminate individual rays

        // calculate pixel index, position, and frame buffer pointer
        int pidx = (ridx / input.aaRays) % input.numPrimaryRays;
        int ppos;

        if (input.flags & RenderFlags_CoarsePass)
        {
            ppos = tex1Dfetch(texIndexToPixelCoarse, pidx);
            aux.framePtr = (U32*)input.frameCoarse + ppos;
        }
        else
        {
            ppos = tex1Dfetch(texIndexToPixel, pidx);
            aux.framePtr = (U32*)input.frame + ppos;
#ifdef LARGE_RECONSTRUCTION_KERNEL
            aux.aaSamplePtr = (U32*)input.aaSampleBuffer + ppos * input.aaRays + (ridx % input.aaRays);
#endif
        }

        // construct ray
        Ray ray = constructPrimaryRay(ppos, ridx, aux);
        aux.ray.orig.x = ray.orig.x;
        aux.ray.orig.y = ray.orig.y;
        aux.ray.orig.z = ray.orig.z;
        aux.ray.dir.x = ray.dir.x;
        aux.ray.dir.y = ray.dir.y;
        aux.ray.dir.z = ray.dir.z;
        aux.ray.orig_sz = ray.orig_sz;
        aux.ray.dir_sz = ray.dir_sz;

        if (getInput().flags & RenderFlags_CoarsePass)
        {
            CastResult castRes;
            CastStack stack;
            castRay(castRes, stack, aux.ray);
            if (castRes.t < 1.0f)
            {
                F32 size = (F32)(1 << castRes.stackPtr) / (F32)(1 << CAST_STACK_DEPTH);
                castRes.t -= size / length(get(aux.ray.dir)) * 0.5f;
            }
            *(float*)aux.framePtr = ::max(castRes.t, 0.0f);
        } else
        {
#ifdef JITTER_LOD
            U32 color = processPrimaryRay(aux.ray, aux.vSizeMultiplier);
#else
            U32 color = processPrimaryRay(aux.ray, aux.ray.orig.x);
#endif

            // write results
            if (input.aaRays == 1)
                *aux.framePtr = color; // no AA
            else
            {
#ifdef LARGE_RECONSTRUCTION_KERNEL
                *aux.aaSamplePtr = color; // individual sample result
#endif
                // unpack result
                U32 resc = (color & 0xff) | ((color & 0xff00) << 2) | ((color & 0xff0000) << 4);
                aux.aa.color = resc;  // rgb with bits shifted up
                aux.aa.alpha = color; // original color

                // sum with one thread
                if ((threadIdx.x & 3) == 0)
                {
                    // rgb
                    U32 resc0 = (&aux)[0].aa.color;
                    U32 resc1 = (&aux)[1].aa.color;
                    U32 resc2 = (&aux)[2].aa.color;
                    U32 resc3 = (&aux)[3].aa.color;
                    resc = (resc0 + resc1 + resc2 + resc3);
                    resc = ((resc >> 2) & 0xff) | ((resc >> 4) & 0xff00) | ((resc >> 6) & 0xff0000);

                    // alpha
                    U32 resa0 = (&aux)[0].aa.alpha;
                    U32 resa1 = (&aux)[1].aa.alpha;
                    U32 resa2 = (&aux)[2].aa.alpha;
                    U32 resa3 = (&aux)[3].aa.alpha;
                    U32 resa = ::min(::min(resa0, resa1), ::min(resa2, resa3));;

                    // combine min alpha and avg color
                    *aux.framePtr = (resa & 0xff000000) | resc;
                }
            }
        }

        // fetch more work
        fetchWorkNext(warp, batchCounter, &g_warpCounter, input.batchSize, aux0.fetchWorkTemp);
    }
}

#endif

//------------------------------------------------------------------------
// Performance measurement kernel.
//------------------------------------------------------------------------

#ifdef KERNEL_RAYCAST_PERF

extern "C" __global__ void kernel(void)
{
    const RenderInput& input = getInput();
    __shared__ Aux auxbuf[RCK_TRACE_BLOCK_WIDTH * RCK_TRACE_BLOCK_HEIGHT];
    volatile Aux& aux0 = auxbuf[RCK_TRACE_BLOCK_WIDTH * threadIdx.y];
    volatile Aux& aux  = auxbuf[threadIdx.x + RCK_TRACE_BLOCK_WIDTH * threadIdx.y];

    // fetch first warp of work
    int warp, batchCounter;
    fetchWorkFirst(warp, batchCounter, &g_warpCounter, input.batchSize, aux0.fetchWorkTemp);
    if (warp * 32 >= input.totalWork)
        return; // terminate before starting at all

#ifdef PERSISTENT_THREADS
    // notice that work is being done in this warp slot
    ((S32*)input.activeWarps)[threadIdx.y + blockIdx.x * RCK_TRACE_BLOCK_HEIGHT] = 1;
#endif

    // main warp loop
    for (;;)
    {
        // ray index
        int ridx = warp * 32 + threadIdx.x;
        if (ridx >= input.totalWork)
            return; // terminate individual rays

        // calculate pixel index, position, and frame buffer pointer
        int pidx = ridx % input.numPrimaryRays;
        int ppos;

        if (input.flags & RenderFlags_CoarsePass)
        {
            ppos = tex1Dfetch(texIndexToPixelCoarse, pidx);
            aux.framePtr = (U32*)input.frameCoarse + ppos;
        }
        else
        {
            ppos = tex1Dfetch(texIndexToPixel, pidx);
            aux.framePtr = (U32*)input.frame + ppos;
        }

        // construct ray
        Ray ray = constructPrimaryRay(ppos, pidx, aux);
        aux.ray.orig.x = ray.orig.x;
        aux.ray.orig.y = ray.orig.y;
        aux.ray.orig.z = ray.orig.z;
        aux.ray.dir.x = ray.dir.x;
        aux.ray.dir.y = ray.dir.y;
        aux.ray.dir.z = ray.dir.z;
        aux.ray.orig_sz = ray.orig_sz;
        aux.ray.dir_sz = ray.dir_sz;

        CastResult castRes;
        CastStack stack;
        castRay(castRes, stack, aux.ray);
        if (castRes.t < 1.0f)
        {
            F32 size = (F32)(1 << castRes.stackPtr) / (F32)(1 << CAST_STACK_DEPTH);
            castRes.t -= size / length(get(aux.ray.dir)) * 0.5f;
        }
        *(float*)aux.framePtr = castRes.t;

        // fetch more work
        fetchWorkNext(warp, batchCounter, &g_warpCounter, input.batchSize, aux0.fetchWorkTemp);
    }
}

#endif

//------------------------------------------------------------------------
// Post-process filter kernel.
//------------------------------------------------------------------------

extern "C" __global__ void blurKernel(void)
{
    const RenderInput& input = getInput();
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int cx = input.frameSize.x;
    int cy = input.frameSize.y;

    if (px >= cx || py >= cy)
        return;

    U32* pResult = ((U32*)input.frame) + (px + cx*py);

    float4 ccol = tex1Dfetch(texTempFrameIn, px + cx*py);
    float rad = exp2f(ccol.w * (255.f / 32.f)) * BLUR_FACTOR;

    if (rad <= 1.f)
    {
        // single-pixel case
#ifdef LARGE_RECONSTRUCTION_KERNEL
        if (input.aaRays == 4)
        {
            int apos = (px + cx*py) * 4;
            cx *= 4;

            ccol *= 4.f;
            ccol += tex1Dfetch(texAASamplesIn, apos-4 +2);
            ccol += tex1Dfetch(texAASamplesIn, apos-4 +3);
            ccol += tex1Dfetch(texAASamplesIn, apos+4 +0);
            ccol += tex1Dfetch(texAASamplesIn, apos+4 +1);
            ccol += tex1Dfetch(texAASamplesIn, apos-cx+1);
            ccol += tex1Dfetch(texAASamplesIn, apos-cx+2);
            ccol += tex1Dfetch(texAASamplesIn, apos+cx+0);
            ccol += tex1Dfetch(texAASamplesIn, apos+cx+3);
            ccol *= (1.f/12.f);
            ccol.w = 1.f;
            *pResult = toABGR(ccol * 255.f);
        } else
#endif
        {
            ccol.w = 1.f;
            *pResult = toABGR(ccol * 255.f);
        }
        return;
    }

    float4 accum = {0, 0, 0, 0};
    for (int i=0; i < BLUR_LUT_SIZE; i++)
    {
        int4 b = c_blurLUT[i];
        float d = __int_as_float(b.w);
        if (d >= rad)
            break;

        int x = px + b.x;
        int y = py + b.y;

        float w = __int_as_float(b.z);

        if (x < 0) w = 0.f;
        if (y < 0) w = 0.f;
        if (x >= input.frameSize.x) w = 0.f;
        if (y >= input.frameSize.y) w = 0.f;

        float4 c = tex1Dfetch(texTempFrameIn, x + __mul24(cx, y));
        float rad2 = exp2f(c.w * (255.f / 32.f)) * BLUR_FACTOR;
        if (w > 0.f)
            rad = ::min(rad, rad2);

        w *= fminf(fmaxf(rad - d, 0.f), 1.f);

        accum.x += c.x * w;
        accum.y += c.y * w;
        accum.z += c.z * w;
        accum.w += w;
    }

    float invw = 1.f / accum.w;
    accum.x *= invw;
    accum.y *= invw;
    accum.z *= invw;

    accum.w = 1.f;
    *pResult = toABGR(accum * 255.f);
}

//------------------------------------------------------------------------
