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

#include "Ambient.hpp"
#include "../io/OctreeRuntime.hpp"
#include "../io/AttachIO.hpp"

using namespace FW;

//------------------------------------------------------------------------
// Global variables.
//------------------------------------------------------------------------

__constant__ int4 c_input[(sizeof(AmbientInput) + sizeof(int4) - 1) / sizeof(int4)];
__device__ S32 g_warpCounter;

//------------------------------------------------------------------------
// Common helper functions.
//------------------------------------------------------------------------

__device__ inline const AmbientInput& getInput(void)
{
    return *(const AmbientInput*)c_input;
}

__device__ inline void updateCounter(PerfCounter counter, int amount = 1)
{
}

__device__ inline void updateCountersForGlobalAccess (int sizeLog2, S32* addr)
{
}

__device__ inline void updateCountersForLocalAccess  (int sizeLog2, int id)
{
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

#define F3COPY(a,b) a.x=b.x, a.y=b.y, a.z=b.z

struct Aux
{
    float  rx;
    float  ry;
    float3 normal;
    float3 orig;
    float  pad;
};

// 2 times S16 packed into a 32-bit value
__constant__ S32 c_aotable[256] = {
    0x9029cc81,0x92f8c801,0xa31fb481,0xa5eeb001,0xac40c201,0xae5ca701,0xb4aecb01,
    0xb6c9b901,0xbb009501,0xbc68a881,0xbf37bc01,0xc2bac081,0xc7a4b301,0xc90c9c81,
    0xcbdb9801,0xd012a101,0xd4488f01,0xd5b0af01,0xd664ca01,0xd87fb801,0xdcb69101,
    0xe0edb501,0xe2549881,0xe73fc701,0xe95aa601,0xed918b01,0xeef9bc81,0xf1c8a001,
    0xf54bc881,0xf5fe8e01,0xfa35bb01,0xfb9d8081,0x8384f081,0x8654ec01,0x8a8ada01,
    0x8ec1e301,0x972ffe01,0x9b65d701,0x9ccdef01,0x9f9cf801,0xa3d3d101,0xa80af501,
    0xa971d881,0xb077e601,0xb615fc81,0xb8e5e001,0xbd1bce01,0xc152fb01,0xc589d401,
    0xc9c0e901,0xcdf6dd01,0xcf5ee481,0xd22df201,0xd5560000,0xda9bfd01,0xdc02d481,
    0xded1d001,0xe308ee01,0xe8a6e081,0xeb76f401,0xefacd901,0xf3e3e501,0xf81adc01,
    0xfc51fa01,0xfe6c9401,0x02a2a901,0x06d89d01,0x0840a481,0x08f4c101,0x0b0fb201,
    0x0f469a01,0x1161c401,0x137da301,0x14e48c81,0x17b38801,0x1beabe01,0x20219701,
    0x2189b081,0x2458ac01,0x27dbc281,0x2aaac001,0x2cc5a501,0x30fc9c01,0x3533ba01,
    0x39699301,0x3ad1ac81,0x3b85c901,0x3da0a801,0x4123c481,0x460eb701,0x47759f01,
    0x4c60c301,0x4e7bb101,0x541ab881,0x5f56ab01,0x6e16c601,0x0086d301,0x01eeec81,
    0x04bde801,0x0d2bf701,0x0e92df01,0x1598f101,0x19cfcd01,0x1b36f881,0x1e06e201,
    0x223cd601,0x2673eb01,0x2ee1f601,0x3317db01,0x347fff01,0x374ee401,0x3fbced01,
    0x43f2d201,0x4829ea01,0x4dc7f481,0x5097f001,0x54cdde01,0x5904e701,0x5a6cd081,
    0x5d3bcc01,0x6172f901,0x65a8d501,0x6710e881,0x69dffc01,0x724df301,0x73b4dc81,
    0x7683d801,0x7abae101,0x810f09ff,0x86ae147f,0x897d0fff,0x8db32dff,0x91ea06ff,
    0x9352207f,0x9a5818ff,0x9e8e24ff,0x9ff6087f,0xa2c51bff,0xab3312ff,0xac9a2c7f,
    0xaf6927ff,0xb3a000ff,0xb93f1eff,0xbc0e03ff,0xc04430ff,0xc47b0cff,0xc8b221ff,
    0xcce915ff,0xd11f2aff,0xd287027f,0xd6be227f,0xddc41aff,0xe1fa23ff,0xe63108ff,
    0xea682cff,0xebcf047f,0xee9f11ff,0xf2d529ff,0xf70c02ff,0xfb432fff,0x962133ff,
    0x9c733fff,0xa6fc39ff,0xad4e48ff,0xb2ec447f,0xb5bc51ff,0xb7d736ff,0xb9f269ff,
    0xbe2942ff,0xc2606fff,0xc5e3387f,0xc6975dff,0xcacd66ff,0xcc35507f,0xcf044bff,
    0xd77244ff,0xd8d9707f,0xd98d35ff,0xdba86bff,0xdf2b3eff,0xdfdf59ff,0xe41662ff,
    0xe57d4c7f,0xe84d47ff,0xec837dff,0xf0ba56ff,0xf2226eff,0xf4f177ff,0xf874347f,
    0xf92850ff,0xfd5e74ff,0xfec6587f,0xff7a1dff,0x03af26ff,0x0517107f,0x07e60bff,
    0x105414ff,0x11bb287f,0x18c105ff,0x1cf832ff,0x1e5f1c7f,0x212f17ff,0x256520ff,
    0x299c0eff,0x2b042eff,0x320a10ff,0x37a8187f,0x3a7701ff,0x3eae25ff,0x42e50aff,
    0x471b1fff,0x4b520dff,0x50f0007f,0x53c013ff,0x57f628ff,0x5c2d1cff,0x5d95247f,
    0x606431ff,0x649b19ff,0x68d122ff,0x6a390c7f,0x6d0807ff,0x757616ff,0x7de304ff,
    0x019441ff,0x05cb65ff,0x0a024aff,0x0b697c7f,0x0c1d38ff,0x0e385fff,0x126f4dff,
    0x148a3bff,0x16a67aff,0x180d407f,0x1add53ff,0x1f1368ff,0x234a5cff,0x24b2647f,
    0x278171ff,0x2bb849ff,0x2dd337ff,0x3156547f,0x34254fff,0x364034ff,0x385c6dff,
    0x3c9346ff,0x3dfa607f,0x444c3c7f,0x450058ff,0x493764ff,0x4a9e487f,0x4d6e5bff,
    0x4f893aff,0x55db52ff,0x5e4940ff,0x66b643ff
};

//------------------------------------------------------------------------

extern "C" __global__ void ambientKernel(void)
{
    const AmbientInput& input = getInput();
    __shared__ Aux auxbuf[AMBK_BLOCK_WIDTH * AMBK_BLOCK_HEIGHT];
    volatile S32& aux0 = *((S32*)&auxbuf[AMBK_BLOCK_WIDTH * threadIdx.y]);
    volatile Aux& aux  = auxbuf[threadIdx.x + AMBK_BLOCK_WIDTH * threadIdx.y];

    const OctreeMatrices& mtx = getInput().octreeMatrices;

    // fetch first warp of work
    if (threadIdx.x == 0)
        aux0 = atomicAdd(&g_warpCounter, 1);
    int warp = aux0;
    if (warp >= input.numRequests)
        return; // terminate before starting at all

    // notice that work is being done in this warp slot
    ((S32*)input.activeWarps)[threadIdx.y + blockIdx.x * AMBK_BLOCK_HEIGHT] = 1;

    CastResult castRes;
    CastStack stack;

    // main warp loop
    for (;;)
    {
        // request index
        int ridx = warp;
        if (ridx >= input.numRequests)
            return;

        {
            AmbientRequest& req = ((AmbientRequest*)input.requestPtr)[ridx];

            // construct node position
            const U64* node     = (const U64*)getInput().rootNode;
            S32        stackPtr = CAST_STACK_DEPTH - 1;
            int        rlevel   = req.level;
            int        cidx     = 0;

            // find the node
            do
            {
                // determine child idx
                U32 smask = 1 << stackPtr;
                cidx = 0;
                if (req.pos.x & smask) cidx |= 1;
                if (req.pos.y & smask) cidx |= 2;
                if (req.pos.z & smask) cidx |= 4;

                if (stackPtr <= rlevel)
                    break;

                // move down
                U32 nodeData = *(const U32*)node;
                S32 bits = nodeData << (8-cidx);
                stack.write(stackPtr, (S32*)node, 0.0f);
                stackPtr--;
                int ofs = nodeData >> 17;
                node += (nodeData & 0x10000) ? *(const S32*)(node + ofs) : ofs;
                node += popc8(bits & 0xFF);
            }
            while (stackPtr >= 0); // always true

            // construct request position in float
            float3 rpos;
            rpos.x  = __int_as_float(req.pos.x + 0x3f800000u);
            rpos.y  = __int_as_float(req.pos.y + 0x3f800000u);
            rpos.z  = __int_as_float(req.pos.z + 0x3f800000u);

            // set up position struct
            castRes.node     = (S32*)node;
            castRes.stackPtr = stackPtr;
            castRes.childIdx = cidx;
            castRes.pos      = rpos;

            float3 orig = rpos;

            // sample color and normal at request position, adjust ray origin
            F32 vsize = __int_as_float((127 - ::min(CAST_STACK_DEPTH - rlevel, 13)) << 23);
            float4 color; // dummy
            float3 normal;
            lookupVoxelColorNormal(color, normal, castRes, stack);
                normal = normalize(normal);
                float nlen = 1.f / fmaxf3(fabsf(normal.x), fabsf(normal.y), fabsf(normal.z));
                orig += normal * (vsize * nlen);

            F3COPY(aux.normal, normal);
            F3COPY(aux.orig, orig);

            // construct 2d rotation for samples
            U32 ix = __float_as_int(rpos.x);
            U32 iy = __float_as_int(rpos.y);
            U32 iz = __float_as_int(rpos.z);
            jenkinsMix(ix, iy, iz);
            ix ^= req.level;
            float rx, ry, rlen;
            do
            {
                jenkinsMix(ix, iy, iz);
                rx = (float)ix / (4.f * (1u << 30)) * 2.f - 1.f;
                ry = (float)iy / (4.f * (1u << 30)) * 2.f - 1.f;
                rlen = rx*rx+ry*ry;
            } while (rlen > 1.f);
            rlen = rsqrtf(rlen);
            aux.rx = rx * rlen;
            aux.ry = ry * rlen;
        }

        // construct ray
        Ray ray;
        F3COPY(ray.orig, aux.orig);
        ray.orig_sz = 0.f;
        ray.dir_sz  = 0.f;

        // light vector
        float3 L = { -.4f, .5f, -.3f };
        L = normalize(L);

        // cast the ao rays
        float3 illum;
#ifdef FLIP_NORMALS
        for (int pass = 0; pass < 2; pass++)
#endif
        {
            illum = make_float3(0.f, 0.f, 0.f);
            for (int i=threadIdx.x; i < input.raysPerNode; i += 32)
            {
                // use ao table
                S32 ao32 = c_aotable[i];
                float sy = (float)ao32 * __int_as_float(0x30000000);
                ao32 <<= 16;
                float sx = (float)ao32 * __int_as_float(0x30000000);

                // rotate in 2d
                float x = aux.rx*sx + aux.ry*sy;
                float y = aux.ry*sx - aux.rx*sy;

                // construct basis for normal
                float3 normal;
                F3COPY(normal, aux.normal);
                float3 b1 = normalize(perpendicular(normal));
                float3 b2 = cross(normal, b1);

                // set ray direction
                float z = sqrtf(fabsf(1.f - x*x - y*y));
                ray.dir = x*b1 + y*b2 + z*normal;
                ray.dir *= input.rayLength;

#ifdef FLIP_NORMALS
                if (pass == 1)
                    ray.dir *= -1.0f;
#endif

                // cast the ray
                CastResult castResRay;
                CastStack  stackRay;
                castRay(castResRay, stackRay, ray);

                float ill = smoothstep(castResRay.t * 2.f - 1.f); // taper off in last 50%
                illum.x += ill;
                illum.y += ill;
                illum.z += ill;
            }

            // calculate result
            illum *= (1.f / input.raysPerNode);

            // sum over warp
            F3COPY(aux.orig, illum);
            if (!(threadIdx.x & 1))  aux.orig.x+=(&aux+ 1)->orig.x,aux.orig.y+=(&aux+ 1)->orig.y,aux.orig.z+=(&aux+ 1)->orig.z;
            if (!(threadIdx.x & 2))  aux.orig.x+=(&aux+ 2)->orig.x,aux.orig.y+=(&aux+ 2)->orig.y,aux.orig.z+=(&aux+ 2)->orig.z;
            if (!(threadIdx.x & 4))  aux.orig.x+=(&aux+ 4)->orig.x,aux.orig.y+=(&aux+ 4)->orig.y,aux.orig.z+=(&aux+ 4)->orig.z;
            if (!(threadIdx.x & 8))  aux.orig.x+=(&aux+ 8)->orig.x,aux.orig.y+=(&aux+ 8)->orig.y,aux.orig.z+=(&aux+ 8)->orig.z;
            if (!(threadIdx.x & 16)) aux.orig.x+=(&aux+16)->orig.x,aux.orig.y+=(&aux+16)->orig.y,aux.orig.z+=(&aux+16)->orig.z;

#ifdef FLIP_NORMALS
            if (auxbuf[AMBK_BLOCK_WIDTH * threadIdx.y].orig.x >= 0.1f)
                break;

            AmbientRequest& req = ((AmbientRequest*)input.requestPtr)[ridx];
            ray.orig.x = __int_as_float(req.pos.x + 0x3f800000u) * 2.0f - ray.orig.x;
            ray.orig.y = __int_as_float(req.pos.y + 0x3f800000u) * 2.0f - ray.orig.y;
            ray.orig.z = __int_as_float(req.pos.z + 0x3f800000u) * 2.0f - ray.orig.z;
#endif
        }

        // write result
        if (threadIdx.x == 0)
        {
            AmbientResult& res = ((AmbientResult*)input.resultPtr)[ridx];
            F3COPY(res.ao, aux.orig);
        }

        // fetch more work
        if (threadIdx.x == 0)
            aux0 = atomicAdd(&g_warpCounter, 1);
        warp = aux0;
    }
}

//------------------------------------------------------------------------
