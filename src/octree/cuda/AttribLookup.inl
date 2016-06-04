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

//------------------------------------------------------------------------

__device__ inline float3    decodeRawNormal         (U32 value);
__device__ inline float3    decodeDXTColor          (U64 block, int texelIdx);
__device__ inline float3    decodeDXTNormal         (U64 blockA, U64 blockB, int texelIdx);
__device__ inline float     decodeAO                (U64 block, int texelIdx);

__device__ void             lookupVoxelColorNormal  (float4& colorRes, float3& normalRes, const CastResult& castRes, const CastStack& stack);
__device__ void             lookupVoxelAO           (float& res, const CastResult& castRes, const CastStack& stack);

//------------------------------------------------------------------------

__device__ inline float3 decodeRawNormal(U32 value)
{
    S32 sign = (S32)value >> 31;
    F32 t = (F32)(sign ^ 0x7fffffff);
    F32 u = (F32)((S32)value << 3);
    F32 v = (F32)((S32)value << 18);

    float3 result = { t, u, v };
    if ((value & 0x20000000) != 0)
        result.x = v, result.y = t, result.z = u;
    else if ((value & 0x40000000) != 0)
        result.x = u, result.y = v, result.z = t;
    return result;
}

//------------------------------------------------------------------------

__constant__ F32 c_dxtColorCoefs[4] =
{
    1.0f / (F32)(1 << 24),
    0.0f,
    2.0f / (F32)(3 << 24),
    1.0f / (F32)(3 << 24),
};

__device__ inline float3 decodeDXTColor(U64 block, int texelIdx)
{
    U32 head = (U32)block;
    U32 bits = (U32)(block >> 32);

    F32 c0 = c_dxtColorCoefs[(bits >> (texelIdx * 2)) & 3];
    F32 c1 = 1.0f / (F32)(1 << 24) - c0;

    return make_float3(
        c0 * (F32)(head << 16) + c1 * (F32)head,
        c0 * (F32)(head << 21) + c1 * (F32)(head << 5),
        c0 * (F32)(head << 27) + c1 * (F32)(head << 11));
}

//------------------------------------------------------------------------

__constant__ F32 c_dxtNormalCoefs[4] = {
    -1.0f,
    -1.0f / 3.0f,
    +1.0f / 3.0f,
    +1.0f,
};

__device__ inline float3 decodeDXTNormal(U64 blockA, U64 blockB, int texelIdx)
{
    U32 headBase = (U32)blockA;
    U32 headUV   = (U32)blockB;
    U32 bitsU    = (U32)(blockA >> 32);
    U32 bitsV    = (U32)(blockB >> 32);

    int shift = texelIdx * 2;
    F32 cu = c_dxtNormalCoefs[(bitsU >> shift) & 3];
    F32 cv = c_dxtNormalCoefs[(bitsV >> shift) & 3];

    cu *= __int_as_float(((headUV & 15) + (127 + 3 - 13)) << 23);
    cv *= __int_as_float((((headUV >> 16) & 15) + (127 + 3 - 13)) << 23);

    float3 base = decodeRawNormal(headBase);
    return make_float3(
        base.x + cu * (F32)(S32)(headUV << 16) + cv * (F32)(S32)headUV,
        base.y + cu * (F32)(S32)(headUV << 20) + cv * (F32)(S32)(headUV << 4),
        base.z + cu * (F32)(S32)(headUV << 24) + cv * (F32)(S32)(headUV << 8));
}

//------------------------------------------------------------------------

__constant__ F32 c_dxtAOCoefs[8] = {
    0.f,
    1.f / ((F32)(1 << 24) * 255.f * 7.f),
    2.f / ((F32)(1 << 24) * 255.f * 7.f),
    3.f / ((F32)(1 << 24) * 255.f * 7.f),
    4.f / ((F32)(1 << 24) * 255.f * 7.f),
    5.f / ((F32)(1 << 24) * 255.f * 7.f),
    6.f / ((F32)(1 << 24) * 255.f * 7.f),
    1.f / ((F32)(1 << 24) * 255.f)
};

__device__ inline float decodeAO(U64 block, int texelIdx)
{
    F32 c0 = c_dxtAOCoefs[((U32)(block >> (texelIdx * 3 + 16))) & 7];
    F32 c1 = c_dxtAOCoefs[7] - c0;
    return c0 * (F32)((U32)block << 16) + c1 * (F32)((U32)block << 24);
}

//------------------------------------------------------------------------
// Uncompressed attribute lookup.
//------------------------------------------------------------------------

#ifdef VOXELATTRIB_PALETTE

__device__ void lookupVoxelColorNormal(float4& colorRes, float3& normalRes, const CastResult& castRes, const CastStack& stack)
{
    U32 px = __float_as_int(castRes.pos.x);
    U32 py = __float_as_int(castRes.pos.y);
    U32 pz = __float_as_int(castRes.pos.z);

    // current position in tree
    S32* node  = castRes.node;
    int  cidx  = castRes.childIdx;
    int  level = castRes.stackPtr;

    // start here
    S32* pageHeader   = (S32*)((S32)node & -OctreeRuntime::PageBytes);
    S32* blockInfo    = pageHeader + *pageHeader;
    S32* blockStart   = blockInfo + blockInfo[OctreeRuntime::BlockInfo_BlockPtr];
    S32* attachInfos  = blockInfo + OctreeRuntime::BlockInfo_End;
    S32* attachInfo   = attachInfos + OctreeRuntime::AttachInfo_End * AttachSlot_Attribute;
    S32* attachData   = blockInfo + attachInfo[OctreeRuntime::AttachInfo_Ptr];
    U32  paletteNode  = attachData[(node - blockStart) >> 1];

    // while node has no color, loop
    while (!((paletteNode >> cidx) & 1))
    {
        level++;
        if (level >= CAST_STACK_DEPTH)
        {
            colorRes = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            normalRes = make_float3(1.0f, 0.0f, 0.0f);
        }

        F32 tmax;
        node = stack.read(level, tmax);
        cidx = 0;
        if ((px & (1 << level)) != 0) cidx |= 1;
        if ((py & (1 << level)) != 0) cidx |= 2;
        if ((pz & (1 << level)) != 0) cidx |= 4;

        // update
        pageHeader   = (S32*)((S32)node & -OctreeRuntime::PageBytes);
        blockInfo    = pageHeader + *pageHeader;
        blockStart   = blockInfo + blockInfo[OctreeRuntime::BlockInfo_BlockPtr];
        attachInfos  = blockInfo + OctreeRuntime::BlockInfo_End;
        attachInfo   = attachInfos + OctreeRuntime::AttachInfo_End * AttachSlot_Attribute;
        attachData   = blockInfo + attachInfo[OctreeRuntime::AttachInfo_Ptr];
        paletteNode  = attachData[(node - blockStart) >> 1];
    }

    // found, return it
    S32* pAttach = attachData + (paletteNode >> 8) + popc8(paletteNode & ((1 << cidx) - 1)) * 2;
    colorRes = fromABGR(pAttach[0]);
    normalRes = decodeRawNormal(pAttach[1]);
}

#endif

//------------------------------------------------------------------------
// Interpolated attribute lookup.
//------------------------------------------------------------------------

#ifdef VOXELATTRIB_CORNER

__device__ void lookupVoxelColorNormal(float4& colorRes, float3& normalRes, const CastResult& castRes, const CastStack& stack)
{
    U32 px = __float_as_int(castRes.pos.x);
    U32 py = __float_as_int(castRes.pos.y);
    U32 pz = __float_as_int(castRes.pos.z);

    // current position in tree
    S32* node = castRes.node;
    int level = castRes.stackPtr;

    // start here
    S32* pageHeader   = (S32*)((S32)node & -OctreeRuntime::PageBytes);
    S32* blockInfo    = pageHeader + *pageHeader;
    S32* blockStart   = blockInfo + blockInfo[OctreeRuntime::BlockInfo_BlockPtr];
    S32* attachInfos  = blockInfo + OctreeRuntime::BlockInfo_End;
    S32* attachInfo   = attachInfos + OctreeRuntime::AttachInfo_End * AttachSlot_Attribute;
    S32* attachData   = blockInfo + attachInfo[OctreeRuntime::AttachInfo_Ptr];
    U32  paletteNode  = attachData[node - blockStart];

    // move upwards until there is a child node with corner colors
    for(;;)
    {
        // find bits and data pointer
        U32 bits = paletteNode;

        // calculate subcube index
        int subIdx = 0;
        U32 lmask = 1 << level;
        if (px & lmask) subIdx += 1;
        if (py & lmask) subIdx += 3;
        if (pz & lmask) subIdx += 9;

        // reposition bits
        U32 bitsOrig = bits;
        bits >>= subIdx;

        // this is a hit if at least some bits for this subcube are nonzero
        if (bits & 0x361b)
        {
            // get and adjust data pointer
            S32* pAttach = attachData + attachData[(node - blockStart) + 1];
            pAttach += 2 * popc16(bitsOrig & ((1<<subIdx)-1));

            // construct lerp factors
            float fx1 = (F32)(px << (32-level)) * (.25f / (F32)(0x40000000u));
            float fy1 = (F32)(py << (32-level)) * (.25f / (F32)(0x40000000u));
            float fz1 = (F32)(pz << (32-level)) * (.25f / (F32)(0x40000000u));
            float fx0 = 1.f - fx1;
            float fy0 = 1.f - fy1;
            float fz0 = 1.f - fz1;

            // process the eight corners
            // optimized to take advantage of always having all 8 corners

#define ACCUM_COLOR_NORMAL(fx,fy,fz) do {           \
            F32 w = (fx)*(fy)*(fz);                 \
            U32 c = pAttach[0];                     \
            color += w*fromABGR(c);                 \
            float3 n = decodeRawNormal(pAttach[1]); \
            normal += w*normalize(n);               \
            pAttach += 2;                           \
} while (0);

            float4 color  = {0, 0, 0, 0};
            float3 normal = {0, 0, 0};
            ACCUM_COLOR_NORMAL(fx0,fy0,fz0);
            ACCUM_COLOR_NORMAL(fx1,fy0,fz0);
            if (bits & 0x0004) pAttach += 2;
            ACCUM_COLOR_NORMAL(fx0,fy1,fz0);
            ACCUM_COLOR_NORMAL(fx1,fy1,fz0);
            pAttach += 2 * popc8((bits & 0x01e0) >> 5);
            ACCUM_COLOR_NORMAL(fx0,fy0,fz1);
            ACCUM_COLOR_NORMAL(fx1,fy0,fz1);
            if (bits & 0x0800) pAttach += 2;
            ACCUM_COLOR_NORMAL(fx0,fy1,fz1);
            ACCUM_COLOR_NORMAL(fx1,fy1,fz1);

#undef ACCUM_COLOR_NORMAL

            // done
            colorRes  = color;
            normalRes = normal;
            return;
        }

        // move upwards
        level++;
        if (level >= CAST_STACK_DEPTH)
        {
            colorRes = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            normalRes = make_float3(1.0f, 0.0f, 0.0f);
            return;
        }

        F32 tmax;
        node = stack.read(level, tmax);

        // update
        S32* pageHeader2 = (S32*)((S32)node & -OctreeRuntime::PageBytes);
        if (pageHeader != pageHeader2)
        {
            pageHeader = pageHeader2;
            S32* blockInfo2 = pageHeader + *pageHeader;
            if (blockInfo != blockInfo2)
            {
                blockInfo    = blockInfo2;
                blockStart   = blockInfo + blockInfo[OctreeRuntime::BlockInfo_BlockPtr];
                attachInfos  = blockInfo + OctreeRuntime::BlockInfo_End;
                attachInfo   = attachInfos + OctreeRuntime::AttachInfo_End * AttachSlot_Attribute;
                attachData   = blockInfo + attachInfo[OctreeRuntime::AttachInfo_Ptr];
            }
        }
        paletteNode = attachData[node - blockStart];
    }
}

#endif

//------------------------------------------------------------------------
// DXT-compressed attribute lookup.
//------------------------------------------------------------------------

#ifdef VOXELATTRIB_DXT

__device__ void lookupVoxelColorNormal(float4& colorRes, float3& normalRes, const CastResult& castRes, const CastStack& stack)
{
    // Find DXTNode.

    S32* pageHeader  = (S32*)((CUdeviceptr)castRes.node & -(CUdeviceptr)OctreeRuntime::PageBytes);
    S32* blockInfo   = pageHeader + *pageHeader;
    S32* blockStart  = blockInfo + blockInfo[OctreeRuntime::BlockInfo_BlockPtr];
    S32* attachInfos = blockInfo + OctreeRuntime::BlockInfo_End;
    S32* attachInfo  = attachInfos + OctreeRuntime::AttachInfo_End * AttachSlot_Attribute;
    S32* attachData  = blockInfo + attachInfo[OctreeRuntime::AttachInfo_Ptr];
    U64* dxtBlock    = (U64*)(attachData + ((castRes.node - blockStart) >> 2) * 6);

    // Fetch.

    U64 colorBlock = dxtBlock[0];
    U64 normalBlockA = dxtBlock[1];
    U64 normalBlockB = dxtBlock[2];

    // Decode.

    int texelIdx = castRes.childIdx | (((castRes.node - pageHeader) & 2) << 2);
    float3 tmp = decodeDXTColor(colorBlock, texelIdx);
    colorRes = make_float4(tmp.x, tmp.y, tmp.z, 255.0f);
    normalRes = decodeDXTNormal(normalBlockA, normalBlockB, texelIdx);
}

#endif

//------------------------------------------------------------------------
// DXT-compressed ambient occlusion lookup.
//------------------------------------------------------------------------

__device__ void lookupVoxelAO(float& res, const CastResult& castRes, const CastStack& stack)
{
    // Find DXTNode.

    S32* pageHeader  = (S32*)((CUdeviceptr)castRes.node & -(CUdeviceptr)OctreeRuntime::PageBytes);
    S32* blockInfo   = pageHeader + *pageHeader;
    S32* blockStart  = blockInfo + blockInfo[OctreeRuntime::BlockInfo_BlockPtr];
    S32* attachInfos = blockInfo + OctreeRuntime::BlockInfo_End;
    S32* attachInfo  = attachInfos + OctreeRuntime::AttachInfo_End * AttachSlot_AO;
    S32* attachData  = blockInfo + attachInfo[OctreeRuntime::AttachInfo_Ptr];
    U64* dxtBlock    = (U64*)(attachData + ((castRes.node - blockStart) >> 2) * 2);

    // Fetch.

    U64 block = dxtBlock[0];

    // Decode.

    int texelIdx = castRes.childIdx | (((castRes.node - pageHeader) & 2) << 2);
    res = decodeAO(block, texelIdx);
#ifdef ENABLE_TWEAKS
    res = 1.0f + (res - 1.0f) * getInput().tweakParameters.aoBlend;
#endif
}

//------------------------------------------------------------------------

