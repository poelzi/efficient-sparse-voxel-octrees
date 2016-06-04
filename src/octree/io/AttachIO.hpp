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
#include "base/Defs.hpp"
#if !FW_CUDA
#   include "base/Array.hpp"
#   include "gpu/Buffer.hpp"
#endif

namespace FW
{
//------------------------------------------------------------------------

class OctreeSlice;

//------------------------------------------------------------------------

class AttachIO
{
public:
    enum AttachType
    {
        VoidAttach                  = 0,    // Empty attachment slot.
        ColorNormalDXTAttach        = 1,    // Compressed attributes.
        ColorNormalPaletteAttach    = 3,    // Uncompressed attributes.
        BuildDataAttach             = 4,    // Internal builder data for unbuilt slices.
        ContourAttach               = 6,    // Contours.
        ColorNormalCornerAttach     = 7,    // Interpolated attributes.
        AOAttach                    = 10,   // Compressed ambient occlusion.

        AttachType_Max
    };

#if !FW_CUDA
    struct AttachTypeInfo
    {
        const char*             name;
        bool                    allowedInFile;
        bool                    allowedInRuntime;
        bool                    allowedInExport;
        bool                    subValues;
        S32                     valueSize;
    };

    struct ImportNode
    {
        const S32*              srcInRuntime;       // NULL if none
        S32                     srcInSlice;         // -1 if none
        U8                      validMask;
        U8                      nonLeafMask;

        U8                      numParentChildren;  // number of children in the parent node
        U8                      idxInParent;
        S32                     firstChild;
        S32                     ofsInBlock;
    };

    struct ExportAttachment
    {
        Array<U32>              attribMask;
        Array<S32>              valueOfs;
        Array<S32>              values;
    };

    struct DXTNode
    {
        U64                     color;
        U64                     normalA;
        U64                     normalB;
    };

public:
                                AttachIO                (const Array<AttachType>& runtimeTypes);
                                ~AttachIO               (void);

    static const AttachTypeInfo& getAttachTypeInfo      (AttachType type);
    const Array<AttachType>&    getRuntimeTypes         (void) const            { return m_types; }
    int                         findRuntimeType         (AttachType type) const { return m_types.indexOf(type); }
    int                         getNodeAlign            (void); // 4 for DXT, 2 otherwise

    // Import.

    void                        beginSliceImport        (const OctreeSlice* slice);
    void                        importNodes             (Array<S32>& block, const Array<ImportNode>& nodes);

    // Export.

    int                         beginSliceExport        (void);
    void                        beginNodeExport         (void);
    void                        exportNodeValue         (AttachType type, const S32* value);
    void                        exportNodeSubValue      (AttachType type, int idxInNode, const S32* value); // childIdx must be ascending
    void                        endNodeExport           (bool cancel);
    void                        endSliceExport          (OctreeSlice& slice);

    // Trunks.

    int                         layoutTrunkBlock        (int trunksPerPage, int pagesPerTrunkBlock, int trunkBlockInfoOfs);
    void                        initTrunkBlock          (S32* blockData);

    void                        copyNodeToSubtrunk      (Buffer& dstBuffer, S64 dstBlockOfs, S64 dstNodeOfs, int dstPage, int dstTrunk, int dstSubtrunk,
                                                         const S32* srcBlock, const S32* srcNode);

    void                        copyTrunkToTrunk        (Buffer& dstBuffer, S64 dstBlockOfs, S64 dstNodeOfs, int dstPage, int dstTrunk,
                                                         Buffer& srcBuffer, S64 srcBlockOfs, S64 srcNodeOfs, int srcPage, int srcTrunk);

private:

    // Generic helpers.

    void                        setCurrBlock            (const S32* blockInfo);
    const S32*                  findSliceAttachData     (const OctreeSlice* slice, AttachType type);

    // ContourAttach.

    void                        importContourAttach     (Array<S32>& block, const Array<ImportNode>& nodes, const S32* sliceAttachData);
    S32                         getTrunkContourValueOfs (int page, int trunk, int subtrunk);

    // XxxPaletteAttach.

    void                        importPaletteAttach     (Array<S32>& block, int attachIdx, int valueSize, const Array<ImportNode>& nodes, const S32* sliceAttachData);
    void                        exportPaletteAttach     (OctreeSlice& slice, AttachType type, const ExportAttachment& ex);
    S32                         getTrunkPaletteValueOfs (int page, int trunk, int subtrunk, int valueSize);

    // XxxCornerAttach.

    void                        importCornerAttach      (Array<S32>& block, int attachIdx, int valueSize, const Array<ImportNode>& nodes, const S32* sliceAttachData);
    void                        exportCornerAttach      (OctreeSlice& slice, AttachType type, const ExportAttachment& ex);
    S32                         getTrunkCornerValueOfs  (int page, int trunk, int subtrunk, int valueSize);

    // ColorNormalDXTAttach.

    void                        importDXTAttach         (Array<S32>& block, int attachIdx, const Array<ImportNode>& nodes, const S32* sliceAttachData);

    // AOAttach.

    void                        importAOAttach          (Array<S32>& block, int attachIdx, const Array<ImportNode>& nodes, const S32* sliceAttachData);

private:
                                AttachIO                (const AttachIO& other); // forbidden
    AttachIO&                   operator=               (const AttachIO& other); // forbidden

private:
    Array<AttachType>           m_types;

    const OctreeSlice*          m_importSlice;
    const DXTNode*              m_importDXTNode;
    const U64*                  m_importAONode;

    Array<ExportAttachment>     m_exports;

    S32                         m_trunksPerPage;
    S32                         m_pagesPerTrunkBlock;
    S32                         m_trunkBlockInfoOfs;
    S32                         m_trunkBlockNodeArraySize;
    Array<S32>                  m_trunkAttachDataOfs;

    const S32*                  m_currBlockInfo;
    const S32*                  m_currBlockStart;
    Array<const S32*>           m_currAttachData;
#endif // !FW_CUDA
};

//------------------------------------------------------------------------
/*

VoidAttach (type 0, file/runtime)
    0

//------------------------------------------------------------------------

ColorNormalDXTAttach (type 1, file/runtime)
    0       n*6     struct  array of DXTNode (file: one for each pair of nodes with the same parent,
                                              runtime: indexed with Node qword pointer divided by two, relative to the Block)
    ?

DXTNode
    0       2       struct  DXT1
    2       4       struct  DXTNormals
    6

DXT1
    0.27    .5      bits    r1: red of the second reference color
    0.21    .6      bits    g1: green of the second reference color
    0.16    .5      bits    b1: blue of the second reference color
    0.11    .5      bits    r0: red of the first reference color
    0.5     .6      bits    g0: green of the first reference color
    0.0     .5      bits    b0: blue of the first reference color
    1.16    .16     bits    bits1: two bits per child cube of the second node
    1.0     .16     bits    bits0: two bits per child cube of the first node
    2

DXTNormals
    0       1       struct  base: Normal
    1.16    .16     bits    ubits1: two bits per child cube of the second node
    1.0     .16     bits    ubits0: two bits per child cube of the first node
    2.28    .4      bits    vx: signed, v.x = (F32)vx * exp2(vw - 13)
    2.24    .4      bits    vy: signed, v.y = (F32)vy * exp2(vw - 13)
    2.20    .4      bits    vz: signed, v.z = (F32)vz * exp2(vw - 13)
    2.16    .4      bits    vexp: unsigned exponent
    2.12    .4      bits    ux: signed, u.x = (F32)ux * exp2(uw - 13)
    2.8     .4      bits    uy: signed, u.y = (F32)uy * exp2(uw - 13)
    2.4     .4      bits    uz: signed, u.z = (F32)uz * exp2(uw - 13)
    2.0     .4      bits    uexp: unsigned exponent
    3.16    .16     bits    vbits1: two bits per child cube of the second node
    3.0     .16     bits    vbits0: two bits per child cube of the first node
    4

//------------------------------------------------------------------------

ColorNormalPaletteAttach (type 3, file/runtime)
    0       n*1     struct  array of PaletteNode (file: SliceInfo.numSplitNodes, runtime: indexed with Node qword pointer relative to the Block)
    ?       n*2     struct  array of ColorNormal (as many as needed)
    ?

PaletteNode
    0.8     .24     bits    ptr: unsigned pointer to the first value, relative to XxxPaletteAttach
    0.0     .8      bits    attribMask: whether each child has an associated value
    1

ColorNormal
    0       1       struct  Color
    1       1       struct  Normal
    2

Color
    0.24    .8      bits    alpha
    0.16    .8      bits    red
    0.8     .8      bits    green
    0.0     .8      bits    blue
    1

Normal
    0.31    .1      bits    sign: negate main axis
    0.29    .2      bits    axis: n[axis] = (sign) ? -1 : +1
    0.14    .15     bits    u: signed, n[(axis + 1) % 3] = (F32)u * exp2(-14)
    0.0     .14     bits    v: signed, n[(axis + 2) % 3] = (F32)v * exp2(-13)
    1

//------------------------------------------------------------------------

BuildDataAttach (type 4, file)
    See BuilderBase.h

//------------------------------------------------------------------------

ContourAttach (type 6, file/runtime)
    0       n*1     struct  array of Contour (as many as needed)
    ?

Contour
    0.25    .7      bits    thickness = (hi - lo) * 4/3 (unsigned)
    0.18    .7      bits    position = (hi + lo) * 2/3 (signed)
    0.12    .6      bits    nx (signed)
    0.6     .6      bits    ny (signed)
    0.0     .6      bits    nz (signed)
    1

//------------------------------------------------------------------------

ColorNormalCornerAttach (type 7, file/runtime)
    0       n*2     struct  array of CornerNode (file: SliceInfo.numSplitNodes, runtime: indexed with Node qword pointer relative to the Block)
    ?       n*2     struct  array of ColorNormal (as many as needed)
    ?

CornerNode
    0       1       int     attribMask: 3x3x3 bitmask telling whether each value exists
    1       1       int     ptr: unsigned pointer to the first value, relative to XxxCornerAttach
    2

*/
//------------------------------------------------------------------------
}
