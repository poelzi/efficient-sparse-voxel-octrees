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

#include "AttachIO.hpp"
#include "OctreeRuntime.hpp"

using namespace FW;

//------------------------------------------------------------------------

AttachIO::AttachIO(const Array<AttachType>& runtimeTypes)
:   m_types                 (runtimeTypes),

    m_importSlice           (NULL),

    m_trunksPerPage         (-1),
    m_pagesPerTrunkBlock    (-1),
    m_trunkBlockInfoOfs     (-1),

    m_currBlockInfo         (NULL)
{
    for (int i = 0; i < m_types.getSize(); i++)
    {
        const AttachTypeInfo& info = getAttachTypeInfo(m_types[i]);
        if (!info.allowedInRuntime)
            fail("AttachIO: %s not allowed as a runtime attachment!", info.name);

        for (int j = 0; j < i; j++)
            if (m_types[i] == m_types[j] && m_types[i] != VoidAttach)
                fail("AttachIO: Duplicate %s!", info.name);
    }
}

//------------------------------------------------------------------------

AttachIO::~AttachIO(void)
{
}

//------------------------------------------------------------------------

const AttachIO::AttachTypeInfo& AttachIO::getAttachTypeInfo(AttachType type)
{
    static const AttachTypeInfo s_infos[] =
    {
        // Name                         File    Runtime Export  SubValues   Valuesize
        { "VoidAttach",                 true,   true,   false,  false,      0 },
        { "ColorNormalDXTAttach",       true,   true,   true,   false,      sizeof(DXTNode) / sizeof(S32) },
        { "<invalid>",                  false,  false,  false,  false,      0 },
        { "ColorNormalPaletteAttach",   true,   true,   true,   true,       2 },
        { "BuildDataAttach",            true,   false,  false,  false,      0 },
        { "<invalid>",                  false,  false,  false,  false,      0 },
        { "ContourAttach",              true,   true,   true,   true,       1 },
        { "ColorNormalCornerAttach",    true,   true,   true,   true,       2 },
        { "<invalid>",                  false,  false,  false,  false,      0 },
        { "<invalid>",                  false,  false,  false,  false,      0 },
        { "AOAttach",                   true,   true,   true,   false,      2 },
    };

    FW_ASSERT(FW_ARRAY_SIZE(s_infos) == AttachType_Max);
    FW_ASSERT(type >= 0 && type < AttachType_Max);
    return s_infos[type];
}

//------------------------------------------------------------------------

int AttachIO::getNodeAlign(void)
{
    for (int i = 0; i < m_types.getSize(); i++)
        if (m_types[i] == ColorNormalDXTAttach || m_types[i] == AOAttach)
            return 4;
    return 2;
}

//------------------------------------------------------------------------

void AttachIO::beginSliceImport(const OctreeSlice* slice)
{
    m_importSlice = slice;
    m_importDXTNode = NULL;
    m_importAONode = NULL;
}

//------------------------------------------------------------------------

void AttachIO::importNodes(Array<S32>& block, const Array<ImportNode>& nodes)
{
    // Allocate BlockAttachInfos.

    int attachInfoOfs = block[0] + OctreeRuntime::BlockInfo_End;
    block[block[0] + OctreeRuntime::BlockInfo_NumAttach] = m_types.getSize();
    block.add(NULL, m_types.getSize() * OctreeRuntime::AttachInfo_End);

    // Import each attachment.

    for (int i = 0; i < m_types.getSize(); i++)
    {
        const S32* sliceAttachData = findSliceAttachData(m_importSlice, m_types[i]);
        block.add(NULL, -block.getSize() & 1); // align to qword boundary

        // Write BlockAttachInfo.

        block[attachInfoOfs + OctreeRuntime::AttachInfo_Type] = m_types[i];
        block[attachInfoOfs + OctreeRuntime::AttachInfo_Ptr] = block.getSize() - block[0];
        attachInfoOfs += OctreeRuntime::AttachInfo_End;

        // Import attachment based on its runtime type.

        const AttachTypeInfo& info = getAttachTypeInfo(m_types[i]);
        switch (m_types[i])
        {
        case VoidAttach:
            break;

        case ContourAttach:
            importContourAttach(block, nodes, sliceAttachData);
            break;

        case ColorNormalPaletteAttach:
            importPaletteAttach(block, i, info.valueSize, nodes, sliceAttachData);
            break;

        case ColorNormalCornerAttach:
            importCornerAttach(block, i, info.valueSize, nodes, sliceAttachData);
            break;

        case ColorNormalDXTAttach:
            importDXTAttach(block, i, nodes, sliceAttachData);
            break;

        case AOAttach:
            importAOAttach(block, i, nodes, sliceAttachData);
            break;

        default:
            fail("%s not supported by AttachIO::importBlock()!", info.name);
            break;
        }
    }
}

//------------------------------------------------------------------------

int AttachIO::beginSliceExport(void)
{
    m_exports.resize(m_types.getSize());
    for (int i = 0; i < m_types.getSize(); i++)
    {
        ExportAttachment& ex = m_exports[i];
        ex.attribMask.clear();
        ex.valueOfs.clear();
        ex.values.clear();
    }
    return m_types.getSize();
}

//------------------------------------------------------------------------

void AttachIO::beginNodeExport(void)
{
    for (int i = 0; i < m_types.getSize(); i++)
    {
        ExportAttachment& ex = m_exports[i];
        ex.attribMask.add(0);
        ex.valueOfs.add(ex.values.getSize());
    }
}

//------------------------------------------------------------------------

void AttachIO::exportNodeValue(AttachType type, const S32* value)
{
    ExportAttachment& ex = m_exports[findRuntimeType(type)];
    const AttachTypeInfo& info = getAttachTypeInfo(type);
    FW_ASSERT(info.allowedInExport);
    FW_ASSERT(!info.subValues);
    ex.values.add(value, info.valueSize);
}

//------------------------------------------------------------------------

void AttachIO::exportNodeSubValue(AttachType type, int idxInNode, const S32* value)
{
    ExportAttachment& ex = m_exports[findRuntimeType(type)];
    const AttachTypeInfo& info = getAttachTypeInfo(type);
    FW_ASSERT(info.allowedInExport);
    FW_ASSERT(info.subValues);
    ex.attribMask.getLast() |= 1 << idxInNode;
    ex.values.add(value, info.valueSize);
}

//------------------------------------------------------------------------

void AttachIO::endNodeExport(bool cancel)
{
    if (!cancel)
        return;

    for (int i = 0; i < m_types.getSize(); i++)
    {
        ExportAttachment& ex = m_exports[i];
        ex.attribMask.removeLast();
        ex.values.resize(ex.valueOfs.removeLast());
    }
}

//------------------------------------------------------------------------

void AttachIO::endSliceExport(OctreeSlice& slice)
{
    for (int i = 0; i < m_types.getSize(); i++)
    {
        const AttachTypeInfo& info = getAttachTypeInfo(m_types[i]);
        switch (m_types[i])
        {
        case VoidAttach:
            break;

        case ContourAttach:
        case ColorNormalPaletteAttach:
            exportPaletteAttach(slice, m_types[i], m_exports[i]);
            break;

        case ColorNormalCornerAttach:
            exportCornerAttach(slice, m_types[i], m_exports[i]);
            break;

        case ColorNormalDXTAttach:
        case AOAttach:
            slice.startAttach(m_types[i]);
            slice.getData().add(m_exports[i].values);
            slice.endAttach();
            break;

        default:
            fail("%s not supported by AttachIO::endSliceExport()!", info.name);
            break;
        }
    }
}

//------------------------------------------------------------------------

int AttachIO::layoutTrunkBlock(int trunksPerPage, int pagesPerTrunkBlock, int trunkBlockInfoOfs)
{
    m_trunksPerPage = trunksPerPage;
    m_pagesPerTrunkBlock = pagesPerTrunkBlock;
    m_trunkBlockInfoOfs = trunkBlockInfoOfs;
    m_trunkBlockNodeArraySize = trunkBlockInfoOfs >> 1;

    int ofs = trunkBlockInfoOfs + OctreeRuntime::BlockInfo_End + m_types.getSize() * OctreeRuntime::AttachInfo_End;
    m_trunkAttachDataOfs.resize(m_types.getSize());

    for (int i = 0; i < m_types.getSize(); i++)
    {
        m_trunkAttachDataOfs[i] = ofs;

        const AttachTypeInfo& info = getAttachTypeInfo(m_types[i]);
        switch (m_types[i])
        {
        case VoidAttach:
            break;

        case ContourAttach:
            ofs += m_trunksPerPage * m_pagesPerTrunkBlock * 64 * info.valueSize;
            break;

        case ColorNormalPaletteAttach:
            ofs += m_trunkBlockNodeArraySize; // PaletteNodes
            ofs += m_trunksPerPage * m_pagesPerTrunkBlock * 64 * info.valueSize; // values
            break;

        case ColorNormalCornerAttach:
            ofs += m_trunkBlockNodeArraySize * 2; // CornerNodes
            ofs += m_trunksPerPage * m_pagesPerTrunkBlock * 8 * 27 * info.valueSize; // values
            break;

        case ColorNormalDXTAttach:
        case AOAttach:
            ofs += (m_trunkBlockNodeArraySize >> 1) * info.valueSize;
            break;

        default:
            fail("%s not supported by AttachIO::layoutTrunkBlock()!", info.name);
            break;
        }
    }
    return ofs;
}

//------------------------------------------------------------------------

void AttachIO::initTrunkBlock(S32* blockData)
{
    blockData[m_trunkBlockInfoOfs + OctreeRuntime::BlockInfo_NumAttach] = m_types.getSize();
    int attachInfoOfs = m_trunkBlockInfoOfs + OctreeRuntime::BlockInfo_End;
    for (int i = 0; i < m_types.getSize(); i++)
    {
        blockData[attachInfoOfs + OctreeRuntime::AttachInfo_Type] = m_types[i];
        blockData[attachInfoOfs + OctreeRuntime::AttachInfo_Ptr] = m_trunkAttachDataOfs[i] - m_trunkBlockInfoOfs;
        attachInfoOfs += OctreeRuntime::AttachInfo_End;
    }
}

//------------------------------------------------------------------------

void AttachIO::copyNodeToSubtrunk(
    Buffer& dstBuffer, S64 dstBlockOfs, S64 dstNodeOfs, int dstPage, int dstTrunk, int dstSubtrunk,
    const S32* srcBlock, const S32* srcNode)
{
    m_currBlockInfo = NULL;
    setCurrBlock(srcBlock + srcBlock[0]);
    S64 dstNodeIdx = (dstNodeOfs - dstBlockOfs) >> 1;
    S64 srcNodeIdx = (srcNode - srcBlock) >> 1;

    for (int i = 0; i < m_types.getSize(); i++)
    {
        const AttachTypeInfo& info = getAttachTypeInfo(m_types[i]);
        S64 dstAttachOfs = dstBlockOfs + m_trunkAttachDataOfs[i];

        switch (m_types[i])
        {
        case VoidAttach:
            break;

        case ContourAttach:
            {
                U32 srcPaletteNode = srcNode[1];
                S32 dstValueOfs = getTrunkContourValueOfs(dstPage, dstTrunk, dstSubtrunk);

                U32 dstPaletteNode = ((S32)(dstAttachOfs + dstValueOfs - dstNodeOfs) << 8) | (srcPaletteNode & 0xFF);
                dstBuffer.setRange((dstNodeOfs + 1) * sizeof(S32), &dstPaletteNode, sizeof(S32));

                dstBuffer.setRange(
                    (dstAttachOfs + dstValueOfs) * sizeof(S32),
                    srcNode + (srcPaletteNode >> 8),
                    popc8(srcPaletteNode & 0xFF) * sizeof(S32));
            }
            break;

        case ColorNormalPaletteAttach:
            {
                U32 srcPaletteNode = m_currAttachData[i][srcNodeIdx];
                S32 dstValueOfs = getTrunkPaletteValueOfs(dstPage, dstTrunk, dstSubtrunk, info.valueSize);

                U32 dstPaletteNode = (dstValueOfs << 8) | (srcPaletteNode & 0xFF);
                dstBuffer.setRange((dstAttachOfs + dstNodeIdx) * sizeof(S32), &dstPaletteNode, sizeof(S32));

                dstBuffer.setRange(
                    (dstAttachOfs + dstValueOfs) * sizeof(S32),
                    m_currAttachData[i] + (srcPaletteNode >> 8),
                    popc8(srcPaletteNode & 0xFF) * sizeof(S32) * info.valueSize);
            }
            break;

        case ColorNormalCornerAttach:
            {
                U32 srcCornerNode[2];
                srcCornerNode[0] = m_currAttachData[i][srcNodeIdx * 2]; // bits
                srcCornerNode[1] = m_currAttachData[i][srcNodeIdx * 2 + 1]; // offset
                S32 dstCornerValueOfs = getTrunkCornerValueOfs(dstPage, dstTrunk, dstSubtrunk, info.valueSize);

                U32 dstCornerNode[2];
                dstCornerNode[0] = srcCornerNode[0];  // bits remain
                dstCornerNode[1] = dstCornerValueOfs; // offset is rewritten
                dstBuffer.setRange((dstAttachOfs + dstNodeIdx * 2) * sizeof(S32), dstCornerNode, 2 * sizeof(S32));

                dstBuffer.setRange(
                    (dstAttachOfs + dstCornerValueOfs) * sizeof(S32),
                    m_currAttachData[i] + srcCornerNode[1],
                    popc32(srcCornerNode[0]) * sizeof(S32) * info.valueSize);
            }
            break;

        case ColorNormalDXTAttach:
            {
                const DXTNode* src = (const DXTNode*)m_currAttachData[i] + (srcNodeIdx >> 1);
                S64 dstOfs = dstAttachOfs * sizeof(S32) + (dstNodeIdx >> 1) * sizeof(DXTNode);
                int srcShift = ((int)srcNodeIdx & 1) * 16;
                int dstShift = ((int)dstNodeIdx & 1) * 16;
                U64 dstMask = (U64)0xFFFF << (48 - dstShift);
                U64 srcMask = (U64)0xFFFF << (dstShift + 32);
                int shl = max(dstShift - srcShift, 0);
                int shr = max(srcShift - dstShift, 0);

                DXTNode dst;
                dstBuffer.getRange(&dst, dstOfs, sizeof(DXTNode));
                dst.color = (U32)src->color | (dst.color & dstMask) | (((src->color << shl) >> shr) & srcMask);
                dst.normalA = (U32)src->normalA | (dst.normalA & dstMask) | (((src->normalA << shl) >> shr) & srcMask);
                dst.normalB = (U32)src->normalB | (dst.normalB & dstMask) | (((src->normalB << shl) >> shr) & srcMask);
                dstBuffer.setRange(dstOfs, &dst, sizeof(DXTNode));
            }
            break;

        case AOAttach:
            {
                const U64* src = (const U64*)m_currAttachData[i] + (srcNodeIdx >> 1);
                S64 dstOfs = dstAttachOfs * sizeof(S32) + (dstNodeIdx >> 1) * sizeof(U64);
                int srcShift = ((int)srcNodeIdx & 1) * 24;
                int dstShift = ((int)dstNodeIdx & 1) * 24;
                U64 dstMask = (U64)0xFFFFFF << (40 - dstShift);
                U64 srcMask = (U64)0xFFFFFF << (dstShift + 16);
                int shl = max(dstShift - srcShift, 0);
                int shr = max(srcShift - dstShift, 0);

                U64 dst;
                dstBuffer.getRange(&dst, dstOfs, sizeof(U64));
                dst = (U32)(*src) | (dst & dstMask) | (((*src << shl) >> shr) & srcMask);
                dstBuffer.setRange(dstOfs, &dst, sizeof(U64));
            }
            break;

        default:
            fail("%s not supported by AttachIO::copyBlockRootToSubtrunk()!", info.name);
            break;
        }
    }
}

//------------------------------------------------------------------------

void AttachIO::copyTrunkToTrunk(
    Buffer& dstBuffer, S64 dstBlockOfs, S64 dstNodeOfs, int dstPage, int dstTrunk,
    Buffer& srcBuffer, S64 srcBlockOfs, S64 srcNodeOfs, int srcPage, int srcTrunk)
{
    S64 dstNodeIdx = (dstNodeOfs - dstBlockOfs) >> 1;
    S64 srcNodeIdx = (srcNodeOfs - srcBlockOfs) >> 1;

    for (int i = 0; i < m_types.getSize(); i++)
    {
        const AttachTypeInfo& info = getAttachTypeInfo(m_types[i]);
        S64 dstAttachOfs = dstBlockOfs + m_trunkAttachDataOfs[i];
        S64 srcAttachOfs = srcBlockOfs + m_trunkAttachDataOfs[i];

        switch (m_types[i])
        {
        case VoidAttach:
            break;

        case ContourAttach:
            {
                S32 dstValueOfs = getTrunkContourValueOfs(dstPage, dstTrunk, 0);
                S32 srcValueOfs = getTrunkContourValueOfs(srcPage, srcTrunk, 0);
                S32 contourPtr = (S32)(dstAttachOfs + dstValueOfs - dstNodeOfs) << 8;

                for (int j = 0; j < 8; j++)
                {
                    U32 paletteNode;
                    srcBuffer.getRange(&paletteNode, (srcNodeOfs + j * 2 + 1) * sizeof(S32), sizeof(S32));
                    paletteNode = contourPtr | (paletteNode & 0xFF);
                    dstBuffer.setRange((dstNodeOfs + j * 2 + 1) * sizeof(S32), &paletteNode, sizeof(S32));
                    contourPtr += (8 - 2) << 8;
                }

                dstBuffer.setRange(
                    (dstAttachOfs + dstValueOfs) * sizeof(S32),
                    srcBuffer, (srcAttachOfs + srcValueOfs) * sizeof(S32),
                    sizeof(S32) * 64);
            }
            break;

        case ColorNormalPaletteAttach:
            {
                S32 dstValueOfs = getTrunkPaletteValueOfs(dstPage, dstTrunk, 0, info.valueSize);
                S32 srcValueOfs = getTrunkPaletteValueOfs(srcPage, srcTrunk, 0, info.valueSize);

                U32 paletteNodes[8];
                srcBuffer.getRange(paletteNodes, (srcAttachOfs + srcNodeIdx) * sizeof(S32), 8 * sizeof(S32));
                S32 bias = (dstValueOfs - srcValueOfs) << 8;
                for (int j = 0; j < 8; j++)
                    paletteNodes[j] += bias;
                dstBuffer.setRange((dstAttachOfs + dstNodeIdx) * sizeof(S32), paletteNodes, 8 * sizeof(S32));

                dstBuffer.setRange(
                    (dstAttachOfs + dstValueOfs) * sizeof(S32),
                    srcBuffer, (srcAttachOfs + srcValueOfs) * sizeof(S32),
                    sizeof(S32) * 64 * info.valueSize);
            }
            break;

        case ColorNormalCornerAttach:
            {
                S32 dstCornerValueOfs = getTrunkCornerValueOfs(dstPage, dstTrunk, 0, info.valueSize);
                S32 srcCornerValueOfs = getTrunkCornerValueOfs(srcPage, srcTrunk, 0, info.valueSize);

                U32 cornerNodes[16];
                srcBuffer.getRange(cornerNodes, (srcAttachOfs + srcNodeIdx * 2) * sizeof(S32), 16 * sizeof(S32));
                for (int j = 0; j < 8; j++)
                    cornerNodes[j * 2 + 1] += dstCornerValueOfs - srcCornerValueOfs;
                dstBuffer.setRange((dstAttachOfs + dstNodeIdx * 2) * sizeof(S32), cornerNodes, 16 * sizeof(S32));

                dstBuffer.setRange(
                    (dstAttachOfs + dstCornerValueOfs) * sizeof(S32),
                    srcBuffer, (srcAttachOfs + srcCornerValueOfs) * sizeof(S32),
                    sizeof(S32) * 8 * 27 * info.valueSize);
            }
            break;

        case ColorNormalDXTAttach:
        case AOAttach:
            dstBuffer.setRange(
                dstAttachOfs * sizeof(S32) + (dstNodeIdx >> 1) * (4 * info.valueSize),
                srcBuffer, srcAttachOfs * sizeof(S32) + (srcNodeIdx >> 1) * (4 * info.valueSize),
                (4 * info.valueSize) * 4);
            break;

        default:
            fail("%s not supported by AttachIO::copyTrunkToTrunk()!", info.name);
            break;
        }
    }
}

//------------------------------------------------------------------------

void AttachIO::setCurrBlock(const S32* blockInfo)
{
    if (m_currBlockInfo == blockInfo)
        return;

    m_currBlockInfo = blockInfo;
    m_currBlockStart = OctreeRuntime::getBlockStart(blockInfo);

    FW_ASSERT(OctreeRuntime::getNumAttach(blockInfo) == m_types.getSize());
    m_currAttachData.resize(m_types.getSize());

    for (int i = 0; i < m_types.getSize(); i++)
    {
        const S32* attachInfo = OctreeRuntime::getAttachInfo(blockInfo, i);
        FW_ASSERT(OctreeRuntime::getAttachType(attachInfo) == m_types[i]);
        m_currAttachData[i] = OctreeRuntime::getAttachData(blockInfo, attachInfo);
    }
}

//------------------------------------------------------------------------

const S32* AttachIO::findSliceAttachData(const OctreeSlice* slice, AttachType type)
{
    if (slice)
        for (int i = slice->getNumAttach() - 1; i >= 0; i--)
            if (slice->getAttachType(i) == type)
                return slice->getData().getPtr(slice->getAttachOfs(i));
    return NULL;
}

//------------------------------------------------------------------------

void AttachIO::importContourAttach(Array<S32>& block, const Array<ImportNode>& nodes, const S32* sliceAttachData)
{
    // Import each node.

    for (int i = 0; i < nodes.getSize(); i++)
    {
        const ImportNode& node = nodes[i];
        U32 paletteNode = 0;
        const S32* srcValues = NULL;

        if (node.srcInRuntime)
        {
            paletteNode = node.srcInRuntime[1];
            srcValues = node.srcInRuntime + (paletteNode >> 8);
        }
        else if (node.srcInSlice != -1 && sliceAttachData)
        {
            paletteNode = sliceAttachData[node.srcInSlice];
            srcValues = sliceAttachData + (paletteNode >> 8);
        }

        S32 valueOfs = block.getSize() - node.ofsInBlock;
        if ((valueOfs & ~0xFFFFFF) != 0)
            fail("AttachIO: Too much contour data in block!");
        block[node.ofsInBlock + 1] = (valueOfs << 8) | (paletteNode & 0xFF);
        block.add(srcValues, popc8(paletteNode & 0xFF));
    }
}

//------------------------------------------------------------------------

S32 AttachIO::getTrunkContourValueOfs(int page, int trunk, int subtrunk)
{
    return ((page * m_trunksPerPage + trunk) * 8 + subtrunk) * 8;
}

//------------------------------------------------------------------------

void AttachIO::importPaletteAttach(Array<S32>& block, int attachIdx, int valueSize, const Array<ImportNode>& nodes, const S32* sliceAttachData)
{
    // Allocate PaletteNodes.

    int attachDataOfs = block.getSize();
    int nodeArraySize = block[0] >> 1;
    block.add(NULL, nodeArraySize);

    // Import each node.

    Array<S32> defaultValues;
    m_currBlockInfo = NULL;
    for (int i = 0; i < nodes.getSize(); i++)
    {
        const ImportNode& node = nodes[i];
        U32 paletteNode;
        const S32* srcValues;

        if (node.srcInRuntime)
        {
            setCurrBlock(OctreeRuntime::getBlockInfo(node.srcInRuntime));
            S64 runtimeNodeIdx = (node.srcInRuntime - m_currBlockStart) >> 1;
            paletteNode = m_currAttachData[attachIdx][runtimeNodeIdx];
            srcValues = m_currAttachData[attachIdx] + (paletteNode >> 8);
        }
        else if (node.srcInSlice != -1 && sliceAttachData)
        {
            paletteNode = sliceAttachData[node.srcInSlice];
            srcValues = sliceAttachData + (paletteNode >> 8);
        }
        else
        {
            if (!defaultValues.getSize())
            {
                defaultValues.reset(valueSize * 8);
                memset(defaultValues.getPtr(), 0, defaultValues.getNumBytes());
            }
            paletteNode = node.validMask;
            srcValues = defaultValues.getPtr();
        }

        S32 paletteNodeOfs = attachDataOfs + (node.ofsInBlock >> 1);
        S32 valueOfs = block.getSize() - attachDataOfs;
        if ((valueOfs & ~0xFFFFFF) != 0)
            fail("AttachIO: Too much palette data in block!");
        block[paletteNodeOfs] = (valueOfs << 8) | (paletteNode & 0xFF);
        block.add(srcValues, popc8(paletteNode & 0xFF) * valueSize);
    }
}

//------------------------------------------------------------------------

void AttachIO::exportPaletteAttach(OctreeSlice& slice, AttachType type, const ExportAttachment& ex)
{
    slice.startAttach(type);
    Array<S32>& data = slice.getData();
    for (int i = 0; i < ex.attribMask.getSize(); i++)
    {
        S32 ofs = ex.attribMask.getSize() + ex.valueOfs[i];
        if ((ofs & ~0xFFFFFF) != 0)
            fail("AttachIO: Too much palette data in slice!");
        data.add((ofs << 8) | ex.attribMask[i]);
    }
    data.add(ex.values);
    slice.endAttach();
}

//------------------------------------------------------------------------

S32 AttachIO::getTrunkPaletteValueOfs(int page, int trunk, int subtrunk, int valueSize)
{
    return m_trunkBlockNodeArraySize + ((page * m_trunksPerPage + trunk) * 8 + subtrunk) * 8 * valueSize;
}

//------------------------------------------------------------------------

void AttachIO::importCornerAttach(Array<S32>& block, int attachIdx, int valueSize, const Array<ImportNode>& nodes, const S32* sliceAttachData)
{
    // Allocate CornerNodes.

    int attachDataOfs = block.getSize();
    int nodeArraySizeX2 = block[0];
    block.add(NULL, nodeArraySizeX2);

    // Import each node.

    m_currBlockInfo = NULL;
    for (int i = 0; i < nodes.getSize(); i++)
    {
        const ImportNode& node = nodes[i];
        U32 cornerNode[2] = {0, 0};
        const S32* srcValues = NULL;

        if (node.srcInRuntime)
        {
            setCurrBlock(OctreeRuntime::getBlockInfo(node.srcInRuntime));
            S64 runtimeNodeIdxX2 = node.srcInRuntime - m_currBlockStart;
            cornerNode[0] = m_currAttachData[attachIdx][runtimeNodeIdxX2 + 0];
            cornerNode[1] = m_currAttachData[attachIdx][runtimeNodeIdxX2 + 1];
            srcValues = m_currAttachData[attachIdx] + cornerNode[1];
        }
        else if (node.srcInSlice != -1 && sliceAttachData)
        {
            cornerNode[0] = sliceAttachData[node.srcInSlice * 2];
            cornerNode[1] = sliceAttachData[node.srcInSlice * 2 + 1];
            srcValues = sliceAttachData + cornerNode[1];
        }

        S32 cornerNodeOfs = attachDataOfs + node.ofsInBlock;
        S32 valueOfs = block.getSize() - attachDataOfs;
        block[cornerNodeOfs + 0] = cornerNode[0];
        block[cornerNodeOfs + 1] = valueOfs;
        block.add(srcValues, popc32(cornerNode[0]) * valueSize);
    }
}

//------------------------------------------------------------------------

void AttachIO::exportCornerAttach(OctreeSlice& slice, AttachType type, const ExportAttachment& ex)
{
    slice.startAttach(type);
    Array<S32>& data = slice.getData();
    for (int i = 0; i < ex.attribMask.getSize(); i++)
    {
        S32 ofs = ex.attribMask.getSize() * 2 + ex.valueOfs[i];
        data.add(ex.attribMask[i]);
        data.add(ofs);
    }
    data.add(ex.values);
    slice.endAttach();
}

//------------------------------------------------------------------------

S32 AttachIO::getTrunkCornerValueOfs(int page, int trunk, int subtrunk, int valueSize)
{
    return m_trunkBlockNodeArraySize * 2 + ((page * m_trunksPerPage + trunk) * 8 + subtrunk) * 27 * valueSize;
}

//------------------------------------------------------------------------

void AttachIO::importDXTAttach(Array<S32>& block, int attachIdx, const Array<ImportNode>& nodes, const S32* sliceAttachData)
{
    // Allocate PaletteNodes.

    int attachDataOfs = block.getSize();
    block.add(NULL, ((block[0] + 3) >> 2) * (sizeof(DXTNode) / sizeof(S32)));

    // Import each node.

    m_currBlockInfo = NULL;
    for (int i = 0; i < nodes.getSize(); i++)
    {
        const ImportNode& node = nodes[i];
        DXTNode* dst = (DXTNode*)block.getPtr(attachDataOfs) + (node.ofsInBlock >> 2);
        if ((node.ofsInBlock & 2) != 0)
            continue;

        if (node.srcInRuntime)
        {
            setCurrBlock(OctreeRuntime::getBlockInfo(node.srcInRuntime));
            S64 runtimeNodeOfs = node.srcInRuntime - m_currBlockStart;
            *dst = *((const DXTNode*)m_currAttachData[attachIdx] + (runtimeNodeOfs >> 2));

            if ((runtimeNodeOfs & 2) != 0)
            {
                dst->color = (U32)dst->color | (dst->color >> 48) << 32;
                dst->normalA = (U32)dst->normalA | (dst->normalA >> 48) << 32;
                dst->normalB = (U32)dst->normalB | (dst->normalB >> 48) << 32;
            }
        }
        else if (node.srcInSlice != -1 && sliceAttachData)
        {
            if (!m_importDXTNode)
                m_importDXTNode = (const DXTNode*)sliceAttachData;
            *dst = *m_importDXTNode++;
        }
        else
        {
            dst->color = 0;
            dst->normalA = 0;
            dst->normalB = 0;
        }
    }
}

//------------------------------------------------------------------------

void AttachIO::importAOAttach(Array<S32>& block, int attachIdx, const Array<ImportNode>& nodes, const S32* sliceAttachData)
{
    // Allocate PaletteNodes.

    int attachDataOfs = block.getSize();
    block.add(NULL, ((block[0] + 3) >> 2) * (sizeof(U64) / sizeof(S32)));

    // Import each node.

    m_currBlockInfo = NULL;
    for (int i = 0; i < nodes.getSize(); i++)
    {
        const ImportNode& node = nodes[i];
        U64* dst = (U64*)block.getPtr(attachDataOfs) + (node.ofsInBlock >> 2);
        if ((node.ofsInBlock & 2) != 0)
            continue;

        if (node.srcInRuntime)
        {
            setCurrBlock(OctreeRuntime::getBlockInfo(node.srcInRuntime));
            S64 runtimeNodeOfs = node.srcInRuntime - m_currBlockStart;
            *dst = *((const U64*)m_currAttachData[attachIdx] + (runtimeNodeOfs >> 2));

            if ((runtimeNodeOfs & 2) != 0)
                *dst = (*dst & 0xFFFFu) | ((*dst >> 40) << 16);
        }
        else if (node.srcInSlice != -1 && sliceAttachData)
        {
            if (!m_importAONode)
                m_importAONode = (const U64*)sliceAttachData;
            *dst = *m_importAONode++;
        }
        else
            *dst = 0;
    }
}

//------------------------------------------------------------------------
