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

#include "MeshBuilder.hpp"
#include "DisplacementMap.hpp"

using namespace FW;

//------------------------------------------------------------------------

#define RANDOM_COLORS               0
#define ENFORCE_CONSISTENT_NORMALS  0
#define ALPHA_TEST_THRESHOLD        0.5f

//------------------------------------------------------------------------

MeshBuilder::ThreadState::ThreadState(MeshBuilder* builder, int idx)
:   m_builder       (builder),
    m_threadIdx     (idx),
    m_cubePos       (-1),
    m_cubeScale     (-1),
    m_nodeScale     (-1),
    m_voxelSize     (-1.0f),

    m_mesh          (NULL),
    m_filter        (NULL),
    m_shaper        (NULL),

    m_currGrid      (NULL),
    m_currVoxel     (-1),

    m_dxtNumParents (0),
    m_dxtIsInSlice  (false)
{
    FW_ASSERT(builder);
}

//------------------------------------------------------------------------

MeshBuilder::ThreadState::~ThreadState(void)
{
    endParentSlice();
    delete m_filter;
    delete m_shaper;
    delete m_mesh; // just the mesh accessor
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::beginParentSlice(const Vec3i& cubePos, int cubeScale, int nodeScale, int objectID, int numNodes)
{
    FW_UNREF(numNodes);

    // Get mesh.

    const BuilderMesh* mesh = m_builder->getMesh(objectID);
    if (!mesh)
        fail("MeshBuilder: Invalid object!");

    // Construct new mesh accessor.

    if (m_mesh)
        delete m_mesh;
    m_mesh = new BuilderMeshAccessor(mesh, m_threadIdx);

    // Read header.

    if (readBits(32) != 1)
        fail("MeshBuilder: Unsupported build data version!");

    Params params;
    params.enableVariableResolution = (readBits(32) != 0);
    params.colorDeviation           = bitsToFloat(readBits(32));
    params.normalDeviation          = bitsToFloat(readBits(32));
    params.contourDeviation         = bitsToFloat(readBits(32));
    params.filter                   = (FilterType)readBits(32);
    params.shaper                   = (ShaperType)readBits(32);

    // Setup filter.

    if (params.filter != m_params.filter)
    {
        delete m_filter;
        m_filter = NULL;
    }

    if (!m_filter)
    {
        switch (params.filter)
        {
        case Filter_Nearest:    m_filter = new BoxFilter(Vec2i(0, 0)); break;
        case Filter_NearestDXT: m_filter = new BoxFilter(Vec2i(0, 0)); break;
        case Filter_Box:        m_filter = new BoxFilter(Vec2i(0, 1)); break;
        case Filter_Pyramid:    m_filter = new PyramidFilter; break;
        default:                FW_ASSERT(false); break;
        }
    }

    // Setup shaper.

    if (params.shaper != m_params.shaper)
    {
        delete m_shaper;
        m_shaper = NULL;
    }

    if (!m_shaper)
    {
        switch (params.shaper)
        {
        case Shaper_None:   m_shaper = NULL; break;
        case Shaper_Hull:   m_shaper = new HullShaper; break;
        default:            FW_ASSERT(false); break;
        }
    }

    // Initialize state.

    int numPosBits  = max(cubeScale - nodeScale, m_builder->m_geomExpansionBits) + 1;
    m_cubePos       = cubePos;
    m_cubeScale     = cubeScale;
    m_nodeScale     = nodeScale;
    m_voxelSize     = exp2(nodeScale - 1);
    m_params        = params;
    m_currVoxel     = 0;

    endParentSlice();

    if (m_shaper)
        m_shaper->init(m_mesh, m_voxelSize, m_params.contourDeviation);

    // Read parent voxels and create child voxels.

    int numVoxels = 0;
    for (;;)
    {
        // Read flags.

        U32 parentFlags = readBits(3);
        if (parentFlags == Voxel_NonExistent)
            break;

        // Read position.

        Vec3i parentPos = cubePos;
        for (int i = 0; i < 3; i++)
            parentPos[i] += ((readBits(numPosBits) - GeomExpansion) << nodeScale);

        // Read inherited attributes.

        U32 parentInheritMask = 0;
        S32 parentAttribData[27][AttribFilter::DataItem_Max];

        if ((parentFlags & Voxel_InheritAttribs) != 0)
        {
            AttribFilter::Value corners[8];
            parentInheritMask = readBits(27);

            for (int i2 = 0; i2 < 8; i2++)
            {
                int i3 = base2ToBase3(i2) * 2;
                if ((parentInheritMask & (1 << i3)) == 0)
                    continue;

                for (int j = 0; j < AttribFilter::DataItem_Max; j++)
                    parentAttribData[i3][j] = readBits(32);
                corners[i2].decode(parentAttribData[i3]);
            }

            // Interpolate.

            for (int i = 0; i < 27; i++)
            {
                if (m_builder->m_lerpCorners[i][0] == -1)
                    continue;

                AttribFilter::Value attribs;
                for (int j = 0; m_builder->m_lerpCorners[i][j] != -1; j++)
                    attribs += corners[m_builder->m_lerpCorners[i][j]];

                Vec4f color;
                Vec3f normal;
                attribs.encode(parentAttribData[i], color, normal);
            }
        }

        // Read triangle list.

        if (readBits(1))
        {
            int num = readBits(m_mesh->getBitsPerTri());
            m_parentTris.resize(num);
            for (int i = 0; i < num; i++)
                m_parentTris[i] = readBits(m_mesh->getBitsPerTri());
        }
        addWorkIn((F32)(m_parentTris.getSize() + 1));

        // Read displacement intersections.

        m_parentDispIsect.clear();
        for (int i = 0; i < m_parentTris.getSize(); i++)
        {
            DisplacedTriangle* tri = m_mesh->getTri(m_parentTris[i]).dispTri;
            if (tri)
                tri->importIsect(m_parentDispIsect, getBitReader(), m_dispTemp);
        }

        // Read auxiliary contours.

        m_parentAuxContours.clear();
        while (readBits(1))
            m_parentAuxContours.add(readBits(32));

        // Create each child voxel.

        int firstChild = numVoxels;
        for (int childIdx = 0; childIdx < 8; childIdx++)
        {
            if (numVoxels == m_voxelHeaders.getSize())
            {
                m_voxelHeaders.add();
                m_voxelDatas.add();
            }

            createVoxel(
                numVoxels,
                childIdx,
                parentPos,
                parentFlags,
                parentInheritMask,
                parentAttribData);

            if (m_voxelHeaders[numVoxels].flags != Voxel_NonExistent)
            {
                gridSet(m_voxelHeaders[numVoxels].pos >> (nodeScale - 1), numVoxels);
                numVoxels++;
            }
        }

        // processNextParentNode() needs at least one child.

        if (numVoxels == firstChild)
            numVoxels++;
    }

    m_voxelHeaders.resize(numVoxels);
    m_voxelDatas.resize(numVoxels);
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::endParentSlice(void)
{
    int voxelCap     = 64 << 10;
    int triCap       = 64 << 10;
    int baryCap      = 256 << 10;
    int dispIsectCap = 1 << 20;

    gridClear();
    m_voxelHeaders.clear();
    m_voxelDatas.clear();
    m_voxelTris.clear();
    m_voxelNumBarys.clear();
    m_voxelBarys.clear();
    m_voxelDispIsect.clear();
    m_voxelAuxContours.clear();

    m_voxelHeaders.setCapacity(voxelCap);
    m_voxelDatas.setCapacity(voxelCap);
    m_voxelTris.setCapacity(triCap);
    m_voxelNumBarys.setCapacity(triCap);
    m_voxelBarys.setCapacity(baryCap);
    m_voxelDispIsect.setCapacity(dispIsectCap);
    m_voxelAuxContours.setCapacity(voxelCap);

    m_gridHash.reset();
    m_parentTris.reset();

    if (m_filter)
        m_filter->init(m_mesh, m_voxelSize, voxelCap);
}

//------------------------------------------------------------------------

Vec3i MeshBuilder::ThreadState::readParentNode(const Vec3i& cubePos, int cubeScale, int nodeScale)
{
    FW_UNREF(cubePos);
    FW_UNREF(cubeScale);
    FW_UNREF(nodeScale);

    Vec3i pos = SpecialParent_OutsideSlice;
    while (pos.x == SpecialParent_OutsideSlice)
        pos = processNextParentNode();

    m_childNodeExistsPtr = m_childNodeExists;
    return (pos.x == SpecialParent_NotSplit) ? -1 : pos;
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::endChildSlice(ChildSlice& cs)
{
    // Process remaining parent nodes outside the slice.

    while (m_currVoxel != m_voxelHeaders.getSize())
        processNextParentNode();
    flushDXTBlock();

    // Write build data header.

    cs.writeBits(32, 1); // version
    writeParams(cs, m_params);

    // Write voxels.

    int numPosBits = max(cs.cubeScale - cs.nodeScale, m_builder->m_geomExpansionBits) + 1;
    S32 maxPos = (1 << (cs.cubeScale - cs.nodeScale)) + GeomExpansion * 2;
    const VoxelData* prevData = NULL;

    for (int voxelIdx = 0; voxelIdx < m_voxelHeaders.getSize(); voxelIdx++)
    {
        // Non-existent => skip.

        VoxelHeader& vh = m_voxelHeaders[voxelIdx];
        if (vh.flags == Voxel_NonExistent)
            continue;

        // Outside the expanded slice => skip.

        Vec3i posBits = ((vh.pos - cs.cubePos) >> cs.nodeScale) + GeomExpansion;
        if (posBits.min() < 0 || posBits.max() >= maxPos)
            continue;

        // Determine whether to inherit attributes.

        VoxelData& vd = m_voxelDatas[voxelIdx];
        if ((vh.flags & Voxel_RefineAttribs) != 0 && vd.inheritMask != 0)
            vh.flags |= Voxel_InheritAttribs;

        // Voxel is done => clear triangle list.

        if ((vh.flags & (Voxel_RefineGeometry | Voxel_RefineAttribs)) == 0)
            vd.numTris = 0;

        // Write flags and position.

        cs.writeBits(3, vh.flags);
        for (int i = 0; i < 3; i++)
            cs.writeBits(numPosBits, posBits[i]);

        // Write inherited attributes.

        if ((vh.flags & Voxel_InheritAttribs) != 0)
        {
            cs.writeBits(27, vd.inheritMask);
            for (int i = 0; i < 8; i++)
                if ((vd.inheritMask & (1 << (base2ToBase3(i) * 2))) != 0)
                    for (int j = 0; j < AttribFilter::DataItem_Max; j++)
                        cs.writeBits(32, vd.attribData[i][j]);
        }

        // Write triangle list.

        const S32* triPtr = m_voxelTris.getPtr(vd.firstTri);
        if (prevData && vd.numTris == prevData->numTris &&
            memcmp(triPtr, m_voxelTris.getPtr(prevData->firstTri), vd.numTris * sizeof(S32)) == 0)
        {
            cs.writeBits(1, 0);
        }
        else
        {
            cs.writeBits(1, 1);
            cs.writeBits(m_mesh->getBitsPerTri(), vd.numTris);
            for (int i = 0; i < vd.numTris; i++)
                cs.writeBits(m_mesh->getBitsPerTri(), triPtr[i]);
        }

        // Write displacement intersections.

        const S32* dispIsectPtr = m_voxelDispIsect.getPtr(vd.firstDispIsect);
        for (int i = 0; i < vd.numTris; i++)
        {
            DisplacedTriangle* tri = m_mesh->getTri(triPtr[i]).dispTri;
            if (tri)
            {
                tri->exportIsect(cs.bitWriter, dispIsectPtr);
                dispIsectPtr = DisplacedTriangle::getNextIsect(dispIsectPtr);
            }
        }

        // Write auxiliary contours.

        const S32* auxContourPtr = m_voxelAuxContours.getPtr(vd.firstAuxContour);
        for (int i = 0; i < vd.numAuxContours; i++)
        {
            cs.writeBits(1, 1);
            cs.writeBits(32, auxContourPtr[i]);
        }
        cs.writeBits(1, 0);

        // Voxel done.

        addWorkOut((F32)(vd.numTris + 1));
        prevData = &vd;
    }

    cs.writeBits(4, Voxel_NonExistent);
}

//------------------------------------------------------------------------

bool MeshBuilder::ThreadState::buildChildNode(ChildSlice& cs, const Vec3i& nodePos)
{
    FW_UNREF(cs);
    FW_UNREF(nodePos);
    return *m_childNodeExistsPtr++;
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::createVoxel(
    int                 voxelIdx,
    int                 childIdx,
    const Vec3i&        parentPos,
    U32                 parentFlags,
    U32                 parentInheritMask,
    const S32           parentAttribData[27][AttribFilter::DataItem_Max])
{
    // Initialize voxel.

    VoxelHeader& vh     = m_voxelHeaders[voxelIdx];
    vh.pos              = getCubeChildPos(parentPos, m_nodeScale, childIdx);
    vh.flags            = 0;
    vh.parentFlags      = (U16)parentFlags;

    VoxelData& vd       = m_voxelDatas[voxelIdx];
    vd.numTris          = 0;
    vd.numAuxContours   = 0;
    vd.firstTri         = m_voxelTris.getSize();
    vd.firstBary        = m_voxelBarys.getSize();
    vd.firstDispIsect   = m_voxelDispIsect.getSize();
    vd.firstAuxContour  = m_voxelAuxContours.getSize();

    vd.colorLo          = 1.0f;
    vd.colorHi          = 0.0f;
    vd.normalLo         = 1.0f;
    vd.normalHi         = -1.0f;

    vd.inheritMask      = 0;
    vd.parentInheritMask = 0;

    // Parent is done => done.

    if ((parentFlags & (Voxel_RefineGeometry | Voxel_RefineAttribs)) == 0)
    {
        vh.flags = Voxel_NonExistent;
        return;
    }

    // Inherit attributes from the parent.

    if ((parentFlags & Voxel_InheritAttribs) != 0)
    {
        int childIdx3 = base2ToBase3(childIdx);
        for (int cornerIdx2 = 0; cornerIdx2 < 8; cornerIdx2++)
        {
            int cornerIdx3 = childIdx3 + base2ToBase3(cornerIdx2);
            if ((parentInheritMask & (1 << cornerIdx3)) == 0)
                continue;

            vd.parentInheritMask |= 1 << cornerIdx2;
            vd.inheritMask |= m_builder->m_childInheritMasks[childIdx][cornerIdx2];
            memcpy(vd.attribData[cornerIdx2], parentAttribData[cornerIdx3], AttribFilter::DataItem_Max * sizeof(S32));
        }
    }

    // Determine position.

    Vec3f halfSize = m_voxelSize * 0.5f;
    Vec3f lo  = Vec3f(vh.pos);
    Vec3f mid = lo + halfSize;
    Vec3f hi  = lo + m_voxelSize;

    // Clip triangles to the voxel.

    m_filter->inputBegin(voxelIdx, lo);
    const S32* dispIsectPtr = m_parentDispIsect.getPtr();

    for (int triIdx = 0; triIdx < m_parentTris.getSize(); triIdx++)
    {
        int objTriIdx = m_parentTris[triIdx];
        const BuilderMesh::Triangle& tri = m_mesh->getTri(objTriIdx);
        int baryOfs = m_voxelBarys.getSize();
        int dispIsectOfs = m_voxelDispIsect.getSize();
        F32 dispArea = 0.0f;

        // Displacement => intersect against DisplacementMap.

        if (tri.dispTri)
        {
            dispArea = tri.dispTri->intersectBox(m_voxelDispIsect, dispIsectPtr, mid, halfSize, m_dispTemp);
            dispIsectPtr = DisplacedTriangle::getNextIsect(dispIsectPtr);
            if (m_voxelDispIsect.getSize() == dispIsectOfs)
                continue;
        }

        // No displacement => intersect against the triangle.

        else
        {
            if (!isectsDeltaTriangleBox(tri.p - mid, tri.pu, tri.pv, halfSize))
                continue;

            if (tri.plo.x < lo.x || tri.phi.x > hi.x ||
                tri.plo.y < lo.y || tri.phi.y > hi.y ||
                tri.plo.z < lo.z || tri.phi.z > hi.z)
            {
                m_voxelBarys.resize(baryOfs + clipDeltaTriangleToBox(m_voxelBarys.add(NULL, 9), tri.p - mid, tri.pu, tri.pv, halfSize));
                if (m_voxelBarys.getSize() == baryOfs)
                    continue;
            }
        }

        // Need to sample attributes => accumulate.

        if ((parentFlags & Voxel_RefineAttribs) != 0 || m_params.filter == Filter_NearestDXT)
        {
            // Sample attributes.

            TextureSampler::Sample colorSample;
            DisplacedTriangle::Normal normalSample;
            sampleAttribs(colorSample, normalSample, tri,
                m_voxelBarys.getSize() - baryOfs, m_voxelBarys.getPtr(baryOfs),
                m_voxelDispIsect.getPtr(dispIsectOfs));

            // Maximum alpha is below threshold => cull triangle.

            if (colorSample.hi.w <= ALPHA_TEST_THRESHOLD)
            {
                m_voxelBarys.resize(baryOfs);
                m_voxelDispIsect.resize(dispIsectOfs);
                continue;
            }

            // Minimum alpha is below threshold => refine attributes.

            if (colorSample.lo.w <= ALPHA_TEST_THRESHOLD)
                vh.flags |= Voxel_RefineAttribs;

            // Accumulate attributes.

            m_filter->inputTriangle(objTriIdx, colorSample,
                (tri.dispTri) ? &normalSample : NULL, dispArea,
                m_voxelBarys.getSize() - baryOfs, m_voxelBarys.getPtr(baryOfs));

            // Update ranges.

            for (int i = 0; i < 3; i++)
            {
                vd.colorLo[i] = fastMin(vd.colorLo[i], colorSample.lo[i]);
                vd.colorHi[i] = fastMax(vd.colorHi[i], colorSample.hi[i]);
                vd.normalLo[i] = fastMin(vd.normalLo[i], normalSample.lo[i]);
                vd.normalHi[i] = fastMax(vd.normalHi[i], normalSample.hi[i]);
            }
        }

        // Add triangle.

        m_voxelTris.add(objTriIdx);
        m_voxelNumBarys.add((S8)(m_voxelBarys.getSize() - baryOfs));
    }

    m_filter->inputEnd();

    // No triangles => no voxel.

    vd.numTris = m_voxelTris.getSize() - vd.firstTri;
    if (!vd.numTris)
    {
        vh.flags = Voxel_NonExistent;
        return;
    }

    // Contours disabled => done.

    if (m_params.shaper == Shaper_None)
        return;

    // Transform auxiliary contours from the parent.

    vd.numAuxContours = m_parentAuxContours.getSize();
    for (int i = 0; i < vd.numAuxContours; i++)
        m_voxelAuxContours.add(xformContourToChild(m_parentAuxContours[i], childIdx));
    m_voxelAuxContours.add(0); // reserve room for one more
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::sampleAttribs(
    TextureSampler::Sample&         color,
    DisplacedTriangle::Normal&      normal,
    const BuilderMesh::Triangle&    tri,
    int                             numBary,
    const Vec2f*                    bary,
    const S32*                      dispIsect)
{
    // Determine texcoord and normal bounds.

    Vec2f texLo;
    Vec2f texHi;

    if (tri.dispTri)
        tri.dispTri->getTexAndNormal(texLo, texHi, normal, dispIsect);
    else if (!numBary)
    {
        texLo = tri.t + Vec2f(min(0.0f, tri.tu.x, tri.tv.x), min(0.0f, tri.tu.y, tri.tv.y));
        texHi = tri.t + Vec2f(max(0.0f, tri.tu.x, tri.tv.x), max(0.0f, tri.tu.y, tri.tv.y));
        normal.lo = tri.nlo;
        normal.hi = tri.nhi;
    }
    else
    {
        texLo = FW_F32_MAX;
        texHi = -FW_F32_MAX;
        normal.lo = 1.0f;
        normal.hi = -1.0f;

        for (int i = 0; i < numBary; i++)
        {
            Vec2f t = tri.t + tri.tu * bary[i].x + tri.tv * bary[i].y;
            texLo.x = min(texLo.x, t.x);
            texLo.y = min(texLo.y, t.y);
            texHi.x = max(texHi.x, t.x);
            texHi.y = max(texHi.y, t.y);

            Vec3f n = (tri.n + tri.nu * bary[i].x + tri.nv * bary[i].y).normalized();
            for (int j = 0; j < 3; j++)
            {
                normal.lo[j] = fastMin(normal.lo[j], n[j]);
                normal.hi[j] = fastMax(normal.hi[j], n[j]);
            }
        }
    }

    // Sample color.

    color.avg = tri.color;
    color.lo = tri.color;
    color.hi = tri.color;

    if (tri.colorTex)
        tri.colorTex->sampleRect(color, texLo, texHi);

    if (tri.alphaTex)
    {
        TextureSampler::Sample tmp;
        tri.alphaTex->sampleRect(tmp, texLo, texHi);
        color.avg.w = tmp.avg.y;
        color.lo.w = tmp.lo.y;
        color.hi.w = tmp.hi.y;
    }
}

//------------------------------------------------------------------------

Vec3i MeshBuilder::ThreadState::processNextParentNode(void)
{
    // Determine parent node position.

    VoxelHeader&    firstHeader     = m_voxelHeaders[m_currVoxel];
    int             voxelScale      = m_nodeScale - 1;
    Vec3i           scaledParentPos = (firstHeader.pos >> m_nodeScale) << 1;
    Vec3i           parentPos       = scaledParentPos << voxelScale;
    bool            isInSlice       = isPosInCube(parentPos, m_cubePos, m_cubeScale);

    // Not split => skip.

    if ((firstHeader.parentFlags & (Voxel_RefineGeometry | Voxel_RefineAttribs)) == 0)
    {
        m_currVoxel++;
        return (isInSlice) ? SpecialParent_NotSplit : SpecialParent_OutsideSlice;
    }

    // Notify DXT compressor of a new parent node.

    if (m_params.filter == Filter_NearestDXT)
        beginDXTParent(m_currVoxel, isInSlice);

    // Find the 4x4x4 voxel neighborhood.

    for (int i = 0; i < 64; i++)
        m_neighbors[i] = gridGet(scaledParentPos + base4ToVec(i) - 1);

    // Process each voxel in the parent node.

    for (int childIdx = 0; childIdx < 8; childIdx++)
    {
        // No voxel => skip.

        m_childNodeExists[childIdx] = false;
        if (m_currVoxel == m_voxelHeaders.getSize())
            continue;

        VoxelHeader& vh = m_voxelHeaders[m_currVoxel];
        if (vh.pos != getCubeChildPos(parentPos, m_nodeScale, childIdx))
            continue;

        // Non-existent => skip.

        int voxelIdx = m_currVoxel++;
        if (vh.flags == Voxel_NonExistent)
            continue;
        m_childNodeExists[childIdx] = true;

        // Handle DXT and palette attributes.

        if (m_params.filter == Filter_NearestDXT)
            collectDXTVoxel(voxelIdx, childIdx);
        else if (m_params.filter == Filter_Nearest)
        {
            if ((vh.parentFlags & Voxel_RefineAttribs) != 0 && attachPaletteAttribs(voxelIdx, childIdx, isInSlice))
                vh.flags |= Voxel_RefineAttribs;
        }

        // Attach contour.

        if ((vh.parentFlags & Voxel_RefineGeometry) != 0 && isInSlice)
        {
            if (m_params.shaper == Shaper_None || attachContour(vh, m_voxelDatas[voxelIdx], childIdx))
                vh.flags |= Voxel_RefineGeometry;
        }
    }

    // Attach corner attributes.

    if (m_params.filter != Filter_Nearest &&
        m_params.filter != Filter_NearestDXT &&
        (firstHeader.parentFlags & Voxel_RefineAttribs) != 0)
    {
        attachCornerAttribs(isInSlice);
    }

    // Return parent node position.

    return (isInSlice) ? parentPos : SpecialParent_OutsideSlice;
}

//------------------------------------------------------------------------

bool MeshBuilder::ThreadState::attachPaletteAttribs(int voxelIdx, int childIdx, bool isInSlice)
{
    // Sample and encode.

    S32 data[AttribFilter::DataItem_Max];
    Vec4f color;
    Vec3f normal;

    FW_ASSERT(m_filter->getExtent() == Vec2i(0, 0));
    m_filter->outputBegin();
    m_filter->outputAccumulate(voxelIdx, Vec3i(0));
    m_filter->outputEnd().encode(data, color, normal);

    // Not an extension node => attach attributes.

    if (isInSlice)
    {
#if RANDOM_COLORS
        const Vec3i& p = m_voxelHeaders[voxelIdx].pos;
        data[0] = (hashBits(p.x, p.y, p.z, m_nodeScale) & 0x1F1F1F) + 0xFFA0A0A0;
#endif
        attachNodeSubValue(AttachIO::ColorNormalPaletteAttach, childIdx, data);
    }

    // Need to refine?

    return checkPaletteAttribs(m_voxelDatas[voxelIdx], color, normal);
}

//------------------------------------------------------------------------

bool MeshBuilder::ThreadState::checkPaletteAttribs(const VoxelData& vd, const Vec4f& color, const Vec3f& normal)
{
    // Variable resolution is disabled => refine.

    if (!m_params.enableVariableResolution)
        return true;

    // Check color.

    if (componentBelow(vd.colorLo, color.getXYZ() - m_params.colorDeviation) ||
        componentBelow(color.getXYZ() + m_params.colorDeviation, vd.colorHi))
    {
        return true;
    }

    // Check normal.

    if (componentBelow(vd.normalLo, normal - m_params.normalDeviation) ||
        componentBelow(normal + m_params.normalDeviation, vd.normalHi))
    {
        return true;
    }

    return false;
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::attachCornerAttribs(bool isInSlice)
{
    // Compute attributes on the 3x3x3 corners.

    Corner corners[27];
    for (int cornerIdx3 = 0; cornerIdx3 < 27; cornerIdx3++)
    {
        int cornerIdx4 = base3ToBase4(cornerIdx3);
        Corner& c = corners[cornerIdx3];
        c.exists = false;

        const S32* forced = NULL;
        FW_ASSERT(m_filter->getExtent() == Vec2i(0, 1));
        m_filter->outputBegin();

        for (int i = 0; i < 8; i++)
        {
            int idx = m_neighbors[cornerIdx4 + base2ToBase4(i)];
            if (idx == -1)
                continue;

            const VoxelData& vd = m_voxelDatas[idx];
            c.exists = true;

            if ((vd.parentInheritMask & (1 << (i ^ 7))) == 0)
                m_filter->outputAccumulate(idx, base2ToVec(i));
            else
            {
                forced = vd.attribData[i ^ 7];
                break;
            }
        }

        if (!c.exists)
            continue;

        if (forced)
            AttribFilter::Value(forced).encode(c.data, c.color, c.normal);
        else
        {
            m_filter->outputEnd().encode(c.data, c.color, c.normal);
        }
    }

    // Attach attributes, enforcing consistent orientation of normals.

    const Vec3f& normalRef = corners[13].normal; // center
    for (int i = 0; i < 27; i++)
    {
        Corner& c = corners[i];
        if (!c.exists)
            continue;

#if ENFORCE_CONSISTENT_NORMALS
        if (normalRef.dot(c.normal) < 0.0f)
        {
            c.normal = -c.normal;
            c.data[AttribFilter::DataItem_Normal] ^= 1 << 31;
        }
#else
        FW_UNREF(normalRef);
#endif

        if (isInSlice)
        {
#if RANDOM_COLORS
            S32 data[2];
            Vec3i p = ((m_voxelHeaders[m_currVoxel - 1].pos >> m_nodeScale) << 1) + base3ToVec(i);
            data[0] = (hashBits(p.x, p.y, p.z, m_nodeScale) & 0x1F1F1F) + 0xFFA0A0A0;
            data[1] = c.data[1];
            attachNodeSubValue(AttachIO::ColorNormalCornerAttach, i, data);
#else
            attachNodeSubValue(AttachIO::ColorNormalCornerAttach, i, c.data);
#endif
        }
    }

    // Determine whether attributes still need to be refined.

    for (int voxelIdx2 = 0; voxelIdx2 < 8; voxelIdx2++)
    {
        int voxelIdx4 = base2ToBase4(voxelIdx2);
        int idx = m_neighbors[voxelIdx4 + 21];
        if (idx == -1)
            continue;

        VoxelHeader& vh = m_voxelHeaders[idx];
        VoxelData& vd = m_voxelDatas[idx];
        int voxelIdx3 = base2ToBase3(voxelIdx2);

        // Variable resolution disabled or attribs exceed deviation => refine and copy attribs.

        if (!m_params.enableVariableResolution || checkCornerAttribs(vd, corners, voxelIdx3))
        {
            vh.flags |= Voxel_RefineAttribs;
            for (int i = 0; i < 8; i++)
            {
                const Corner& c = corners[voxelIdx3 + base2ToBase3(i)];
                memcpy(vd.attribData[i], c.data, AttribFilter::DataItem_Max * sizeof(S32));
            }
        }

        // Good enough => force neighbors to inherit the shared parts.

        else
        {
            for (int i = 0; i < 27; i++)
            {
                int idx = m_neighbors[voxelIdx4 + base3ToBase4(i)];
                if (idx != -1)
                    m_voxelDatas[idx].inheritMask |= m_builder->m_neighborInheritMasks[i];
            }
        }
    }
}

//------------------------------------------------------------------------

bool MeshBuilder::ThreadState::checkCornerAttribs(const VoxelData& vd, const Corner corners[27], int cornerIdx)
{
    static const Vec2f unclippedBary[] = { Vec2f(0.0f, 0.0f), Vec2f(1.0f, 0.0f), Vec2f(0.0f, 1.0f) };

    // Find attribs and determine ranges for the first pass.

    const F32* cc[8];
    const Vec3f* cn[8];
    Vec3f colorLo = 1.0f;
    Vec3f colorHi = 0.0f;
    Vec3f normalLo = 1.0f;
    Vec3f normalHi = -1.0f;

    for (int i = 0; i < 8; i++)
    {
        const Corner& c = corners[cornerIdx + base2ToBase3(i)];
        cc[i] = c.color.getPtr();
        cn[i] = &c.normal;

        for (int j = 0; j < 3; j++)
        {
            colorLo[j] = fastMin(colorLo[j], cc[i][j]);
            colorHi[j] = fastMax(colorHi[j], cc[i][j]);
            normalLo[j] = fastMin(normalLo[j], cn[i]->get(j));
            normalHi[j] = fastMax(normalHi[j], cn[i]->get(j));
        }
    }

    // Check color.

    if (componentBelow(vd.colorLo, colorHi - m_params.colorDeviation) ||
        componentBelow(colorLo + m_params.colorDeviation, vd.colorHi))
    {
        return true;
    }

    // Check normal.

    if (componentBelow(vd.normalLo, normalHi - m_params.normalDeviation) ||
        componentBelow(normalLo + m_params.normalDeviation, vd.normalHi))
    {
        return true;
    }

    return false;
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::beginDXTParent(int voxelIdx, bool isInSlice)
{
    // Block is full or grandparent changed => flush.

    const Vec3i& pos = m_voxelHeaders[voxelIdx].pos;
    if (m_dxtNumParents == 2 || ((pos ^ m_dxtCurrPos) >> (m_nodeScale + 1)).max() || !isInSlice)
        flushDXTBlock();

    // Empty block => clear voxels.

    if (!m_dxtNumParents)
        for (int i = 0; i < 16; i++)
            m_dxtVoxels[i] = -1;

    // Add parent.

    m_dxtNumParents++;
    m_dxtCurrPos = pos;
    m_dxtIsInSlice = isInSlice;
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::collectDXTVoxel(int voxelIdx, int childIdx)
{
    m_dxtVoxels[childIdx + (m_dxtNumParents - 1) * 8] = voxelIdx;
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::flushDXTBlock(void)
{
    // DXT has no parent voxels => ignore.

    if (!m_dxtNumParents)
        return;
    m_dxtNumParents = 0;

    // Sample attributes.

    Vec3f colors[16];
    Vec3f normals[16];
    S32 indices[16];
    int num = 0;

    for (int i = 0; i < 16; i++)
    {
        if (m_dxtVoxels[i] == -1)
            continue;

        FW_ASSERT(m_filter->getExtent() == Vec2i(0, 0));
        m_filter->outputBegin();
        m_filter->outputAccumulate(m_dxtVoxels[i], Vec3i(0));
        const AttribFilter::Value& value = m_filter->outputEnd();

        colors[num] = value.getColor().getXYZ() * rcp(value.getWeight());
        normals[num] = value.getNormal().normalized();
        indices[num] = i;
        num++;
    }

    // Encode DXT block.

    AttachIO::DXTNode data;
    if (!num)
        memset(&data, 0, sizeof(data));
    else
    {
        data.color = encodeDXTColors(colors, indices, num);
        encodeDXTNormals(data.normalA, data.normalB, normals, indices, num);
    }

    // Not an extension node => attach.

    if (m_dxtIsInSlice)
        attachNodeValue(AttachIO::ColorNormalDXTAttach, (const S32*)&data);

    // Check whether the decoded attributes are good enough.

    decodeDXTColors(colors, data.color);
    decodeDXTNormals(normals, data.normalA, data.normalB);

    for (int i = 0; i < 16; i++)
    {
        int idx = m_dxtVoxels[i];
        if (idx != -1 && checkPaletteAttribs(m_voxelDatas[idx], Vec4f(colors[i], 1.0f), normals[i].normalized()))
            m_voxelHeaders[idx].flags |= Voxel_RefineAttribs;
    }
}

//------------------------------------------------------------------------

bool MeshBuilder::ThreadState::attachContour(const VoxelHeader& vh, VoxelData& vd, int childIdx)
{
    FW_ASSERT(m_shaper);
    FW_ASSERT(vd.numTris > 0);

    // Input data.

    m_shaper->setVoxel(
        Vec3f(vh.pos),
        vd.numTris,
        m_voxelTris.getPtr(vd.firstTri),
        m_voxelNumBarys.getPtr(vd.firstTri),
        m_voxelBarys.getPtr(vd.firstBary),
        m_voxelDispIsect.getPtr(vd.firstDispIsect),
        vd.numAuxContours,
        m_voxelAuxContours.getPtr(vd.firstAuxContour));

    if (m_shaper->needsNeighbors())
    {
        int childIdx4 = base2ToBase4(childIdx);
        for (int neighborIdx = 13; neighborIdx < 40; neighborIdx++)
        {
            int nvi = m_neighbors[childIdx4 + base3ToBase4(neighborIdx % 27)];
            if (nvi == -1)
                continue;

            const VoxelData& nvd = m_voxelDatas[nvi];
            m_shaper->addNeighbor(nvd.numTris, m_voxelTris.getPtr(nvd.firstTri));
        }
    }

    // Attach contour.

    S32 contour = m_shaper->getContour();
    vd.numAuxContours = m_shaper->getNumAuxContours();
    attachNodeSubValue(AttachIO::ContourAttach, childIdx, &contour);

    // Need to refine?

    return (!m_params.enableVariableResolution || m_shaper->needToRefine());
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::gridClear(void)
{
    for (int i = m_gridHash.firstSlot(); i != -1; i = m_gridHash.nextSlot(i))
        delete m_gridHash.getSlot(i).value;
    m_gridHash.clear();
    m_currGrid = NULL;
    m_currGridPos = FW_S32_MIN;
}

//------------------------------------------------------------------------

MeshBuilder::Grid* MeshBuilder::ThreadState::gridFindOrCreate(Vec3i gridPos, S32 createForVoxelIdx)
{
    if (m_currGridPos == gridPos)
        return m_currGrid;

    Grid** found = m_gridHash.search(gridPos);
    if (found)
        m_currGrid = *found;
    else
    {
        if (createForVoxelIdx == -1)
            return NULL;

        pushMemOwner("MeshBuilder grids");
        m_currGrid = m_gridHash.add(gridPos, new Grid);
        m_currGrid->base = createForVoxelIdx;
        memset(m_currGrid->ofs, -1, sizeof(m_currGrid->ofs));
        popMemOwner();
    }

    m_currGridPos = gridPos;
    return m_currGrid;
}

//------------------------------------------------------------------------

S32 MeshBuilder::ThreadState::gridGet(Vec3i scaledPos)
{
    const Grid* grid = gridFindOrCreate(scaledPos >> GridSizeLog2, -1);
    if (!grid)
        return -1;

    Vec3i p = scaledPos & (GridSize - 1);
    S16 ofs = grid->ofs[p.x][p.y][p.z];
    return (ofs == -1) ? -1 : grid->base + ofs;
}

//------------------------------------------------------------------------

void MeshBuilder::ThreadState::gridSet(Vec3i scaledPos, S32 voxelIdx)
{
    Grid* grid = gridFindOrCreate(scaledPos >> GridSizeLog2, voxelIdx);
    Vec3i p = scaledPos & (GridSize - 1);
    S32 ofs = voxelIdx - grid->base;
    FW_ASSERT((ofs >> 15) == 0);
    grid->ofs[p.x][p.y][p.z] = (S16)ofs;
}

//------------------------------------------------------------------------

MeshBuilder::MeshBuilder(OctreeFile* file)
:   BuilderBase (file)
{
    // Precalculate m_geomExpansionBits.

    m_geomExpansionBits = 0;
    while (GeomExpansion * 2 + 1 > (1 << m_geomExpansionBits))
        m_geomExpansionBits++;

    // Precalculate m_lerpCorners.

    for (int valueIdx = 0; valueIdx < 27; valueIdx++)
    {
        Vec3i cp = base3ToVec(valueIdx);
        Vec3i mask = cp - 1;
        int iter = 0;
        if (mask.abs().min() == 0 && !mask.isZero())
            for (int i = 0; i < 8; i++)
                if ((mask * (base2ToVec(i) * 2 - cp)).isZero())
                    m_lerpCorners[valueIdx][iter++] = (S8)i;
        m_lerpCorners[valueIdx][iter] = -1;
    }

    // Precalculate m_childInheritMasks.

    for (int childIdx = 0; childIdx < 8; childIdx++)
    {
        for (int cornerIdx = 0; cornerIdx < 8; cornerIdx++)
        {
            Vec3i cp = base2ToVec(cornerIdx);
            Vec3i pp = base2ToVec(childIdx) + cp - 1;
            m_childInheritMasks[childIdx][cornerIdx] = 0;

            for (int i = 0; i < 27; i++)
                if ((pp * (base3ToVec(i) - cp * 2)).isZero())
                    m_childInheritMasks[childIdx][cornerIdx] |= 1 << i;
        }
    }

    // Precalculate m_neighborInheritMasks.

    for (int neighborIdx = 0; neighborIdx < 27; neighborIdx++)
    {
        Vec3i np = base3ToVec(neighborIdx) - 1;
        m_neighborInheritMasks[neighborIdx] = 0;

        for (int i = 0; i < 27; i++)
        {
            Vec3i p = base3ToVec(i) + np * 2;
            if (p.min() >= 0 && p.max() < 3)
                m_neighborInheritMasks[neighborIdx] |= (1 << i);
        }
    }
}

//------------------------------------------------------------------------

MeshBuilder::~MeshBuilder(void)
{
    asyncAbort();
    for (int i = 0; i < m_meshes.getSize(); i++)
        delete m_meshes[i];
}

//------------------------------------------------------------------------

bool MeshBuilder::createRootSlice(
    ChildSlice&                     cs,
    Array<AttachIO::AttachType>&    attach,
    Mat4f&                          octreeToObject,
    int                             objectID,
    const Params&                   params)
{
    const BuilderMesh* bareMesh = getOrCreateMesh(objectID);
    if (!bareMesh)
        return false;

    // Use dummy mesh accessor.

    BuilderMeshAccessor* mesh = new BuilderMeshAccessor(bareMesh, 0);

    // Write header.

    cs.writeBits(32, 1); // version
    writeParams(cs, params);

    // Write flags and position.

    cs.writeBits(3, Voxel_RefineGeometry | Voxel_RefineAttribs);
    for (int i = 0; i < 3; i++)
        cs.writeBits(m_geomExpansionBits + 1, GeomExpansion);

    // Write triangle list.

    cs.writeBits(1, 1);
    cs.writeBits(mesh->getBitsPerTri(), mesh->getNumTris());
    for (int i = 0; i < mesh->getNumTris(); i++)
        cs.writeBits(mesh->getBitsPerTri(), i);

    // Write displacement rects.

    DisplacedTriangle::Temp dispTemp;
    Array<S32> dispIsect;

    for (int i = 0; i < mesh->getNumTris(); i++)
    {
        DisplacedTriangle* tri = mesh->getTri(i).dispTri;
        if (!tri)
            continue;

        dispIsect.clear();
        tri->getInitialIsect(dispIsect, dispTemp);
        tri->exportIsect(cs.bitWriter, dispIsect.getPtr());
    }

    // Write auxiliary contours.

    cs.writeBits(1, 0);

    // Write end marker.

    cs.writeBits(4, Voxel_NonExistent);

    // Setup attachments and transform.

    if (params.shaper != Shaper_None)
        attach.add(AttachIO::ContourAttach);

    switch (params.filter)
    {
    case Filter_Nearest:    attach.add(AttachIO::ColorNormalPaletteAttach); break;
    case Filter_NearestDXT: attach.add(AttachIO::ColorNormalDXTAttach); break;
    default:                attach.add(AttachIO::ColorNormalCornerAttach); break;
    }

    octreeToObject = mesh->getOctreeToObject();
    delete mesh; // this is just the accessor
    return true;
}

//------------------------------------------------------------------------

BuilderBase::ThreadState* MeshBuilder::createThreadState(int idx)
{
    return new ThreadState(this, idx);
}

//------------------------------------------------------------------------

void MeshBuilder::writeParams(ChildSlice& cs, const Params& params)
{
    cs.writeBits(32, (params.enableVariableResolution) ? 1 : 0);
    cs.writeBits(32, floatToBits(params.colorDeviation));
    cs.writeBits(32, floatToBits(params.normalDeviation));
    cs.writeBits(32, floatToBits(params.contourDeviation));
    cs.writeBits(32, params.filter);
    cs.writeBits(32, params.shaper);
}

//------------------------------------------------------------------------

void MeshBuilder::prepareTask(Task& task)
{
    getOrCreateMesh(task.objectID);
}

//------------------------------------------------------------------------

const BuilderMesh* MeshBuilder::getMesh(int objectID)
{
    BuilderMesh* mesh;
    m_lock.enter();
    mesh = m_meshes[objectID];
    m_lock.leave();
    FW_ASSERT(mesh);
    return mesh;
}

//------------------------------------------------------------------------

const BuilderMesh* MeshBuilder::getOrCreateMesh(int objectID)
{
    m_lock.enter();

    while (objectID >= m_meshes.getSize())
        m_meshes.add(NULL);

    BuilderMesh* mesh = m_meshes[objectID];
    if (!mesh)
    {
        MeshBase* foreignMesh = getFile()->getMeshCopy(objectID);
        if (foreignMesh)
        {
            mesh = new BuilderMesh(foreignMesh); // transfers foreightMesh ownership
            m_meshes[objectID] = mesh;
        }
    }

    m_lock.leave();
    return mesh;
}

//------------------------------------------------------------------------
