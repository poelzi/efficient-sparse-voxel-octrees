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

#include "BuilderMesh.hpp"
#include "DisplacementMap.hpp"
#include "../io/OctreeFile.hpp"

using namespace FW;

//------------------------------------------------------------------------

BuilderMesh::BuilderMesh(const MeshBase* foreignMesh)
:   m_mesh(0)
{
    // Convert mesh to PNT.
    pushMemOwner("BuilderMesh");

    m_mesh = new Mesh<VertexPNT>;
    m_mesh->set(*foreignMesh); // take a copy
    delete foreignMesh; // we own this so we can delete it
    foreignMesh = 0;

    // Count triangles and find maximum displacement.

    int numTris = 0;
    int numDispTris = 0;
    F32 maxDisp = 0.0f;

    for (int i = 0; i < m_mesh->numSubmeshes(); i++)
    {
        const MeshBase::Material& mat = m_mesh->material(i);
        numTris += m_mesh->indices(i).getSize();
        if (mat.textures[MeshBase::TextureType_Displacement].exists())
        {
            numDispTris += m_mesh->indices(i).getSize();
            maxDisp = max(maxDisp, -mat.displacementBias, mat.displacementCoef + mat.displacementBias);
        }
    }

    // Calculate transform.

    Vec3f lo, hi;
    m_mesh->getBBox(lo, hi);
    m_octreeToObject =
        Mat4f::translate((lo + hi) * 0.5f) *
        Mat4f::scale(Vec3f((hi - lo).max() + maxDisp * 2.0f)) *
        Mat4f::translate(Vec3f(-0.5f));

    // This maps everything to range 0 .. 2^23 (= OctreeFile::UnitScale)

    m_xform = Mat4f::scale(Vec3f(exp2(OctreeFile::UnitScale))) * m_octreeToObject.inverted();

    // Calculate number of bits per triangle index.

    m_bitsPerTri = 0;
    while (numTris >= (1 << m_bitsPerTri))
        m_bitsPerTri++;

    // Remap vertices.

    Array<S32> vmap(NULL, m_mesh->numVertices());
    {
        Hash<Vec3f, S32> vertexHash;
        for (int i = 0; i < m_mesh->numVertices(); i++)
        {
            const Vec3f& pos = m_mesh->vertex(i).p;
            S32* found = vertexHash.search(pos);
            if (found)
                vmap[i] = *found;
            else
            {
                vmap[i] = i;
                vertexHash.add(pos, i);
            }
        }
    }

    // Collect edges.

    Set<Vec2i> edges;
    for (int i = 0; i < m_mesh->numSubmeshes(); i++)
    {
        const Array<Vec3i>& inds = m_mesh->indices(i);
        for (int j = 0; j < inds.getSize(); j++)
        {
            for (int k = 0; k < 3; k++)
            {
                Vec2i edge(vmap[inds[j][k]], vmap[inds[j][(k == 2) ? 0 : k + 1]]);
                if (!edges.contains(edge))
                    edges.add(edge);
            }
        }
    }

    // Construct triangle map.

    pushMemOwner("BuilderMesh.triMap");
    int triIdx = 0;
    m_triMap.reset(numTris);
    for (int i = 0; i < m_mesh->numSubmeshes(); i++)
    {
        const Array<Vec3i>& inds = m_mesh->indices(i);
        for (int j = 0; j < inds.getSize(); j++)
        {
            const Vec3i& t = inds[j];

            U32 bmask = 0;
            for (int k = 0; k < 3; k++)
            {
                Vec2i edge(vmap[t[(k == 2) ? 0 : k + 1]], vmap[t[k]]);
                if (!edges.contains(edge))
                    bmask |= 1 << k;
            }

            TriangleEntry& te = m_triMap[triIdx++];
            te.submesh        = i;
            te.indexInSubmesh = j;
            te.batch          = 0;
            te.setIndexInBatch(0);
            te.setBoundaryMask(bmask);
        }
    }
    popMemOwner();

    // Clear unnecessary hashes etc.

    vmap.reset();
    edges.reset();

    // Construct texture and displacement map hashes.

    pushMemOwner("BuilderMesh.hashes");
    for (int i = 0; i < m_mesh->numSubmeshes(); i++)
    {
        const MeshBase::Material& mat = m_mesh->material(i);

        // Update texture hash.

        int colorIdx = hashTexture(m_texHash, mat.textures[MeshBase::TextureType_Diffuse]);
        if (colorIdx == m_textures.getSize())
            m_textures.add(new TextureSampler(mat.textures[MeshBase::TextureType_Diffuse]));

        int alphaIdx = hashTexture(m_texHash, mat.textures[MeshBase::TextureType_Alpha]);
        if (alphaIdx == m_textures.getSize())
            m_textures.add(new TextureSampler(mat.textures[MeshBase::TextureType_Alpha]));

        // Update displacement hash.

        int dispIdx = hashTexture(m_dispHash, mat.textures[MeshBase::TextureType_Displacement]);
        if (dispIdx == m_dispMaps.getSize())
        {
//          FW::printf("add disp %d: '%s' (%016I64x)\n", m_dispMaps.getSize(), mat.textures[MeshBase::TextureType_Displacement].getID().getPtr(), mat.textures[MeshBase::TextureType_Displacement].getImage());
            m_dispMaps.add(new DisplacementMap(
                mat.textures[MeshBase::TextureType_Displacement],
                mat.displacementCoef * m_xform.m00 / 255.0f,
                mat.displacementBias * m_xform.m00));
        }
    }
    popMemOwner();
//  validateTextures();

    // Calculate triangle centers.

    Array<Vec3f> tCenter;
    tCenter.reset(numTris);
    for (int i=0; i < numTris; i++)
    {
        const Vec3i& tri = m_mesh->indices(m_triMap[i].submesh)[m_triMap[i].indexInSubmesh];
        Vec3f c = m_mesh->vertex(tri[0]).p;
        c += m_mesh->vertex(tri[1]).p;
        c += m_mesh->vertex(tri[2]).p;
        c *= (1.f/3.f);
        tCenter[i] = c;
    }

    // Subdivide triangle bulk recursively until small enough batches are found.

    FW::printf("BuilderMesh: Subdividing triangles into batches\n");
    subdivideTriangles(0, numTris, tCenter);
    FW::printf("BuilderMesh: %d triangle batches constructed\n", m_batches.getSize());

    // Set globals.

    m_numTris = numTris;
    m_batchLRUHead = 0;
    m_batchLRUTail = 0;
    m_numExpandedBatches = 0;
    m_lockedBatches.reset(MaxThreads);
    for (int i=0; i < MaxThreads; i++)
    {
        LockedBatches& le = m_lockedBatches[i];
        le.list.reset(MaxLockedBatchesPerThread);
        le.count = 0;
        le.head  = 0;
        le.tail  = 0;
    }

    // Go to simple mode if few enough batches.
    if (m_batches.getSize() <= MaxExpandedBatches)
    {
        FW::printf("BuilderMesh: Few enough batches, pre-expanding everything\n");
        for (int i=0; i < m_batches.getSize(); i++)
        {
            expandBatch(m_batches[i]);
            m_batches[i]->lockMask = (U32)(-1);
        }
    }
    popMemOwner();
}

//------------------------------------------------------------------------

BuilderMesh::~BuilderMesh(void)
{
    delete m_mesh;
    for (int i = 0; i < m_textures.getSize(); i++)
        delete m_textures[i];
    for (int i = 0; i < m_dispMaps.getSize(); i++)
        delete m_dispMaps[i];
    for (int i = 0; i < m_batches.getSize(); i++)
        delete m_batches[i];
}

//------------------------------------------------------------------------

int BuilderMesh::hashTexture(Hash<const Image*, S32>& hash, Texture tex)
{
    if (!tex.exists())
        return -1;

    S32* found = hash.search(tex.getImage());
    if (found)
        return *found;

    return hash.add(tex.getImage(), hash.getSize());
}

//------------------------------------------------------------------------

void BuilderMesh::subdivideTriangles(int firstTri, int numTris, Array<Vec3f>& tCenter)
{
    if (numTris <= MaxTriangleBatchSize)
        constructBatch(firstTri, numTris);
    else
    {
        // calculate bounding box
        Vec3f bbMin = tCenter[firstTri];
        Vec3f bbMax = bbMin;
        for (int i = firstTri + 1; i < firstTri + numTris; i++)
        {
            Vec3f c = tCenter[i];
            for (int j = 0; j < 3; j++)
            {
                bbMin[j] = fastMin(bbMin[j], c[j]);
                bbMax[j] = fastMax(bbMax[j], c[j]);
            }
        }

        // split along longest axis

        Vec3f diag = (bbMax - bbMin);
        int axis = 0;
        if (fabsf(diag.y) > fabsf(diag.x)) axis = 1;
        if (fabsf(diag.z) > fabsf(diag[axis])) axis = 2;
        float spos = .5f*(bbMin[axis] + bbMax[axis]);

        // partition as in quicksort

        int sidx = 0;
        for (int i = firstTri; i < firstTri + numTris; i++)
        {
            if (tCenter[i][axis] < spos)
            {
                swap(tCenter[i],  tCenter[firstTri + sidx]);
                swap(m_triMap[i], m_triMap[firstTri + sidx]);
                sidx++;
            }
        }

        if (sidx == 0 || sidx == numTris)
        {
            // failed to split
            FW::printf("BuilderMesh: Split failed, batch with %d triangles constructed\n", numTris);
            constructBatch(firstTri, numTris);
            return;
        }

        subdivideTriangles(firstTri, sidx, tCenter);
        subdivideTriangles(firstTri + sidx, numTris - sidx, tCenter);
    }
}

//------------------------------------------------------------------------

void BuilderMesh::constructBatch(int firstTri, int numTris)
{
    // Count displaced triangles.

    pushMemOwner("BuilderMesh.batches");
    int numDispTris = 0;
    for (int i = firstTri; i < firstTri + numTris; i++)
    {
        const MeshBase::Material& mat = m_mesh->material(m_triMap[i].submesh);
        if (hashTexture(m_dispHash, mat.textures[MeshBase::TextureType_Displacement]) != -1)
            numDispTris++;
    }

    // Init batch and update triangle map.

    Batch* batch = new Batch;
    m_batches.add(batch);

    batch->expanded = false;
    batch->numDispTris = numDispTris;
    batch->prevLRU = 0;
    batch->nextLRU = 0;
    batch->firstTri = firstTri;
    batch->numTris  = numTris;
    batch->lockMask = 0;
    for (int i = 0; i < numTris; i++)
    {
        int idx = i + firstTri;
        m_triMap[idx].batch = batch;
        m_triMap[idx].setIndexInBatch(i);
    }
    popMemOwner();
}

//------------------------------------------------------------------------

void BuilderMesh::expandBatch(Batch* batch)
{
    if (batch->expanded)
        return;

    m_numExpandedBatches++;

    pushMemOwner("BuilderMesh.expand");
    batch->expanded = true;
    batch->tris.reset(batch->numTris);
    batch->dispTris.reset(batch->numDispTris);
    popMemOwner();

//  validateTextures();

    // Construct triangles.

    int dispTriIdx = 0;
    for (int i=0; i < batch->numTris; i++)
    {
        TriangleEntry& te = m_triMap[batch->firstTri + i];
        Vec3i inds = m_mesh->indices(te.submesh)[te.indexInSubmesh];
        const MeshBase::Material& mat = m_mesh->material(te.submesh);

        Triangle& tri = batch->tris[i];

        VertexPNT v[3];
        tri.plo = exp2(OctreeFile::UnitScale);
        tri.phi = 0.0f;
        tri.nlo = 1.0f;
        tri.nhi = -1.0f;

        // Transform vertices.

        for (int k = 0; k < 3; k++)
        {
            v[k] = m_mesh->vertex(inds[k]);
            v[k].p = m_xform * v[k].p;
            v[k].n = v[k].n.normalized();

            for (int l = 0; l < 3; l++)
            {
                tri.plo[l] = fastMin(tri.plo[l], v[k].p[l]);
                tri.phi[l] = fastMax(tri.phi[l], v[k].p[l]);
                tri.nlo[l] = fastMin(tri.nlo[l], v[k].n[l]);
                tri.nhi[l] = fastMax(tri.nhi[l], v[k].n[l]);
            }
        }

        // Store vertices.

        tri.p           = v[0].p;
        tri.pu          = v[1].p - v[0].p;
        tri.pv          = v[2].p - v[0].p;

        tri.n           = v[0].n;
        tri.nu          = v[1].n - v[0].n;
        tri.nv          = v[2].n - v[0].n;

        tri.t           = v[0].t;
        tri.tu          = v[1].t - v[0].t;
        tri.tv          = v[2].t - v[0].t;

        // Store attributes.

        tri.color       = mat.diffuse;
        tri.geomNormal  = tri.pu.cross(tri.pv);
        tri.avgNormal   = (v[0].n + v[1].n + v[2].n) * (1.0f / 3.0f);
        tri.area        = tri.geomNormal.length();

        F32 weight = max(tri.area * tri.color.w, FW_MIN_ATTRIB_WEIGHT);
        tri.average.setWeight(weight);
        tri.average.setColor(tri.color * weight);
        tri.average.setNormal(tri.avgNormal * weight);

        tri.boundaryMask = te.getBoundaryMask();

        // Textures.

        int colorIdx = hashTexture(m_texHash, mat.textures[MeshBase::TextureType_Diffuse]);
        tri.colorTex = (colorIdx == -1) ? NULL : m_textures[colorIdx];

        int alphaIdx = hashTexture(m_texHash, mat.textures[MeshBase::TextureType_Alpha]);
        tri.alphaTex = (alphaIdx == -1) ? NULL : m_textures[alphaIdx];

        // Displacement.

        int dispIdx = hashTexture(m_dispHash, mat.textures[MeshBase::TextureType_Displacement]);
        if (dispIdx == -1 || abs(tri.tu.cross(tri.tv)) < 1.0e-8f)
            tri.dispTri = 0;
        else
        {
            tri.dispTri = batch->dispTris.getPtr(dispTriIdx++);
            if (dispIdx >= m_dispMaps.getSize())
            {
                fail("missing disp %d (%016I64x)\n", dispIdx, mat.textures[MeshBase::TextureType_Displacement].getImage());
            }
            tri.dispTri->set(m_dispMaps[dispIdx], tri.p, tri.pu, tri.pv, tri.n, tri.nu, tri.nv, tri.t, tri.tu, tri.tv);
        }
    }
}

//------------------------------------------------------------------------

void BuilderMesh::collapseBatch(Batch* batch)
{
    if (!batch->expanded)
        return;

    m_numExpandedBatches--;

    // remove from LRU list
    if (batch->prevLRU) batch->prevLRU->nextLRU = batch->nextLRU;
    if (batch->nextLRU) batch->nextLRU->prevLRU = batch->prevLRU;
    if (m_batchLRUHead == batch) m_batchLRUHead = batch->nextLRU;
    if (m_batchLRUTail == batch) m_batchLRUTail = batch->prevLRU;
    batch->prevLRU = 0;
    batch->nextLRU = 0;

    // collapse
    batch->expanded = false;
    batch->tris.reset();
    batch->dispTris.reset();
}

//------------------------------------------------------------------------

void BuilderMesh::validateLRUList(void) const
{
    Batch* p = m_batchLRUHead;
    while (p)
    {
        Batch* q = p->nextLRU;
        if (q)
        {
            if (q->prevLRU != p)
                fail("LRU list fail 1");
        } else
            if (p != m_batchLRUTail)
                fail("LRU list fail 2");
        p = q;
    }
}

void BuilderMesh::validateTextures(void) const
{
    for (int i=0; i < m_mesh->numSubmeshes(); i++)
    {
        const MeshBase::Material& mat = m_mesh->material(i);
        if (mat.textures[MeshBase::TextureType_Diffuse].exists())
        {
            const Image* img = mat.textures[MeshBase::TextureType_Diffuse].getImage();
            if (!m_texHash.contains(img))
                fail("BuilderMesh::m_texHash invalid (missing %016I64x)", (U64)img);
        }
        if (mat.textures[MeshBase::TextureType_Alpha].exists())
        {
            const Image* img = mat.textures[MeshBase::TextureType_Alpha].getImage();
            if (!m_texHash.contains(img))
                fail("BuilderMesh::m_texHash invalid (missing %016I64x)", (U64)img);
        }
        if (mat.textures[MeshBase::TextureType_Displacement].exists())
        {
            const Image* img = mat.textures[MeshBase::TextureType_Displacement].getImage();
            if (!m_dispHash.contains(img))
                fail("BuilderMesh::m_dispHash invalid (missing %016I64x)", (U64)img);
        }
    }
}

//------------------------------------------------------------------------

void BuilderMesh::lockBatch(int tid, Batch* batch)
{
    if (!batch)
        return;

    if (batch->lockMask == 0)
    {
        // remove from LRU list to avoid collapse
        if (batch->prevLRU) batch->prevLRU->nextLRU = batch->nextLRU;
        if (batch->nextLRU) batch->nextLRU->prevLRU = batch->prevLRU;
        if (m_batchLRUHead == batch) m_batchLRUHead = batch->nextLRU;
        if (m_batchLRUTail == batch) m_batchLRUTail = batch->prevLRU;
        batch->prevLRU = 0;
        batch->nextLRU = 0;
    }

    LockedBatches& le = m_lockedBatches[tid];
    le.list[le.head++] = batch;
    le.count++;
    if (le.head == MaxLockedBatchesPerThread)
        le.head = 0;
    batch->lockMask |= (1<<tid);
}

void BuilderMesh::unlockBatch(int tid)
{
    LockedBatches& le = m_lockedBatches[tid];
    if (le.count == 0)
        return;

    Batch* batch = le.list[le.tail++];
    le.count--;
    if (le.tail == MaxLockedBatchesPerThread)
        le.tail = 0;

    batch->lockMask &= ~(1<<tid);
    if (batch->lockMask == 0)
    {
        // place in head of LRU list
        batch->nextLRU = m_batchLRUHead;
        if (m_batchLRUHead)  m_batchLRUHead->prevLRU = batch;
        if (!m_batchLRUTail) m_batchLRUTail = batch;
        m_batchLRUHead = batch;
    }
}

const BuilderMesh::Triangle& BuilderMesh::getTriProper(Batch* batch, int i, int tid)
{
    // enter critical section, unlock currently locked batch if at limit
    m_lock.enter();
    if (m_lockedBatches[tid].count == MaxLockedBatchesPerThread)
        unlockBatch(tid);

    // if our batch is not expanded, we need to expand it
    if (!batch->expanded)
    {
        // if we're out of space, collapse the LRU batch (if any unlocked batches exist)
        if (m_numExpandedBatches >= MaxExpandedBatches && m_batchLRUTail)
            collapseBatch(m_batchLRUTail);
        // now expand our batch
        expandBatch(batch);
    }

    // lock the batch for us now
    lockBatch(tid, batch);
    m_lock.leave();

    // access!
    return batch->tris[i];
}

void BuilderMesh::freeThreadSlot(int tid)
{
    m_lock.enter();

    // unlock all locked batches
    while (m_lockedBatches[tid].count)
        unlockBatch(tid);

    m_lock.leave();
}
