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

#include "OctreeFile.hpp"
#include "3d/Mesh.hpp"
#include "io/MeshBinaryIO.hpp"
#include "../Util.hpp"

using namespace FW;

//------------------------------------------------------------------------

OctreeFile::OctreeFile(const String& fileName, File::Mode mode, int clusterSize)
:   m_file              (fileName, mode, clusterSize, true),
    m_octreeChunkDirty  (false)
{
    switch (mode)
    {
    case File::Read:
        if (!readOctreeChunk())
            clearInternal();
        break;

    case File::Create:
        clear();
        break;

    case File::Modify:
        if (!readOctreeChunk())
            clear();
        break;

    default:
        FW_ASSERT(false);
        break;
    }
}

//------------------------------------------------------------------------

OctreeFile::~OctreeFile(void)
{
    flush();
    clearInternal();
}

//------------------------------------------------------------------------

void OctreeFile::clear(void)
{
    if (!checkWritable())
        return;

    clearInternal();
    m_file.clear();

    m_octreeChunkDirty = true;
}

//------------------------------------------------------------------------

void OctreeFile::clearSlices(void)
{
    if (!checkWritable())
        return;

    for (int i = 0; i < m_file.getNumIDs(GroupID_Slices); i++)
    {
        m_file.remove(GroupID_Slices, i);
        setSliceState(i, SliceState_Unused);
    }

    for (int i = 0; i < m_objects.getSize(); i++)
    {
        if (m_objects[i].object.rootSlice == -1)
            continue;

        m_objects[i].object.rootSlice = -1;
        m_octreeChunkDirty = true;
    }
}

//------------------------------------------------------------------------

void OctreeFile::clearSlices(int objID)
{
    FW_ASSERT(objID >= 0 && objID < m_objects.getSize());
    int rootID = m_objects[objID].object.rootSlice;
    if (!checkWritable() || rootID == -1)
        return;

    Array<S32> stack;
    while (stack.getSize())
    {
        int sliceID = stack.removeLast();
        OctreeSlice slice;
        readSlice(sliceID, slice);

        m_file.remove(GroupID_Slices, sliceID);
        setSliceState(sliceID, SliceState_Unused);

        for (int i = 0; i < slice.getNumChildEntries(); i++)
            if (slice.getChildEntry(i) >= 0)
                stack.add(slice.getChildEntry(i));
    }

    m_objects[objID].object.rootSlice = -1;
    m_octreeChunkDirty = true;
}

//------------------------------------------------------------------------

void OctreeFile::set(OctreeFile& other, int maxLevels, bool includeMeshes, bool enablePrints)
{
    FW_ASSERT(maxLevels >= 0);
    if (!checkWritable() || &other == this)
        return;

    // Flush source and clear destination.

    other.flush();
    clear();

    // Copy objects and queue root slices.

    Hash<S32, S32> sliceRemap;
    Array<Vec2i> queue;

    for (int i = 0; i < other.getNumObjects() && !hasError(); i++)
    {
        if (enablePrints)
            printf("Copying objects... %d/%d\r", i + 1, other.getNumObjects());

        OctreeFile::Object obj = other.getObject(i);
        if (other.getSliceState(obj.rootSlice) != OctreeFile::SliceState_Complete || !maxLevels)
            obj.rootSlice = -1;
        else
        {
            queue.add(Vec2i(obj.rootSlice, 0));
            obj.rootSlice = sliceRemap.add(obj.rootSlice, sliceRemap.getSize());
        }

        int dstIdx = addObject();
        setObject(dstIdx, obj);
        if (includeMeshes)
            setMesh(dstIdx, other.getMeshCopy(i));
    }

    if (enablePrints && other.getNumObjects())
    {
        if (!hasError())
            printf("Copying objects... Done.%-8s", "");
        printf("\n");
    }

    // Copy slices.

    for (int i = 0; i < queue.getSize() && !hasError(); i++)
    {
        if (enablePrints)
            printf("Copying slices... %d\r", i + 1);

        // Prefetch.

        int prefetchBytes = 0;
        for (int j = i; j < queue.getSize(); j++)
        {
            prefetchBytes += other.getSliceSize(queue[j].x);
            if (prefetchBytes > MaxPrefetchBytesTotal)
                break;

            other.readSlicePrefetch(queue[j].x);
        }

        // Read.

        OctreeSlice slice;
        other.readSlice(queue[i].x, slice);
        if (hasError())
            break;

        // Remap child IDs and queue them.

        for (int j = 0; j < slice.getNumChildEntries(); j++)
        {
            // No child => ignore.

            int child = slice.getChildEntry(j);
            if (child < 0)
                continue;

            // Child is not acceptable => cull.

            if (other.getSliceState(child) != OctreeFile::SliceState_Complete || queue[i].y + 1 >= maxLevels)
            {
                slice.setChildEntry(j, OctreeSlice::ChildEntry_NoChild);
                continue;
            }

            // Remap ID and queue.

            slice.setChildEntry(j, sliceRemap.add(child, sliceRemap.getSize()));
            queue.add(Vec2i(child, queue[i].y + 1));
        }

        // Remap ID and write.

        slice.setID(sliceRemap[queue[i].x]);
        writeSlice(slice);
    }

    if (enablePrints && queue.getSize())
    {
        if (!hasError())
            printf("Copying slices... Done.%-8s", "");
        printf("\n");
    }
}

//------------------------------------------------------------------------

void OctreeFile::flush(bool clearCache)
{
    if (m_octreeChunkDirty)
    {
        writeOctreeChunk();
        m_octreeChunkDirty = false;
    }
    m_file.flush(clearCache);
}

//------------------------------------------------------------------------

int OctreeFile::addObject(void)
{
    ObjectInfo& obj         = m_objects.add();
    obj.object.rootSlice    = -1;
    obj.mesh                = NULL;
    obj.meshValid           = false;

    m_octreeChunkDirty = true;
    return m_objects.getSize() - 1;
}

//------------------------------------------------------------------------

void OctreeFile::setObject(int objID, const Object& obj)
{
    if (!checkWritable())
        return;

    m_objects[objID].object = obj;
    m_octreeChunkDirty = true;
}

//------------------------------------------------------------------------

MeshBase* OctreeFile::getMesh(int objID)
{
    pushMemOwner("OctreeFile meshes");
    ObjectInfo& obj = m_objects[objID];
    if (!obj.meshValid)
    {
        if (m_file.exists(GroupID_Meshes, objID))
        {
            Array<U8> data;
            m_file.read(GroupID_Meshes, objID, data);
            MemoryInputStream stream(data);
            obj.mesh = importBinaryMesh(stream);
        }
        if (!hasError())
            obj.meshValid = true;
        else
        {
            delete obj.mesh;
            obj.mesh = NULL;
        }
    }
    popMemOwner();
    return obj.mesh;
}

//------------------------------------------------------------------------

MeshBase* OctreeFile::getMeshCopy(int objID)
{
    MeshBase* mesh = getMesh(objID);
    if (!mesh)
        return NULL;

    m_objects[objID].mesh = NULL;
    m_objects[objID].meshValid = false;
    return mesh;
}

//------------------------------------------------------------------------

void OctreeFile::setMesh(int objID, MeshBase* mesh)
{
    if (!checkWritable())
        return;

    ObjectInfo& obj = m_objects[objID];
    delete obj.mesh;
    obj.mesh = mesh;
    obj.meshValid = true;

    if (!mesh)
        m_file.remove(GroupID_Meshes, objID);
    else
    {
        MemoryOutputStream stream;
        exportBinaryMesh(stream, mesh);
        m_file.write(GroupID_Meshes, objID, stream.getData());
    }
}

//------------------------------------------------------------------------

OctreeFile::SliceState OctreeFile::getSliceState(int sliceID) const
{
    int idx = sliceID >> 4;
    int shift = (sliceID & 0xF) << 1;
    if (idx < 0 || idx >= m_sliceState.getSize())
        return SliceState_Unused;
    return (SliceState)((m_sliceState[idx] >> shift) & 3);
}

//------------------------------------------------------------------------

void OctreeFile::readSlice(int sliceID, OctreeSlice& slice)
{
    m_file.read(GroupID_Slices, sliceID, slice.getData());
    FW_ASSERT(!slice.getSize() || slice.getID() == sliceID);
    FW_ASSERT(!slice.getSize() || slice.getState() == getSliceState(sliceID));
}

//------------------------------------------------------------------------

void OctreeFile::readSlicePrefetch(int sliceID)
{
    m_file.readPrefetch(GroupID_Slices, sliceID, m_file.getSize(GroupID_Slices, sliceID));
}

//------------------------------------------------------------------------

bool OctreeFile::readSliceIsReady(int sliceID)
{
    return m_file.readIsReady(GroupID_Slices, sliceID, m_file.getSize(GroupID_Slices, sliceID));
}

//------------------------------------------------------------------------

void OctreeFile::writeSlice(const OctreeSlice& slice)
{
    if (!checkWritable())
        return;

    FW_ASSERT(slice.getID() >= 0);
    FW_ASSERT(slice.getState() != SliceState_Unused);
    m_file.write(GroupID_Slices, slice.getID(), slice.getData().getPtr(), slice.getData().getNumBytes());
    setSliceState(slice.getID(), slice.getState());
}

//------------------------------------------------------------------------

void OctreeFile::removeSlice(int sliceID)
{
    if (!checkWritable() || !m_file.exists(GroupID_Slices, sliceID))
        return;

    m_file.remove(GroupID_Slices, sliceID);
    setSliceState(sliceID, SliceState_Unused);

    for (int i = 0; i < m_objects.getSize(); i++)
    {
        if (m_objects[i].object.rootSlice != sliceID)
            continue;

        m_objects[i].object.rootSlice = -1;
        m_octreeChunkDirty = true;
    }
}

//------------------------------------------------------------------------

void OctreeFile::printStats(void)
{
    struct Level
    {
        S32         slices;
        S64         nodes;
        S64         children;
        S64         builtBytes;
        S32         builtSlices;
        Array<S32>  sliceIDs;

        Level(void) : slices(0), nodes(0), children(0), builtBytes(0), builtSlices(0) {}
    };

    // No objects => skip.

    if (!getNumObjects())
    {
        printf("\n");
        printf("No objects.\n");
    }

    // Flush the file to get up-to-date stats.

    flush(false);

    // Process each object.

    S64 builtBytes   = 0;
    S64 unbuiltBytes = 0;
    S64 totalBytes   = 0;

    for (int objID = 0; objID < getNumObjects(); objID++)
    {
        printf("\n");
        printf("%-11s%d\n", "Object", objID);
        const Object& object = getObject(objID);

        // Print stats for the mesh.

        S64 meshBytes = 0;
        if (m_file.exists(GroupID_Meshes, objID))
            meshBytes = m_file.getSize(GroupID_Meshes, objID);

        totalBytes += meshBytes;
        printf("%-11s%.0f megs\n", "Mesh size", (!m_file.exists(GroupID_Meshes, objID)) ? 0.0f : (F32)meshBytes * exp2(-20));

        if (meshBytes)
        {
            printf("...\r");
            MeshBase* mesh = getMesh(objID);
            FW_ASSERT(mesh);

            int numTris = 0;
            int numTextures = 0;
            S64 numTexels = 0;
            Set<const Image*> textures;

            for (int i = 0; i < mesh->numSubmeshes(); i++)
            {
                numTris += mesh->indices(i).getSize();
                const MeshBase::Material& mat = mesh->material(i);
                for (int j = 0; j < MeshBase::TextureType_Max; j++)
                {
                    const Image* image = mat.textures[j].getImage();
                    if (!image || textures.contains(image))
                        continue;

                    textures.add(image);
                    numTextures++;
                    numTexels += image->getSize().x * image->getSize().y;
                }
            }

            printf("%-11s%d\n",   "Triangles", numTris);
            printf("%-11s%d\n",   "Vertices", (!mesh) ? 0 : mesh->numVertices());
            printf("%-11s%d\n",   "Submeshes", (!mesh) ? 0 : mesh->numSubmeshes());
            printf("%-11s%d\n",   "Textures", numTextures);
            printf("%-11s%.3f\n", "MTexels", numTexels / (1024.f*1024.f));
        }

        printf("\n");

        // No octree => skip.

        if (object.rootSlice == -1)
        {
            printf("Not built.\n");
            continue;
        }

        // Print stats for each octree level.

        printf("%-6s%-7s%-7s%-7s%-8s%-7s%-7s%-8s%-9s\n", "Level", "Slices", "Built%", "KNodes", "KChild", "Branch", "MBytes", "N/Slice", "KB/Slice");
        printf("%-6s%-7s%-7s%-7s%-8s%-7s%-7s%-8s%-9s\n", "---", "---", "---", "---", "---", "---", "---", "---", "---");

        Level curr;
        Level all;
        curr.sliceIDs.add(object.rootSlice);

        for (int level = 0;; level++)
        {
            printf("...\r");

            // Accumulate over slices.

            Level next;
            for (int sliceIdx = 0; sliceIdx < curr.sliceIDs.getSize(); sliceIdx++)
            {
                // Prefetch.

                int prefetchBytes = 0;
                for (int i = sliceIdx; i < curr.sliceIDs.getSize(); i++)
                {
                    prefetchBytes += getSliceSize(curr.sliceIDs[i]);
                    if (prefetchBytes > MaxPrefetchBytesTotal)
                        break;

                    readSlicePrefetch(curr.sliceIDs[i]);
                }

                // Read.

                OctreeSlice slice;
                readSlice(curr.sliceIDs[sliceIdx], slice);
                S64 bytes = m_file.getSize(GroupID_Slices, slice.getID());

                // List children.

                for (int i = 0; i < slice.getNumChildEntries(); i++)
                    if (slice.getChildEntry(i) >= 0)
                        next.sliceIDs.add(slice.getChildEntry(i));

                // Update stats.

                curr.slices++;
                totalBytes += bytes;

                if (slice.getState() != SliceState_Complete)
                    unbuiltBytes += bytes;
                else
                {
                    builtBytes += bytes;
                    curr.builtSlices++;
                    curr.nodes += slice.getNumSplitNodes();
                    curr.builtBytes += bytes;
                    for (int i = 0; i < slice.getNumSplitNodes(); i++)
                        curr.children += popc8(slice.getNodeValidMask(i));
                }
            }

            // Accumulate total.

            if (curr.sliceIDs.getSize())
            {
                printf("%-6d", level + 1);
                all.slices      += curr.slices;
                all.nodes       += curr.nodes;
                all.children    += curr.children;
                all.builtBytes  += curr.builtBytes;
                all.builtSlices += curr.builtSlices;
            }
            else
            {
                printf("%-6s%-7s%-7s%-7s%-8s%-7s%-7s%-8s%-9s\n", "---", "---", "---", "---", "---", "---", "---", "---", "---");
                printf("%-6s", "Total");
                curr = all;
            }

            // Print.

            printf("%-7d%-7.0f%-7.0f%-8.0f%-7.2f%-7.0f%-8.0f%-9.0f\n",
                curr.slices,
                (F32)curr.builtSlices / max((F32)curr.slices, 1.0f) * 100.0f,
                (F32)curr.nodes * exp2(-10),
                (F32)curr.children * exp2(-10),
                (F32)curr.children / max((F32)curr.nodes, 1.0f),
                (F32)curr.builtBytes * exp2(-20),
                (F32)curr.nodes / max((F32)curr.builtSlices, 1.0f),
                (F32)curr.builtBytes / max((F32)curr.builtSlices, 1.0F) * exp2(-10));

            if (!curr.sliceIDs.getSize())
                break;
            curr = next;
        }
    }

    // Print general stats for the file itself.

    printf("\n");
    printf("%-17s%.0f megs\n", "Built slices", (F32)builtBytes * exp2(-20));
    printf("%-17s%.0f megs\n", "Unbuilt slices", (F32)unbuiltBytes * exp2(-20));
    printf("%-17s%.0f megs\n", "Total payload", (F32)totalBytes * exp2(-20));
    printf("%-17s%.0f megs\n", "Size on disk", (F32)m_file.getFileSize() * exp2(-20));
    printf("%-17s%.1f%%\n", "Overhead", 100.0f * (1.0f - (F32)totalBytes / (F32)m_file.getFileSize()));
    printf("%-17s%.2f\n", "Fragments/chunk", m_file.getFragmentsPerChunk());
    printf("\n");
}

//------------------------------------------------------------------------

void OctreeFile::clearInternal(void)
{
    for (int i = 0; i < m_objects.getSize(); i++)
        delete m_objects[i].mesh;
    m_objects.reset();
    m_sliceState.reset();
}

//------------------------------------------------------------------------

void OctreeFile::setSliceState(int sliceID, SliceState state)
{
    FW_ASSERT(sliceID >= 0);
    FW_ASSERT(state >= 0 && state < SliceState_Max);

    if (getSliceState(sliceID) == state)
        return;

    int idx = sliceID >> 4;
    int shift = (sliceID & 0xF) << 1;
    while (idx >= m_sliceState.getSize())
    {
        m_sliceState.add(0);
        m_octreeChunkDirty = true;
    }

    m_sliceState[idx] = (m_sliceState[idx] & ~(3 << shift)) | ((U32)state << shift);
    while (m_sliceState.getSize() && m_sliceState.getLast() == 0)
        m_sliceState.removeLast();
    m_octreeChunkDirty = true;
}

//------------------------------------------------------------------------

bool OctreeFile::readOctreeChunk(void)
{
    if (!m_file.exists(GroupID_Static, StaticChunkID_Octree))
        return false;

    // Read chunk.

    Array<U8> data;
    m_file.read(GroupID_Static, StaticChunkID_Octree, data);
    MemoryInputStream stream(data);

    // OctreeHeader.

    char formatID[9];
    stream.readFully(formatID, 8);
    formatID[8] = '\0';
    if (String(formatID) != "Octree  ")
        setError("Not an octree file!");

    S32 formatVersion;
    stream >> formatVersion;
    if (formatVersion != 1)
        setError("Unsupported octree file version!");

    S32 numObjects, numSlices;
    stream >> numObjects >> numSlices;
    if (numObjects < 0 || numSlices < 0)
        setError("Corrupt octree chunk!");

    // Array of Object.

    for (int i = 0; i < numObjects && !hasError(); i++)
    {
        ObjectInfo& obj = m_objects.add();
        obj.mesh = NULL;
        obj.meshValid = false;

        S32 numTypes;
        stream >> obj.object.objectToWorld >> obj.object.octreeToObject >> obj.object.rootSlice >> numTypes;
        if (obj.object.rootSlice < -1 || numTypes < 0)
            setError("Corrupt octree chunk!");

        for (int i = 0; i < numTypes; i++)
        {
            S32 type;
            stream >> type;
            obj.object.runtimeAttachTypes.add((AttachIO::AttachType)type);
        }
    }

    // Array of SliceStatePacked.

    m_sliceState.resize((numSlices + 0xF) >> 4);
    stream.readFully(m_sliceState.getPtr(), m_sliceState.getNumBytes());

    return (!hasError());
}

//------------------------------------------------------------------------

void OctreeFile::writeOctreeChunk(void)
{
    // OctreeHeader.

    MemoryOutputStream stream;
    stream.write("Octree  ", 8);
    stream << (S32)1 << (S32)m_objects.getSize() << (S32)(m_sliceState.getSize() << 4);

    // Array of Object.

    for (int i = 0; i < m_objects.getSize(); i++)
    {
        const Object& obj = getObject(i);
        stream << obj.objectToWorld << obj.octreeToObject << obj.rootSlice << obj.runtimeAttachTypes.getSize();
        for (int i = 0; i < obj.runtimeAttachTypes.getSize(); i++)
            stream << obj.runtimeAttachTypes[i];
    }

    // Array of SliceStatePacked.

    stream.write(m_sliceState.getPtr(), m_sliceState.getNumBytes());

    // Write chunk.

    m_file.write(GroupID_Static, StaticChunkID_Octree, stream.getData());
}

//------------------------------------------------------------------------

void OctreeSlice::init(int numChildEntries, int maxAttach, int numNodes, int numSplitNodes)
{
    FW_ASSERT(numChildEntries >= 1);
    FW_ASSERT(maxAttach >= 0);
    FW_ASSERT(numNodes >= 0);
    FW_ASSERT(numSplitNodes >= 0 && numSplitNodes <= numNodes);

    // Determine layout.

    int childEntryOfs    = SliceInfo_End;
    int attachInfoOfs    = childEntryOfs + numChildEntries;
    int nodeSplitOfs     = attachInfoOfs + maxAttach * AttachInfo_End;
    int nodeValidMaskOfs = nodeSplitOfs + (numNodes + 31) / 32;
    int attachDataOfs    = nodeValidMaskOfs + (numSplitNodes + 3) / 4;

    // Set layout.

    m_data.resize(attachDataOfs);
    set(SliceInfo_NumChildEntries, numChildEntries);
    set(SliceInfo_ChildEntryPtr, childEntryOfs);
    set(SliceInfo_NumAttach, 0);
    set(SliceInfo_AttachInfoPtr, attachInfoOfs);
    set(SliceInfo_NumNodes, numNodes);
    set(SliceInfo_NodeSplitPtr, nodeSplitOfs);
    set(SliceInfo_NumSplitNodes, numSplitNodes);
    set(SliceInfo_NodeValidMaskPtr, nodeValidMaskOfs);

    // Defaults.

    setID(0);
    setState(OctreeFile::SliceState_Complete);
    setCubePos(Vec3i(0));
    setCubeScale(OctreeFile::UnitScale);
    setNodeScale(OctreeFile::UnitScale);

    for (int i = 0; i < numChildEntries; i++)
        setChildEntry(i, ChildEntry_NoChild);
}

//------------------------------------------------------------------------

void OctreeSlice::startAttach(AttachIO::AttachType type)
{
    FW_ASSERT(type >= 0 && type < AttachIO::AttachType_Max);
    FW_ASSERT(AttachIO::getAttachTypeInfo(type).allowedInFile);

    int numAttach = getNumAttach();
    int attachInfoOfs = get(SliceInfo_AttachInfoPtr) + numAttach * AttachInfo_End;
    FW_ASSERT(attachInfoOfs + AttachInfo_End <= getNodeSplitOfs());

    set(SliceInfo_NumAttach, numAttach + 1);
    set(attachInfoOfs + AttachInfo_Type, type);
    set(attachInfoOfs + AttachInfo_Ptr, m_data.getSize());
}

//------------------------------------------------------------------------

void OctreeSlice::endAttach(void)
{
    int attachInfoOfs = getAttachInfoOfs(getNumAttach() - 1);
    set(attachInfoOfs + AttachInfo_Size, m_data.getSize() - get(attachInfoOfs + AttachInfo_Ptr));
}

//------------------------------------------------------------------------

bool OctreeSlice::isNodeSplit(int nodeIdx) const
{
    FW_ASSERT(nodeIdx >= 0 && nodeIdx < getNumNodes());
    int ofs = getNodeSplitOfs() + (nodeIdx >> 5);
    U32 mask = 1 << (nodeIdx & 0x1F);
    return ((get(ofs) & mask) != 0);
}

//------------------------------------------------------------------------

void OctreeSlice::setNodeSplit(int nodeIdx, bool isSplit)
{
    FW_ASSERT(nodeIdx >= 0 && nodeIdx < getNumNodes());
    int ofs = getNodeSplitOfs() + (nodeIdx >> 5);
    U32 mask = 1 << (nodeIdx & 0x1F);

    if (isSplit)
        set(ofs, get(ofs) | mask);
    else
        set(ofs, get(ofs) & ~mask);
}

//------------------------------------------------------------------------
