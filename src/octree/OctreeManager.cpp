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

#include "OctreeManager.hpp"
#include "build/MeshBuilder.hpp"

using namespace FW;

//------------------------------------------------------------------------

S32 OctreeManager::s_numTmpFileIDs  = 0;
S32 OctreeManager::s_maxTmpFileIDs  = 0;
Array<S32> OctreeManager::s_freeTmpFileIDs;

//------------------------------------------------------------------------

OctreeManager::OctreeManager(RenderMode renderMode)
:   m_maxBuilderThreads     (FW_S32_MAX),

    m_renderMode            (renderMode),
    m_dynamicLoad           (true),
    m_dynamicBuild          (true),
    m_maxLevels             (OctreeFile::UnitScale),

    m_tmpFileID             (-1),
    m_file                  (NULL),

    m_cpuRuntime            (NULL),
    m_cudaRuntime           (NULL),
    m_cudaRenderer          (NULL),

    m_loadSliceID           (-1),
    m_loadSliceBytesDisk    (0),
    m_loadSliceBytesMemory  (0),

    m_loadBytesTotal        (0.0f),

    m_frameDeltaAvg         (0.001f),
    m_frameTimeAvg          (0.001f),
    m_updateTimeAvg         (0.0f),
    m_renderTimeAvg         (0.0f),
    m_loadBytesAvg          (0.0f)
{
    for (int i = 0; i < BuilderType_Max; i++)
        m_builders[i] = NULL;
}

//------------------------------------------------------------------------

OctreeManager::~OctreeManager(void)
{
    unloadFile(true);
    destroyRuntime();
    delete m_cudaRenderer;
}

//------------------------------------------------------------------------

void OctreeManager::setRenderMode(RenderMode renderMode)
{
    if (m_renderMode == renderMode)
        return;

    m_renderMode = renderMode;
    m_loadSliceID = -1;
}

//------------------------------------------------------------------------

OctreeFile* OctreeManager::getFile(void)
{
    if (!m_file)
    {
        m_tmpFileID = allocateTmpFileID();
        m_file = new OctreeFile(sprintf("tmp%d.oct", m_tmpFileID), File::Create);
    }
    return m_file;
}

//------------------------------------------------------------------------

OctreeRuntime* OctreeManager::getRuntime(void)
{
    switch (m_renderMode)
    {
    case RenderMode_Mesh:
        return NULL;

    case RenderMode_Cuda:
        if (!m_cudaRuntime && CudaModule::isAvailable())
            m_cudaRuntime = new OctreeRuntime(MemoryManager::Mode_Cuda);
        return m_cudaRuntime;

    default:
        FW_ASSERT(false);
        return NULL;
    }
}

//------------------------------------------------------------------------

CudaRenderer* OctreeManager::getCudaRenderer(void)
{
    if (m_renderMode != RenderMode_Cuda || !CudaModule::isAvailable())
        return NULL;
    if (!m_cudaRenderer)
        m_cudaRenderer = new CudaRenderer;
    return m_cudaRenderer;
}

//------------------------------------------------------------------------

BuilderBase* OctreeManager::getBuilder(BuilderType type)
{
    FW_ASSERT(type >= 0 && type < BuilderType_Max);
    if (!m_builders[type])
    {
        OctreeFile* f = getFile();
        BuilderBase* b;

        switch (type)
        {
        case BuilderType_Mesh:  b = new MeshBuilder(f); break;
        default:                FW_ASSERT(false); return NULL;
        }

        b->setMaxConcurrency(m_maxBuilderThreads);
        m_builders[type] = b;
    }
    return m_builders[type];
}

//------------------------------------------------------------------------

void OctreeManager::clearRuntime(void)
{
    if (m_cpuRuntime)
        m_cpuRuntime->clear();
    if (m_cudaRuntime)
        m_cudaRuntime->clear();
    m_loadSliceID = -1;
}

//------------------------------------------------------------------------

void OctreeManager::destroyRuntime(void)
{
    delete m_cpuRuntime;
    delete m_cudaRuntime;
    m_cpuRuntime = NULL;
    m_cudaRuntime = NULL;
    m_loadSliceID = -1;
}

//------------------------------------------------------------------------

void OctreeManager::newFile(void)
{
    unloadFile(true);
    clearRuntime();
}

//------------------------------------------------------------------------

void OctreeManager::loadFile(const String& fileName, bool edit)
{
    if (m_file && m_file->getName() == fileName)
        return;

    OctreeFile* old = m_file;
    m_file = NULL;
    unloadFile(true);
    clearRuntime();

    m_file = new OctreeFile(fileName, (edit) ? File::Modify : File::Read);
    if (!hasError())
        delete old;
    else
    {
        delete m_file;
        m_file = old;
    }
}

//------------------------------------------------------------------------

void OctreeManager::saveFile(const String& fileName, bool edit)
{
    if (!m_file)
        m_file = new OctreeFile(fileName, File::Create);

    if (m_file->getName() != fileName && m_tmpFileID == -1)
    {
        OctreeFile* oldFile = m_file;
        m_file = new OctreeFile(fileName, File::Create);
        m_file->set(*oldFile);
        delete oldFile;
    }

    if (m_file->getName() == fileName)
    {
        m_file->flush();
        return;
    }

    String oldName = m_file->getName();
    unloadFile(false);

    BOOL ok = MoveFileEx(oldName.getPtr(), fileName.getPtr(),
        MOVEFILE_COPY_ALLOWED | MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);

    if (ok)
    {
        unloadFile(true);
        m_file = new OctreeFile(fileName, (edit) ? File::Modify : File::Read);
    }
    else
    {
        setError("Cannot move '%s' to '%s'!", oldName.getPtr(), fileName.getPtr());
        m_file = new OctreeFile(oldName, File::Modify);
    }
}

//------------------------------------------------------------------------

void OctreeManager::editFile(bool inPlace)
{
    if (isEditable())
        return;

    if (inPlace)
    {
        String name = m_file->getName();
        unloadFile(false);
        m_file = new OctreeFile(name, File::Modify);
    }
    else
    {
        OctreeFile* old = m_file;
        m_file = NULL;
        unloadFile(false);
        getFile()->set(*old);
        delete old;
    }
}

//------------------------------------------------------------------------

void OctreeManager::rebuildFile(BuilderType builderType, const BuilderBase::Params& params, int numLevelsToBuild, const String& saveFileName)
{
    FW_ASSERT(numLevelsToBuild >= 0);
    clearRuntime();

    if (isEditable() && !saveFileName.getLength())
    {
        for (int i = 0; i < BuilderType_Max; i++)
            if (m_builders[i])
                m_builders[i]->asyncAbort();
        m_file->clearSlices();
    }
    else
    {
        Array<OctreeFile::Object> objects;
        Array<MeshBase*> meshes;
        for (int i = 0; i < getFile()->getNumObjects(); i++)
        {
            objects.add(getFile()->getObject(i));
            meshes.add(getFile()->getMeshCopy(i));
        }

        unloadFile(true);
        if (saveFileName.getLength())
            m_file = new OctreeFile(saveFileName, File::Create);

        for (int i = 0; i < objects.getSize(); i++)
        {
            getFile()->addObject();
            OctreeFile::Object obj = getFile()->getObject(i);
            obj.objectToWorld = objects[i].objectToWorld;
            getFile()->setObject(i, obj);
            getFile()->setMesh(i, meshes[i]);
        }
    }

    for (int i = 0; i < getFile()->getNumObjects(); i++)
        getBuilder(builderType)->buildObject(i, numLevelsToBuild, params, (numLevelsToBuild != 0));
}

//------------------------------------------------------------------------

int OctreeManager::addMesh(MeshBase* mesh, BuilderType builderType, const BuilderBase::Params& params, int numLevelsToBuild)
{
    FW_ASSERT(mesh);
    FW_ASSERT(numLevelsToBuild >= 0);

    editFile();
    int objectID = getFile()->addObject();
    getFile()->setMesh(objectID, mesh);

    getBuilder(builderType)->buildObject(objectID, numLevelsToBuild, params, (numLevelsToBuild != 0));
    return objectID;
}

//------------------------------------------------------------------------

void OctreeManager::renderObject(GLContext* gl, int objectID, const Mat4f& worldToCamera, const Mat4f& projection)
{
    FW_ASSERT(gl);

    F32 frameDelta = m_frameDeltaTimer.end();
    m_frameTimer.clearTotal();
    m_updateTimer.clearTotal();
    m_renderTimer.clearTotal();
    m_loadBytesTotal = 0.0f;

    m_frameTimer.start();
    renderInternal(gl, objectID, worldToCamera, projection, frameDelta);
    m_frameTimer.end();

    F32 t = exp2(-frameDelta / 0.3f);
    m_frameDeltaAvg = lerp(frameDelta, m_frameDeltaAvg, t);
    m_frameTimeAvg  = lerp(m_frameTimer.getTotal(), m_frameTimeAvg, t);
    m_updateTimeAvg = lerp(m_updateTimer.getTotal(), m_updateTimeAvg, t);
    m_renderTimeAvg = lerp(m_renderTimer.getTotal(), m_renderTimeAvg, t);
    m_loadBytesAvg  = lerp(m_loadBytesTotal, m_loadBytesAvg, t);
}

//------------------------------------------------------------------------

int OctreeManager::allocateTmpFileID(void)
{
    s_numTmpFileIDs++;
    if (s_freeTmpFileIDs.getSize())
        return s_freeTmpFileIDs.removeLast();
    return s_maxTmpFileIDs++;
}

//------------------------------------------------------------------------

void OctreeManager::freeTmpFileID(int id)
{
    if (id == -1)
        return;

    s_freeTmpFileIDs.add(id);
    s_numTmpFileIDs--;

    if (!s_numTmpFileIDs)
    {
        s_maxTmpFileIDs = 0;
        s_freeTmpFileIDs.reset();
    }
}

//------------------------------------------------------------------------

void OctreeManager::unloadFile(bool freeID)
{
    for (int i = 0; i < BuilderType_Max; i++)
    {
        delete m_builders[i];
        m_builders[i] = NULL;
    }

    delete m_file;
    m_file = NULL;

    if (freeID)
    {
        freeTmpFileID(m_tmpFileID);
        m_tmpFileID = -1;
    }
}

//------------------------------------------------------------------------

void OctreeManager::renderInternal(GLContext* gl, int objectID, const Mat4f& worldToCamera, const Mat4f& projection, F32 frameDelta)
{
    FW_ASSERT(gl);

    // Check CUDA availability.

    if (m_renderMode == RenderMode_Cuda && !CudaModule::isAvailable())
    {
        gl->drawModalMessage("CUDA not available!");
        return;
    }

    // Check that the object is valid.

    if (objectID < 0 || objectID >= getFile()->getNumObjects())
    {
        gl->drawModalMessage("No object loaded!");
        return;
    }

    // Update runtime.

    const OctreeFile::Object& obj = getFile()->getObject(objectID);
    OctreeRuntime* runtime = getRuntime();
    Mat4f octreeToCamera;

    if (runtime && obj.rootSlice != -1)
    {
        profilePush("Update runtime");
        octreeToCamera = worldToCamera * obj.objectToWorld * obj.octreeToObject;
        Vec3f cameraInOctree = Vec4f(octreeToCamera.inverted().col(3)).getXYZ();
        F32 timeLimit = min(frameDelta * (F32)UpdateTimePct / 100.0f, (F32)UpdateTimeMaxMillis / 1000.0f);
        m_updateTimer.start();
        updateRuntime(obj.rootSlice, cameraInOctree, timeLimit);
        m_updateTimer.end();
        profilePop();
    }

    // Render.

    profilePush("Render");
    glClearColor(0.2f, 0.4f, 0.8f, 1.0f);

    switch (m_renderMode)
    {
    case RenderMode_Mesh:
        {
            MeshBase* mesh = getFile()->getMesh(objectID);
            if (!mesh)
                gl->drawModalMessage("Object does not have a mesh!");
            else
            {
                m_renderTimer.start();
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                glEnable(GL_DEPTH_TEST);
                glDisable(GL_CULL_FACE);
                mesh->draw(gl, worldToCamera * obj.objectToWorld, projection);
                m_renderTimer.end();
            }
        }
        break;

    case RenderMode_Cuda:
        if (!getRuntime()->getRootNodeCuda(obj.rootSlice))
        {
            if (m_dynamicLoad && m_dynamicBuild && isEditable())
                gl->drawModalMessage("Building root slice...");
            else
            gl->drawModalMessage("No slices loaded!");
        }
        else
        {
            m_renderTimer.start();
            glDisable(GL_DEPTH_TEST);
            String error = getCudaRenderer()->renderObject(gl, runtime, objectID, obj.objectToWorld * obj.octreeToObject, worldToCamera, projection);
            if (error.getLength())
                gl->drawModalMessage(error);
            m_renderTimer.end();
        }
        break;

    default:
        FW_ASSERT(false);
        break;
    }

    profilePop();
}

//------------------------------------------------------------------------

void OctreeManager::updateRuntime(int objectID, const Vec3f& cameraInOctree, F32 timeLimit)
{
    const Vec3f&    cam     = cameraInOctree;
    OctreeFile*     file    = getFile();
    OctreeRuntime*  runtime = getRuntime();
    Timer           timer   (true);
    FW_ASSERT(file && runtime);

    // Introduce object to runtime.

    const OctreeFile::Object& obj = file->getObject(objectID);
    Array<AttachIO::AttachType> attach;
    if (m_renderMode == RenderMode_Cuda)
        m_cudaRenderer->selectAttachments(attach, obj.runtimeAttachTypes);
    else
        attach = obj.runtimeAttachTypes;

    if (runtime->hasObject(objectID) && attach != runtime->getAttachTypes(objectID))
    {
        runtime->removeObject(objectID);
        m_loadSliceID = -1;
    }

    if (!runtime->hasObject(objectID))
        runtime->addObject(objectID, obj.rootSlice, attach);

    // Unload slices exceeding the level limit.

    while (timer.getElapsed() < timeLimit)
    {
        OctreeRuntime::FindResult deep = runtime->findSlice(OctreeRuntime::FindMode_UnloadDeepest,
            objectID, cam, cam, OctreeFile::UnitScale);

        if (deep.sliceID == -1 || deep.score > (F32)(OctreeFile::UnitScale - m_maxLevels))
            break;

        runtime->unloadSlice(deep.sliceID);
    }

    // Dynamic loading is disabled => done.

    if (!m_dynamicLoad)
        return;

    // Load and build slices while we still have time.

    bool build = (m_dynamicBuild && isEditable());
    while (timer.getElapsed() < timeLimit)
    {
        // Find slices.

        Array<OctreeRuntime::FindResult> slices;
        runtime->findSlices(slices,
            (build) ? OctreeRuntime::FindMode_LoadOrBuild : OctreeRuntime::FindMode_Load,
            objectID, cam, cam, m_maxLevels, MaxPrefetchSlices);

        // Refresh state and prefetch.

        for (int i = 0; i < slices.getSize(); i++)
            if (!runtime->setSliceState(slices[i].sliceID, file->getSliceState(slices[i].sliceID)) && !build)
                slices[i].sliceID = -1;
        prefetchSlices(slices, timer, timeLimit);

        // Load and build.

        if (!loadSlices(slices, timer, timeLimit, objectID, cam) &&
            (!build || !buildSlices(slices, timer, timeLimit)))
        {
            break;
        }
    }
}

//------------------------------------------------------------------------

void OctreeManager::prefetchSlices(const Array<OctreeRuntime::FindResult>& slices, Timer& timer, F32 timeLimit)
{
    OctreeFile* file = getFile();
    FW_ASSERT(file);
    profilePush("Prefetch slices");

    S32 bytesTotal = 0;
    S32 bytesPending = 0;

    for (int i = 0; i < slices.getSize() && timer.getElapsed() < timeLimit; i++)
    {
        int sliceID = slices[i].sliceID;
        if (sliceID == -1)
            continue;

        int size = file->getSliceSize(sliceID);
        bytesTotal += size;
        if (i && bytesTotal > MaxPrefetchBytesTotal)
            break;

        if (!file->readSliceIsReady(sliceID))
        {
            bytesPending += size;
            if (i && bytesPending > MaxPrefetchBytesPending)
                break;
            file->readSlicePrefetch(sliceID);
        }
    }

    profilePop();
}

//------------------------------------------------------------------------

bool OctreeManager::loadSlices(const Array<OctreeRuntime::FindResult>& slices, Timer& timer, F32 timeLimit, int objectID, const Vec3f& cameraInOctree)
{
    const Vec3f&    cam             = cameraInOctree;
    OctreeFile*     file            = getFile();
    OctreeRuntime*  runtime         = getRuntime();
    bool            retry           = false;
    F32             scoreRequired   = -FW_F32_MAX;

    FW_ASSERT(file && runtime);
    profilePush("Load slices");

    for (int i = 0; i < slices.getSize() && timer.getElapsed() < timeLimit; i++)
    {
        int sliceID = slices[i].sliceID;
        if (sliceID == -1)
            continue;

        if (!file->readSliceIsReady(sliceID) ||
            slices[i].score <= scoreRequired ||
            file->getSliceState(sliceID) != OctreeFile::SliceState_Complete)
        {
            break;
        }

        // Prepare to load slice.

        if (m_loadSliceID != sliceID)
        {
            profilePush("Decode");
            OctreeSlice slice;
            file->readSlice(sliceID, slice);
            m_loadSliceID = sliceID;
            m_loadSliceBytesDisk = slice.getSize() * sizeof(S32);
            m_loadSliceBytesMemory = runtime->setSliceToLoad(slice);
            profilePop();
        }

        // Try to load, unloading slices as necessary.

        while (timer.getElapsed() < timeLimit)
        {
            profilePush("Upload");
            bool loaded = false;
            if (runtime->getFreeBytes() - m_loadSliceBytesMemory >= RuntimeSlackBytes)
                loaded = runtime->loadSlice();
            profilePop();

            if (loaded)
            {
                m_loadSliceID = -1;
                m_loadBytesTotal += (F32)m_loadSliceBytesDisk;
                retry = true;
                break;
            }

            OctreeRuntime::FindResult unload = runtime->findSlice(OctreeRuntime::FindMode_Unload,
                objectID, cam, cam, OctreeFile::UnitScale);

            if (unload.sliceID == -1 || unload.score >= slices[i].score)
            {
                profilePop();
                return retry;
            }

            profilePush("Free up memory");
            runtime->unloadSlice(unload.sliceID);
            scoreRequired = max(scoreRequired, unload.score);
            profilePop();
        }
    }

    profilePop();
    return retry;
}

//------------------------------------------------------------------------

bool OctreeManager::buildSlices(const Array<OctreeRuntime::FindResult>& slices, Timer& timer, F32 timeLimit)
{
    OctreeFile*     file    = getFile();
    OctreeRuntime*  runtime = getRuntime();
    bool            retry   = false;
    FW_ASSERT(file && runtime);
    profilePush("Build slices");

    // Finish async builds.

    for (int i = 0; i < (int)BuilderType_Max; i++)
    {
        BuilderBase* b = getBuilder((BuilderType)i);
        while (b && timer.getElapsed() < timeLimit)
        {
            OctreeSlice* slice = b->asyncFinishSlice(false);
            if (!slice)
                break;

            runtime->setSliceState(slice->getID(), slice->getState());
            delete slice;
            retry = true;
        }
    }

    // Start async builds.

    for (int i = 0; i < slices.getSize() && timer.getElapsed() < timeLimit; i++)
    {
        int sliceID = slices[i].sliceID;
        if (sliceID == -1 || file->getSliceState(sliceID) != OctreeFile::SliceState_Unbuilt)
            continue;

        if (!file->readSliceIsReady(sliceID))
            break;

        // Do not exceed MaxAsyncBuildSlices.

        int numAsync = 0;
        bool alreadyBuilding = false;
        for (int i = 0; i < (int)BuilderType_Max; i++)
        {
            BuilderBase* b = getBuilder((BuilderType)i);
            if (b)
            {
                numAsync += b->asyncGetNumPending();
                if (b->asyncIsPending(sliceID))
                    alreadyBuilding = true;
            }
        }
        if (numAsync >= MaxAsyncBuildSlices)
            break;
        if (alreadyBuilding)
            continue;

        // Read from the file.

        OctreeSlice* slice = new OctreeSlice;
        file->readSlice(sliceID, *slice);

        // Try each builder.

        for (int i = 0; i < (int)BuilderType_Max; i++)
        {
            BuilderBase* b = getBuilder((BuilderType)i);
            if (b && b->asyncBuildSlice(slice))
            {
                slice = NULL;
                break;
            }
        }
        if (slice)
            delete slice;
    }

    profilePop();
    return retry;
}

//------------------------------------------------------------------------

String OctreeManager::getStats(void) const
{
    return sprintf("OctreeManager: render %.2f ms (%.0f%%), update %.2f ms (%.0f%%), load %.2f MB/s",
        1000.0f * m_renderTimeAvg, 100.0f * m_renderTimeAvg / m_frameTimeAvg,
        1000.0f * m_updateTimeAvg, 100.0f * m_updateTimeAvg / m_frameTimeAvg,
        m_loadBytesAvg / m_frameDeltaAvg / 1024.0f / 1024.0f);
}

//------------------------------------------------------------------------
