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
#include "io/OctreeFile.hpp"
#include "io/OctreeRuntime.hpp"
#include "build/BuilderBase.hpp"
#include "render/CudaRenderer.hpp"
#include "base/Timer.hpp"

namespace FW
{
//------------------------------------------------------------------------

class OctreeManager
{
public:
    enum
    {
        RuntimeSlackBytes       = 10 << 20,   // Minimum amount of free runtime memory.
        UpdateTimePct           = 50,
        UpdateTimeMaxMillis     = 200,
        MaxPrefetchSlices       = OctreeFile::MaxPrefetchSlices,
        MaxPrefetchBytesTotal   = OctreeFile::MaxPrefetchBytesTotal,
        MaxPrefetchBytesPending = 8 << 20,
        MaxAsyncBuildSlices     = BuilderBase::MaxAsyncBuildSlices
    };

    enum RenderMode
    {
        RenderMode_Mesh = 0,
        RenderMode_Cuda = 2,
    };

    enum BuilderType
    {
        BuilderType_Mesh = 0,

        BuilderType_Max
    };

public:
                        OctreeManager       (RenderMode renderMode = RenderMode_Mesh);
                        ~OctreeManager      (void);

    void                setMaxConcurrency   (int maxBuilderThreads) { FW_ASSERT(maxBuilderThreads > 0); m_maxBuilderThreads = maxBuilderThreads; }

    void                setRenderMode       (RenderMode renderMode);
    void                setDynamicLoad      (bool dynamicLoad)  { m_dynamicLoad = dynamicLoad; }
    void                setDynamicBuild     (bool dynamicBuild) { m_dynamicBuild = dynamicBuild; }
    void                setMaxLevels        (int maxLevels)     { m_maxLevels = maxLevels; }

    OctreeFile*         getFile             (void);
    OctreeRuntime*      getRuntime          (void);
    CudaRenderer*       getCudaRenderer     (void);
    BuilderBase*        getBuilder          (BuilderType type);

    void                clearRuntime        (void);
    void                destroyRuntime      (void);

    void                newFile             (void);
    void                loadFile            (const String& fileName, bool edit = false);
    void                saveFile            (const String& fileName, bool edit = false);
    void                editFile            (bool inPlace = false);
    bool                isEditable          (void) const        { return (!m_file || m_file->getMode() != File::Read); }
    void                rebuildFile         (BuilderType builderType, const BuilderBase::Params& params, int numLevelsToBuild = 0, const String& saveFileName = "");

    int                 addMesh             (MeshBase* mesh, BuilderType builderType, const BuilderBase::Params& params, int numLevelsToBuild = 0);

    void                renderObject        (GLContext* gl, int objectID, const Mat4f& worldToCamera, const Mat4f& projection);

    String              getStats            (void) const;

private:
    static int          allocateTmpFileID   (void);
    static void         freeTmpFileID       (int id);

    void                unloadFile          (bool freeID);

    void                renderInternal      (GLContext* gl, int objectID, const Mat4f& worldToCamera, const Mat4f& projection, F32 frameDelta);

    void                updateRuntime       (int objectID, const Vec3f& cameraInOctree, F32 timeLimit);
    void                prefetchSlices      (const Array<OctreeRuntime::FindResult>& slices, Timer& timer, F32 timeLimit);
    bool                loadSlices          (const Array<OctreeRuntime::FindResult>& slices, Timer& timer, F32 timeLimit, int objectID, const Vec3f& cameraInOctree);
    bool                buildSlices         (const Array<OctreeRuntime::FindResult>& slices, Timer& timer, F32 timeLimit);

private:
                        OctreeManager       (OctreeManager&); // forbidden
    OctreeManager&      operator=           (OctreeManager&); // forbidden

private:
    static S32          s_numTmpFileIDs;
    static S32          s_maxTmpFileIDs;
    static Array<S32>   s_freeTmpFileIDs;

    S32                 m_maxBuilderThreads;

    RenderMode          m_renderMode;
    bool                m_dynamicLoad;
    bool                m_dynamicBuild;
    S32                 m_maxLevels;

    S32                 m_tmpFileID;        // -1 if none
    OctreeFile*         m_file;
    BuilderBase*        m_builders[BuilderType_Max];

    OctreeRuntime*      m_cpuRuntime;
    OctreeRuntime*      m_cudaRuntime;
    CudaRenderer*       m_cudaRenderer;

    S32                 m_loadSliceID;
    S32                 m_loadSliceBytesDisk;
    S32                 m_loadSliceBytesMemory;

    Timer               m_frameDeltaTimer;
    Timer               m_frameTimer;
    Timer               m_updateTimer;
    Timer               m_renderTimer;
    F32                 m_loadBytesTotal;

    F32                 m_frameDeltaAvg;
    F32                 m_frameTimeAvg;
    F32                 m_updateTimeAvg;
    F32                 m_renderTimeAvg;
    F32                 m_loadBytesAvg;
};

//------------------------------------------------------------------------
}
