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
#include "../io/OctreeRuntime.hpp"
#include "../cuda/Render.hpp"
#include "PixelTable.hpp"
#include "gui/Image.hpp"
#include "gpu/CudaCompiler.hpp"

namespace FW
{
//------------------------------------------------------------------------

class CudaRenderer
{
public:
    enum Visualization
    {
        Visualization_Primary,
        Visualization_PrimaryAndShadow,
        Visualization_RaycastLevel,
        Visualization_IterationCount,
    };

    struct Params
    {
        Visualization   visualization;
        bool            enableContours;
        bool            enableAntialias;
        bool            enableLargeReconstruction;
        bool            enableJitterLOD;
        bool            enableBlur;
        bool            enableBeamOptimization;
        bool            measureRaycastPerf;
        bool            enablePerfCounters;
        S32             batchSize;
        S32             coarseSize;
        F32             maxVoxelSize;
        F32             brightness;
        S32             numFrameRepeats;

        Params(void)
        {
            visualization             = Visualization_Primary;
            enableContours            = true;
            enableAntialias           = false;
            enableLargeReconstruction = false;
            enableJitterLOD           = false;
            enableBlur                = true;
            enableBeamOptimization    = false;
            measureRaycastPerf        = false;
            enablePerfCounters        = false;
            batchSize                 = 3;
            coarseSize                = 4;
            maxVoxelSize              = 1.0f;
            brightness                = 1.7f;
            numFrameRepeats           = 1;
        }
    };

    struct Results
    {
        F32             launchTime; // total time spent in CUDA
        F32             coarseTime; // time spent on the coarse pass (beam optimization)
        S32             renderWarps;
        S32             coarseWarps;
        S64             threadCounters[PerfCounter_Max];
        S64             warpCounters[PerfCounter_Max];
    };

private:
    struct LaunchResult
    {
        F32             time;
        S32             numWarps;

        LaunchResult(void)
        {
            time        = 0.0f;
            numWarps    = 0;
        }
    };

public:
                        CudaRenderer        (void);
                        ~CudaRenderer       (void);

    void                selectAttachments   (Array<AttachIO::AttachType>& out, const Array<AttachIO::AttachType>& in) const;

    String              renderObject        (Image&         frame,
                                             OctreeRuntime* runtime,
                                             int            objectID,
                                             const Mat4f&   octreeToWorld,
                                             const Mat4f&   worldToCamera,
                                             const Mat4f&   projection);

    String              renderObject        (GLContext*     gl,
                                             OctreeRuntime* runtime,
                                             int            objectID,
                                             const Mat4f&   octreeToWorld,
                                             const Mat4f&   worldToCamera,
                                             const Mat4f&   projection);

    String              getStats            (void) const        { return m_stats; }
    void                setParams           (const Params& p)   { m_params = p; }
    const Results&      getResults          (void) const        { return m_results; }
    void                clearResults        (void);
    void                setWindow           (Window* window)    { m_compiler.setMessageWindow(window); }

    void                populateCompilerCache(void);

private:
    LaunchResult        launch              (int totalWork, bool persistentThreads);
    void                constructBlurLUT    (void);

private:
                        CudaRenderer        (CudaRenderer&); // forbidden
    CudaRenderer&       operator=           (CudaRenderer&); // forbidden

private:
    CudaCompiler        m_compiler;
    PixelTable          m_pixelTable;
    Buffer              m_frameBuffer;
    Buffer              m_indexToPixel;
    PixelTable          m_coarsePixelTable;
    Buffer              m_coarseFrameBuffer;
    Buffer              m_coarseIndexToPixel;
    Buffer              m_tempFrameBuffer;
    Buffer              m_aaSampleBuffer;
    Array<Vec3i>        m_blurLUT; // (x,y,area)

    RenderInput         m_input;
    S32                 m_numWarps;
    Buffer              m_activeWarps;
    Buffer              m_perfCounters;

    String              m_stats;
    Params              m_params;

    Results             m_results;
};

//------------------------------------------------------------------------
}
