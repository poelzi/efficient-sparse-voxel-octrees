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
#include "BenchmarkContext.hpp"
#include "cuda/Render.hpp"

namespace FW
{

class Benchmark
{
public:
    struct Result
    {
        String          title;
        F32             mraysPerSecActual;
        F32             mraysPerSecWallclock;
        F32             mraysPerSecSimulated;
        F32             gigsPerSec;
        F32             renderWarps;
        F32             coarseWarps;
        F32             coarsePassPct;
        F32             threadCountersPerRay[PerfCounter_Max];
        F32             warpCountersPerRay[PerfCounter_Max];
    };

public:
                        Benchmark               (void);
                        ~Benchmark              (void);

    void                setFrameSize            (const Vec2i& value)            { m_frameSize = value; }
    void                setFramesPerLaunch      (S32 value)                     { m_framesPerLaunch = value; }
    void                setWarmupLaunches       (S32 value)                     { m_warmupLaunches = value; }
    void                setMeasureFrames        (S32 value)                     { m_measureFrames = value; }

    void                loadOctree              (const String& fileName, int numLevels = OctreeFile::UnitScale) { m_ctx.setFile(fileName); m_ctx.load(numLevels); }
    void                setCameras              (const Array<String>& value)    { m_cameras = value; }

    void                clearResults            (void)                          { m_results.clear(); }
    void                measure                 (const String& columnTitle, const CudaRenderer::Params& renderParams);
    void                printResults            (const String& majorTitle, const String& minorTitle);

private:
                        Benchmark               (const Benchmark&); // forbidden
    Benchmark&          operator=               (const Benchmark&); // forbidden

private:
    BenchmarkContext    m_ctx;

    Vec2i               m_frameSize;
    S32                 m_framesPerLaunch;
    S32                 m_warmupLaunches;
    S32                 m_measureFrames;
    Array<String>       m_cameras;

    Array<Result>       m_results;
};

//------------------------------------------------------------------------
}
