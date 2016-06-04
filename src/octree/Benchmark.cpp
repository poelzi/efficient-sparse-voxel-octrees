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

#include "Benchmark.hpp"
#include "Util.hpp"
#include "build/MeshBuilder.hpp"

using namespace FW;

//------------------------------------------------------------------------

Benchmark::Benchmark(void)
:   m_frameSize         (1024, 768),
    m_framesPerLaunch   (10),
    m_warmupLaunches    (4),
    m_measureFrames     (2000)
{
}

//------------------------------------------------------------------------

Benchmark::~Benchmark(void)
{
}

//------------------------------------------------------------------------

void Benchmark::measure(const String& columnTitle, const CudaRenderer::Params& renderParams)
{
    int frameDenom = m_cameras.getSize() * m_framesPerLaunch;
    int framesPerCamera = (m_measureFrames + frameDenom - 1) / frameDenom;
    int repeatsPerCamera = framesPerCamera * m_framesPerLaunch;

    // Initialize system.

    String fullTitle = String("Measuring raycast perf: ") + columnTitle;
    printf("%s\n", fullTitle.getPtr());
    m_ctx.setWindowTitle(fullTitle);
    Image frame(m_frameSize);
    CudaRenderer* renderer = m_ctx.getRenderer();

    // Initialize measurements.

    F32 launchTime  = 0.0f;
    F32 coarseTime  = 0.0f;
    F32 totalTime   = 0.0f;
    F32 renderWarps = 0.0f;
    F32 coarseWarps = 0.0f;

    Array<Vec2f> counters;
    for (int i = 0; i < PerfCounter_Max; i++)
        counters.add(Vec2f(0.0f));

    // Measure at each camera position.

    for (int cameraIdx = 0; cameraIdx < m_cameras.getSize(); cameraIdx++)
    {
        // Display rendered image.

        CudaRenderer::Params params = renderParams;
        renderer->setParams(params);
        m_ctx.setCamera(m_cameras[cameraIdx]);
        m_ctx.showOctree(frame.getSize());

        // Warm up performance.

        params.enableBlur = false;
        params.measureRaycastPerf = true;
        params.numFrameRepeats = m_framesPerLaunch;
        renderer->setParams(params);
        for (int i = 0; i < m_warmupLaunches; i++)
            m_ctx.renderOctree(frame);

        // Measure performance.

        Timer timer;
        timer.start();
        renderer->clearResults();
        for (int i = 0; i < framesPerCamera; i++)
            m_ctx.renderOctree(frame);

        launchTime  += renderer->getResults().launchTime;
        coarseTime  += renderer->getResults().coarseTime;
        totalTime   += timer.getElapsed();
        renderWarps += (F32)renderer->getResults().renderWarps;
        coarseWarps += (F32)renderer->getResults().coarseWarps;

        // Measure counters.

        params.enablePerfCounters = true;
        params.numFrameRepeats = 1;
        renderer->setParams(params);
        renderer->clearResults();
        m_ctx.renderOctree(frame);

        for (int i = 0; i < PerfCounter_Max; i++)
        {
            counters[i].x += (F32)renderer->getResults().threadCounters[i] * (F32)repeatsPerCamera;
            counters[i].y += (F32)renderer->getResults().warpCounters[i] * (F32)repeatsPerCamera;
        }
    }

    m_ctx.hideWindow();

    // Query/compute useful numbers.

    F32 raysPerFrame = (F32)m_frameSize.x * (F32)m_frameSize.y * (F32)m_framesPerLaunch;
    F32 framesTotal  = (F32)m_cameras.getSize() * (F32)framesPerCamera;
    F32 raysTotal    = framesTotal * raysPerFrame;
    F32 mraysTotal   = raysTotal * 1.0e-6f;
    F32 numSMs       = (F32)CudaModule::getDeviceAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    F32 clockRate    = (F32)CudaModule::getDeviceAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE) * 1000.0f;
    F32 flops        = numSMs * clockRate * 32.0f / 4.0f;

    // Kernel overhead, excluding the raycast loop.
    // Calibrated to match measured performance on an empty octree.

    F32 instrOverhead = 270.5f; // Without persistent threads.
//    F32 instrOverhead = 971.2f; // With persistent threads.

    // Output results.

    Result& res = m_results.add();
    res.title = columnTitle;
    res.mraysPerSecActual = mraysTotal / launchTime;
    res.mraysPerSecWallclock = mraysTotal / totalTime;

    F32 warpOps = counters[PerfCounter_Instructions].y;
    warpOps += instrOverhead * (F32)raysTotal / 32.0f;
    res.mraysPerSecSimulated = mraysTotal * flops / (warpOps * 32.0f);

    F32 bytes = 0.0f;
    bytes += counters[PerfCounter_GlobalBytes].x;
    bytes += counters[PerfCounter_LocalBytes].x;
    res.gigsPerSec = bytes / launchTime * exp2(-30);

    res.renderWarps = renderWarps / framesTotal;
    res.coarseWarps = coarseWarps / framesTotal;
    res.coarsePassPct = coarseTime / launchTime * 100.0f;

    for (int i = 0; i < PerfCounter_Max; i++)
    {
        res.threadCountersPerRay[i] = counters[i].x / raysTotal;
        res.warpCountersPerRay[i] = counters[i].y / raysTotal;
    }
}

//------------------------------------------------------------------------

void Benchmark::printResults(const String& majorTitle, const String& minorTitle)
{
    static const char* const s_perfCounterNames[] =
    {
    #define X(NAME) #NAME,
        PERF_COUNTER_LIST(X)
    #undef X
    };

    printf("\n");
    printf("%-24s", majorTitle.getPtr());
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-14s", m_results[i].title.getPtr());
    printf("\n");
        printf("%-24s", minorTitle.getPtr());
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8s%-6s", "Value", "SIMD%");
    printf("\n");
    printf("%-24s", "---");
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8s%-6s", "---", "---");
    printf("\n");

    printf("%-24s", "Measured MRays/sec");
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8.2f%-6s", m_results[i].mraysPerSecActual, "-");
    printf("\n");

    printf("%-24s", "Simulated MRays/sec");
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8.2f%-6s", m_results[i].mraysPerSecSimulated, "-");
    printf("\n");

    printf("%-24s", "Measured % of simulated");
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8.1f%-6s", m_results[i].mraysPerSecActual / m_results[i].mraysPerSecSimulated * 100.0f, "-");
    printf("\n");

    printf("%-24s", "Bandwidth GB/sec");
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8.2f%-6s", m_results[i].gigsPerSec, "-");
    printf("\n");
/*
    printf("%-24s", "Simultaneous warps");
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8.1f%-6s", m_results[i].renderWarps, "-");
    printf("\n");

    printf("%-24s", "Coarse warps");
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8.1f%-6s", m_results[i].coarseWarps, "-");
    printf("\n");
*/
    printf("%-24s", "Coarse pass cost %");
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8.1f%-6s", m_results[i].coarsePassPct, "-");
    printf("\n");

    for (int i = 0; i < PerfCounter_Max; i++)
    {
        if (i == 0 || i == PerfCounter_GlobalAccesses)
        {
            printf("%-24s", "---");
            for (int i = 0; i < m_results.getSize(); i++)
                printf("| %-8s%-6s", "---", "---");
            printf("\n");
        }

        printf("%-24s", s_perfCounterNames[i]);
        for (int j = 0; j < m_results.getSize(); j++)
        {
            printf("| %-8.1f", m_results[j].threadCountersPerRay[i]);
            if (i == PerfCounter_GlobalTransactions || i == PerfCounter_LocalTransactions || m_results[j].warpCountersPerRay[i] <= 0.0f)
                printf("%-6s", "-");
            else
                printf("%-6.1f", m_results[j].threadCountersPerRay[i] / (m_results[j].warpCountersPerRay[i] * 32.0f) * 100.0f);
        }
        printf("\n");
    }

    printf("%-24s", "---");
    for (int i = 0; i < m_results.getSize(); i++)
        printf("| %-8s%-6s", "---", "---");
    printf("\n");
    printf("\n");
}

//------------------------------------------------------------------------
