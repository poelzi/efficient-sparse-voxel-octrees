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

#include "CudaRenderer.hpp"

using namespace FW;

//------------------------------------------------------------------------

CudaRenderer::CudaRenderer(void)
:   m_frameBuffer       (NULL, 0, Buffer::Hint_CudaGL),
    m_numWarps          (256) // initial guess
{
    m_compiler.setSourceFile("src/octree/cuda/Render.cu");
    m_compiler.addOptions("-use_fast_math");
    m_compiler.include("src/framework");
    clearResults();
}

//------------------------------------------------------------------------

CudaRenderer::~CudaRenderer(void)
{
}

//------------------------------------------------------------------------

void CudaRenderer::selectAttachments(Array<AttachIO::AttachType>& out, const Array<AttachIO::AttachType>& in) const
{
    // Clear slots.

    out.clear();
    for (int i = 0; i < AttachSlot_Max; i++)
        out.add(AttachIO::VoidAttach);

    // Assign imported attachments.

    for (int i = 0; i < in.getSize(); i++)
    {
        AttachIO::AttachType t = in[i];
        switch (t)
        {
        case AttachIO::ColorNormalPaletteAttach:    out[AttachSlot_Attribute] = t; break;
        case AttachIO::ColorNormalCornerAttach:     out[AttachSlot_Attribute] = t; break;
        case AttachIO::ColorNormalDXTAttach:        out[AttachSlot_Attribute] = t; break;
        case AttachIO::ContourAttach:               out[AttachSlot_Contour] = t; break;
        case AttachIO::AOAttach:                    out[AttachSlot_AO] = t; break;
        default:                                    break;
        }
    }
}

//------------------------------------------------------------------------

String CudaRenderer::renderObject(
    Image&          frame,
    OctreeRuntime*  runtime,
    int             objectID,
    const Mat4f&    octreeToWorld,
    const Mat4f&    worldToCamera,
    const Mat4f&    projection)
{
    FW_ASSERT(runtime);

    // Check frame buffer validity.

    if (frame.getSize().min() <= 0)
        return "";

    if (frame.getFormat() != ImageFormat::ABGR_8888 ||
        frame.getStride() != frame.getSize().x * frame.getBPP())
    {
        return "CudaRenderer: Incompatible framebuffer!";
    }

    // Determine preprocessor defines.

    const Array<AttachIO::AttachType>& attach = runtime->getAttachTypes(objectID);
    FW_ASSERT(attach.getSize() == AttachSlot_Max);

    m_compiler.clearDefines();

    bool enableContours = (attach[AttachSlot_Contour] == AttachIO::ContourAttach && m_params.enableContours);
    if (enableContours)
        m_compiler.define("ENABLE_CONTOURS");

    switch (attach[AttachSlot_Attribute])
    {
    case AttachIO::ColorNormalPaletteAttach:    m_compiler.define("VOXELATTRIB_PALETTE"); m_compiler.define("DISABLE_PUSH_OPTIMIZATION"); break;
    case AttachIO::ColorNormalCornerAttach:     m_compiler.define("VOXELATTRIB_CORNER"); m_compiler.define("DISABLE_PUSH_OPTIMIZATION"); break;
    case AttachIO::ColorNormalDXTAttach:        m_compiler.define("VOXELATTRIB_DXT"); break;
    default:                                    return "Unsupported attribute attachment!";
    }

    if (attach[AttachSlot_AO] == AttachIO::AOAttach)
        m_compiler.define("VOXELATTRIB_AO");

    if (m_params.measureRaycastPerf)
        m_compiler.define("KERNEL_RAYCAST_PERF");
    else
        m_compiler.define("KERNEL_RENDER");

    if (m_params.enablePerfCounters)
        m_compiler.define("ENABLE_PERF_COUNTERS");

    if (m_params.enableLargeReconstruction)
        m_compiler.define("LARGE_RECONSTRUCTION_KERNEL");

    if (m_params.enableJitterLOD)
        m_compiler.define("JITTER_LOD");

    if (m_params.visualization == Visualization_PrimaryAndShadow)
        m_compiler.define("ENABLE_SHADOWS");

    if (!m_blurLUT.getSize())
        constructBlurLUT();
    m_compiler.define("BLUR_LUT_SIZE", String(m_blurLUT.getSize()));

    // Determine flags.

    U32 flags = 0;
    if (m_params.visualization == Visualization_IterationCount)
        flags |= RenderFlags_VisualizeIterations;
    else if (m_params.visualization == Visualization_RaycastLevel)
        flags |= RenderFlags_VisualizeRaycastLevel;

    // Set input.

    m_input.frameSize       = frame.getSize();
    m_input.flags           = flags;
    m_input.batchSize       = m_params.batchSize;
    m_input.aaRays          = (m_params.enableAntialias) ? 4 : 1;
    m_input.maxVoxelSize    = m_params.maxVoxelSize;
    m_input.brightness      = m_params.brightness;
    m_input.coarseSize      = m_params.coarseSize;
    m_input.coarseFrameSize = (m_input.frameSize + (m_params.coarseSize - 1)) / m_params.coarseSize + 1;
    m_input.frame           = frame.getBuffer().getMutableCudaPtr();
    m_input.rootNode        = runtime->getRootNodeCuda(objectID);

    OctreeMatrices& om      = m_input.octreeMatrices;
    Vec3f scale             = Vec3f(Vec2f(2.0f) / Vec2f(m_input.frameSize), 1.0f);
    om.viewportToCamera     = projection.inverted() * Mat4f::translate(Vec3f(-1.0f, -1.0f, 0.0f)) * Mat4f::scale(scale);
    om.cameraToOctree       = Mat4f::translate(Vec3f(1.0f)) * (worldToCamera * octreeToWorld).inverted();
    Mat4f vto               = om.cameraToOctree * om.viewportToCamera;
    om.pixelInOctree        = sqrt(Vec4f(vto.col(0)).getXYZ().cross(Vec4f(vto.col(1)).getXYZ()).length());

    om.octreeToWorld        = octreeToWorld * Mat4f::translate(Vec3f(-1.0f));
    om.worldToOctree        = invert(om.octreeToWorld);
    om.octreeToWorldN       = octreeToWorld.getXYZ().inverted().transposed();
    om.cameraPosition       = invert(worldToCamera) * Vec3f(0.f, 0.f, 0.f);
    om.octreeToViewport     = invert(om.viewportToCamera) * invert(om.cameraToOctree);
    om.viewportToOctreeN    = (om.octreeToViewport).transposed();

    // Setup frame-related buffers.

    int numPixels = m_input.frameSize.x * m_input.frameSize.y;
    if (m_pixelTable.getSize() != m_input.frameSize)
    {
        m_indexToPixel.resizeDiscard(numPixels * sizeof(S32));
        m_pixelTable.setSize(m_input.frameSize);
        memcpy(m_indexToPixel.getMutablePtr(), m_pixelTable.getIndexToPixel(), numPixels * sizeof(S32));
    }

    // Coarse frame and pixel buffers.

    int coarseNumPixels = m_input.coarseFrameSize.x * m_input.coarseFrameSize.y;
    m_coarseFrameBuffer.resizeDiscard(coarseNumPixels * sizeof(S32));
    m_input.frameCoarse = m_coarseFrameBuffer.getMutableCudaPtr();

    if (m_coarsePixelTable.getSize() != m_input.coarseFrameSize)
    {
        m_coarseIndexToPixel.resizeDiscard(coarseNumPixels * sizeof(S32));
        m_coarsePixelTable.setSize(m_input.coarseFrameSize);
        memcpy(m_coarseIndexToPixel.getMutablePtr(), m_coarsePixelTable.getIndexToPixel(), coarseNumPixels * sizeof(S32));
        m_coarseIndexToPixel.free(Buffer::CPU);
    }

    // Temp frame buffer for blurring.

    if (m_params.enableBlur)
    {
        // override frame buffer address!
        m_tempFrameBuffer.resizeDiscard(numPixels * sizeof(U32));
        m_input.frame = m_tempFrameBuffer.getMutableCudaPtr();
    }

    // AA sample buffer
    if (m_input.aaRays > 1)
    {
        m_aaSampleBuffer.resizeDiscard(numPixels * m_input.aaRays * sizeof(U32));
        m_input.aaSampleBuffer = m_aaSampleBuffer.getMutableCudaPtr();
    }

    // Setup performance counter buffer.

    if (m_params.enablePerfCounters)
    {
        m_perfCounters.resizeDiscard(m_numWarps * PerfCounter_Max * 33 * sizeof(S64));
        memset(m_perfCounters.getMutablePtr(), 0, (size_t)m_perfCounters.getSize());
        m_input.perfCounters = m_perfCounters.getMutableCudaPtr();
    }

    // Render.

    LaunchResult coarseResult;
    if (m_params.enableBeamOptimization)
    {
        RenderInput old = m_input;
        m_input.numPrimaryRays = coarseNumPixels;
        m_input.aaRays = 1;
        m_input.flags |= RenderFlags_CoarsePass;
        m_input.batchSize = 1;
        m_compiler.undef("ENABLE_CONTOURS");

        coarseResult = launch(coarseNumPixels * m_params.numFrameRepeats, false);

        m_input = old;
        m_input.flags |= RenderFlags_UseCoarseData;
        if (enableContours)
            m_compiler.define("ENABLE_CONTOURS");
    }

    m_input.numPrimaryRays = numPixels * m_input.aaRays;
    LaunchResult renderResult = launch(m_input.numPrimaryRays * m_params.numFrameRepeats, true);

    // Post-process blur.
    F32 blurTime = 0.f;
    if (m_params.enableBlur)
    {
        // restore true frame buffer pointer
        m_input.frame = frame.getBuffer().getMutableCudaPtr();

        // get module
        CudaModule* module = m_compiler.compile();

        // update blur LUT
        Vec4i* pLUT = (Vec4i*)module->getGlobal("c_blurLUT").getMutablePtr();
        for (int i=0; i < m_blurLUT.getSize(); i++)
        {
            float d = sqrtf((float)sqr(m_blurLUT[i].x) + (float)sqr(m_blurLUT[i].y));
            Vec4i& v = pLUT[i];
            v.x = m_blurLUT[i].x;
            v.y = m_blurLUT[i].y;
            v.z = floatToBits((float)m_blurLUT[i].z);
            v.w = floatToBits(d);
        }

        // update globals
        *(RenderInput*)module->getGlobal("c_input").getMutablePtr() = m_input;
        module->setTexRef("texTempFrameIn", m_tempFrameBuffer, CU_AD_FORMAT_UNSIGNED_INT8, 4);
        module->setTexRef("texAASamplesIn", m_aaSampleBuffer, CU_AD_FORMAT_UNSIGNED_INT8, 4);

        // launch
        blurTime = module->getKernel("blurKernel").launchTimed(frame.getSize(), Vec2i(8));

    }

    // Update statistics.

    F32 totalTime           = renderResult.time + coarseResult.time + blurTime;
    m_results.launchTime    += totalTime;
    m_results.coarseTime    += coarseResult.time;
    m_results.renderWarps   += renderResult.numWarps;
    m_results.coarseWarps   += coarseResult.numWarps;

    if (m_params.enablePerfCounters)
    {
        const S64* ptr = (const S64*)m_perfCounters.getPtr();
        for (int warpIdx = 0; warpIdx < m_numWarps; warpIdx++)
        {
            for (int counterIdx = 0; counterIdx < PerfCounter_Max; counterIdx++)
            {
                for (int threadIdx = 0; threadIdx < 32; threadIdx++)
                    m_results.threadCounters[counterIdx] += *ptr++;
                m_results.warpCounters[counterIdx] += *ptr++;
            }
        }
    }

    m_stats = sprintf("CudaRenderer: launch %.2f ms (%.2f FPS), %.2f MPix/s",
        totalTime * 1.0e3f,
        1.0f / totalTime,
        numPixels * 1.0e-6f / totalTime);

    if (m_params.enableBlur)
        m_stats += sprintf(", blur %.2f MPix/s", numPixels * 1.0e-6f / blurTime);

    // Adjust the number of warps for the next run.

    int maxWarps = max(renderResult.numWarps, coarseResult.numWarps);
    if (maxWarps * 2 > m_numWarps)
    {
        if (maxWarps == m_numWarps)
            printf("CudaRenderer: warp count auto-detect overflow, increasing warp count to %d\n", maxWarps * 2);
        else
            printf("CudaRenderer: warp count auto-detected: %d warps, launching %d\n", maxWarps, maxWarps * 2);
        m_numWarps = maxWarps * 2;
    }
    return "";
}

//------------------------------------------------------------------------

String CudaRenderer::renderObject(
    GLContext*      gl,
    OctreeRuntime*  runtime,
    int             objectID,
    const Mat4f&    octreeToWorld,
    const Mat4f&    worldToCamera,
    const Mat4f&    projection)
{
    // Setup framebuffer.

    FW_ASSERT(gl);
    const Vec2i& size = gl->getViewSize();
    int stride = size.x * sizeof(U32);
    m_frameBuffer.resizeDiscard(size.y * stride);
    Image image(size, ImageFormat::ABGR_8888, m_frameBuffer, 0, stride);

    // Render.

    String error = renderObject(image, runtime, objectID, octreeToWorld, worldToCamera, projection);
    if (error.getLength())
        return error;

    // Blit to the screen.

    Mat4f old = gl->setVGXform(Mat4f());
    gl->drawImage(image, Vec2f(0.0f), 0.5f, false);
    gl->setVGXform(old);
    return "";
}

//------------------------------------------------------------------------

void CudaRenderer::clearResults(void)
{
    m_results.launchTime    = 0.0f;
    m_results.coarseTime    = 0.0f;
    m_results.renderWarps   = 0;
    m_results.coarseWarps   = 0;

    for (int i = 0; i < PerfCounter_Max; i++)
    {
        m_results.threadCounters[i] = 0;
        m_results.warpCounters[i] = 0;
    }
}

//------------------------------------------------------------------------

void CudaRenderer::populateCompilerCache(void)
{
    Array<Array<String> > variants;
    for (int i = 0;; i++)
    {
        Array<String> defs;
        if (i & (1 << 0)) defs.add("VOXELATTRIB_AO");
        if (i & (1 << 1)) defs.add("LARGE_RECONSTRUCTION_KERNEL");
        if (i & (1 << 2)) defs.add("ENABLE_SHADOWS");
        if (i & (1 << 3)) break;

        variants.add(defs);
        defs.add("PERSISTENT_THREADS");
        variants.add(defs);
        defs.add("ENABLE_CONTOURS");
        variants.add(defs);
    }

    if (!m_blurLUT.getSize())
        constructBlurLUT();

    for (int i = 0; i < variants.getSize(); i++)
    {
        printf("CudaRenderer: Populating compiler cache... %d/%d\r", i + 1, variants.getSize());
        m_compiler.clearDefines();
        m_compiler.define("KERNEL_RENDER");
        m_compiler.define("VOXELATTRIB_DXT");
        m_compiler.define("JITTER_LOD");
        m_compiler.define("BLUR_LUT_SIZE", String(m_blurLUT.getSize()));
        for (int j = 0; j < variants[i].getSize(); j++)
            m_compiler.define(variants[i][j]);
        m_compiler.compile(false);
        failIfError();
    }
    printf("CudaRenderer: Populating compiler cache... Done.\n");
}

//------------------------------------------------------------------------

CudaRenderer::LaunchResult CudaRenderer::launch(int totalWork, bool persistentThreads)
{
    // Setup warps.

    Vec2i blockSize = Vec2i(RCK_TRACE_BLOCK_WIDTH, RCK_TRACE_BLOCK_HEIGHT);
    int blockThreads = blockSize.x * blockSize.y;
    Vec2i gridSize = Vec2i(0, 1);
    int activeWarpTableSize = 0;

    m_input.totalWork = totalWork;

    if (persistentThreads || m_params.enablePerfCounters)
    {
        m_compiler.define("PERSISTENT_THREADS");
        gridSize.x = (m_numWarps * 32 + blockThreads - 1) / blockThreads;
        activeWarpTableSize = (gridSize.x * gridSize.y) * ((blockThreads + 31) / 32);

        m_activeWarps.resizeDiscard(activeWarpTableSize * sizeof(S32));
        memset(m_activeWarps.getMutablePtr(), 0, activeWarpTableSize * sizeof(S32));
        m_input.activeWarps = m_activeWarps.getMutableCudaPtr();
    }
    else
    {
        m_compiler.undef("PERSISTENT_THREADS");
        gridSize.x = (totalWork + blockThreads - 1) / blockThreads;
    }

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Update globals.

    *(RenderInput*)module->getGlobal("c_input").getMutablePtr() = m_input;
    *(S32*)module->getGlobal("g_warpCounter").getMutablePtr() = 0;
    module->setTexRef("texIndexToPixel", m_indexToPixel, CU_AD_FORMAT_UNSIGNED_INT32, 1);
    module->setTexRef("texFrameCoarseIn", m_coarseFrameBuffer, CU_AD_FORMAT_FLOAT, 1);
    module->setTexRef("texIndexToPixelCoarse", m_coarseIndexToPixel, CU_AD_FORMAT_UNSIGNED_INT32, 1);

    // Launch.

    LaunchResult res;
    res.time = module->getKernel("kernel").launchTimed(gridSize * blockSize, blockSize, (!m_params.measureRaycastPerf));

    // Determine results.

    const S32* activeWarps = (const S32*)m_activeWarps.getPtr();
    res.numWarps = 0;
    for (int i = 0; i < activeWarpTableSize; i++)
        if (activeWarps[i])
            res.numWarps++;
    return res;
}

//------------------------------------------------------------------------

void CudaRenderer::constructBlurLUT(void)
{
    // start by constructing low-discrepancy unit square
    int N = 100; // number of samples
    int S = 24;  // max radius

    // init to random
    Array<Vec2f> pos;
    for (int i=0; i < N; i++)
    {
        Vec2f p;
        p.x = (float)hashBits(i) / (float)(((U64)1)<<32);
        p.y = (float)hashBits(i+N) / (float)(((U64)1)<<32);
        pos.add(p);
    }

    // relax by repulsion force
    Array<Vec2f> force;
    force.resize(N);
    for (int i=0; i < 20; i++)
    {
        for (int j=0; j < N; j++)
        {
            Vec2f f(0.f, 0.f);
            Vec2f p = pos[j];
            for (int k=0; k < N; k++)
            {
                if (k==j)
                    continue;

                Vec2f q = pos[k];
                Vec2f d = p-q;
                if (d.x > .5f)  d.x -= 1.f;
                if (d.y > .5f)  d.y -= 1.f;
                if (d.x < -.5f) d.x += 1.f;
                if (d.y < -.5f) d.y += 1.f;
                float d2 = (d.x*d.x + d.y*d.y);
                float m  = 1.f/d2;
                m *= .0001f;
                f += d.normalized() * m;
            }
            float flen = f.length();
            flen = min(flen, .2f);
            f = f.normalized() * flen;
            force[j] = f;
        }
        for (int j=0; j < N; j++)
        {
            Vec2f p = pos[j];
            p += force[j];
            while (p.x > 1.f) p.x -= 1.f;
            while (p.y > 1.f) p.y -= 1.f;
            while (p.x < 0.f) p.x += 1.f;
            while (p.y < 0.f) p.y += 1.f;
            pos[j] = p;
        }
    }

    // form into a disc
    for (int i=0; i < N; i++)
    {
        Vec2f p = pos[i];
        float an = p.x * 2.f * 3.14159f;
        float d  = p.y; // inverse square weighting
        p.x = d * sinf(an);
        p.y = d * cosf(an);
        p = (p + 1.f) * .5f;
        pos[i] = p;
    }

    // work with pixels from now on
    Array<Vec2i> ipos;
    for (int i=0; i < N; i++)
    {
        Vec2f p = pos[i];
        p.x = min(max(p.x, 0.f), 1.f);
        p.y = min(max(p.y, 0.f), 1.f);
        p = (pos[i] - .5f) * 2.f * (float)S;
        int x = (int)(p.x+.5f);
        int y = (int)(p.y+.5f);
        ipos.add(Vec2i(x, y));
    }

    // trim set to unique pixels
    Set<Vec2i> used;
    Array<Vec2i> iposTemp = ipos;
    ipos.clear();
    for (int i=0; i < N; i++)
    {
        Vec2i p = iposTemp[i];
        if (used.contains(p))
            continue;
        used.add(p);
        ipos.add(p);
    }
    N = ipos.getSize();
    printf("CudaRenderer: blur LUT has %d samples\n", ipos.getSize());

    // sort according to distance
    for (int i=0; i < N; i++)
    {
        int dmin = -1;
        int imin = 0;
        for (int j=i; j < N; j++)
        {
            Vec2i p = ipos[j];
            int d = p.x*p.x + p.y*p.y;
            if (d < dmin || dmin < 0)
            {
                dmin = d;
                imin = j;
            }
        }
        swap(ipos[i], ipos[imin]);
    }

    // construct bitmap for support find
    Array<int> bmap;
    bmap.resize(4*S*S);
    for (int i=0; i < bmap.getSize(); i++)
        bmap[i] = -1;

    // blit seeds
    for (int i=0; i < N; i++)
    {
        Vec2i p = ipos[i];
        p.x += S;
        p.y += S;
        p.x = min(max(p.x, 0), 2*S-1);
        p.y = min(max(p.y, 0), 2*S-1);
        bmap[p.x + 2*S*p.y] = i;
    }

    // fill until full
    for(;;)
    {
        Array<int> bmapTemp = bmap;
        bool changed = false;
        for (int y=0; y < 2*S; y++)
        for (int x=0; x < 2*S; x++)
        {
            int d2 = sqr(x-S) + sqr(y-S);
            if (d2 > S*S)
                continue;

            int b = bmap[x+2*S*y];
            if (b >= 0)
                continue;

            int x0 = max(x-1, 0);
            int x1 = min(x+1, 2*S-1);
            int y0 = max(y-1, 0);
            int y1 = min(y+1, 2*S-1);
            for (int py=y0; py<=y1; py++)
            for (int px=x0; px<=x1; px++)
            {
                int c = bmap[px+2*S*py];
                if (c >= 0)
                    b = c;
            }

            if (b >= 0)
            {
                changed = true;
                bmapTemp[x+2*S*y] = b;
            }
        }

        if (!changed)
            break;

        bmap = bmapTemp;
    }

    // count weights
    Array<int> weight;
    weight.resize(N);
    for (int i=0; i < N; i++)
        weight[i] = 0;
    for (int i=0; i < 4*S*S; i++)
    {
        int b = bmap[i];
        if (b >= 0)
            weight[b]++;
    }

    // construct lut
    for (int i=0; i < N; i++)
        m_blurLUT.add(Vec3i(ipos[i].x, ipos[i].y, weight[i]));

    // convert bitmap into lut
#if 0
    m_blurLUT.clear();
    for (int y=0; y < 2*S; y++)
    for (int x=0; x < 2*S; x++)
    {
        int b = bmap[x+2*S*y];
        if (b < 0)
            continue;

        m_blurLUT.add(Vec3i(x-S, y-S, hashBits(b)));
    }
#endif
}
