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

#include "BenchmarkContext.hpp"
#include "io/StateDump.hpp"

using namespace FW;

//------------------------------------------------------------------------

BenchmarkContext::BenchmarkContext(void)
:   m_file          (NULL),
    m_runtime       (NULL),
    m_renderer      (NULL),

    m_numLevels     (0),

    m_window        (NULL),
    m_image         (NULL)
{
    m_runtime = new OctreeRuntime(MemoryManager::Mode_Cuda);
    m_renderer = new CudaRenderer;
    failIfError();

    m_window = new Window;
    m_window->setVisible(false);
    m_window->setTitle("Octree benchmark");
    m_window->addListener(this);
    m_window->realize();
}

//------------------------------------------------------------------------

BenchmarkContext::~BenchmarkContext(void)
{
    delete m_file;
    delete m_runtime;
    delete m_renderer;
    delete m_window;
    delete m_image;
}

//------------------------------------------------------------------------

void BenchmarkContext::setFile(const String& fileName)
{
    // Already current => done.

    if (m_file && m_file->getName() == fileName)
        return;

    // Open file.

    delete m_file;
    m_file = new OctreeFile(fileName, File::Read);
    failIfError();

    // Clear state.

    m_numLevels = 0;
    clearRuntime();
}

//------------------------------------------------------------------------

void BenchmarkContext::clearRuntime(void)
{
    // Clear.

    m_runtime->clear();
    if (!m_file)
        return;

    // Reintroduce objects.

    for (int i = 0; i < m_file->getNumObjects(); i++)
    {
        const OctreeFile::Object& obj = m_file->getObject(i);
        if (obj.rootSlice == -1)
            continue;

        Array<AttachIO::AttachType> attach;
        m_renderer->selectAttachments(attach, obj.runtimeAttachTypes);
        m_runtime->addObject(i, obj.rootSlice, attach);
    }
}

//------------------------------------------------------------------------

void BenchmarkContext::load(int numLevels)
{
    // Already loaded => done.

    if (!m_file || m_numLevels == numLevels)
        return;

    clearRuntime();
    printf("Loading %d levels from '%s'...\r", numLevels, m_file->getName().getPtr());

    // Queue root slices.

    Array<Vec2i> queue;
    for (int i = 0; i < m_file->getNumObjects(); i++)
        queue.add(Vec2i(m_file->getObject(i).rootSlice, 0));

    // Load slices asynchronously.

    Timer timer(true);
    int slicesDone = 0;
    int progress = -1;

    for (int i = 0; i < queue.getSize() && !hasError(); i++)
    {
        // Prefetch slices.

        int totalBytes = 0;
        for (int j = i; j < queue.getSize(); j++)
        {
            if (queue[j].y >= numLevels || m_file->getSliceState(queue[j].x) != OctreeFile::SliceState_Complete)
                continue;

            totalBytes += m_file->getSliceSize(queue[j].x);
            if (totalBytes > MaxPrefetchBytesTotal)
                break;

            m_file->readSlicePrefetch(queue[j].x);
        }

        // Print status and ignore non-complete slices.

        int newProgress = (int)(100.0f * (F32)slicesDone / (F32)m_file->getNumSliceIDs() + 0.5f);
        if (newProgress != progress)
        {
            progress = newProgress;
            printf("Loading %d levels from '%s'... %d%%\r",
                numLevels, m_file->getName().getPtr(), progress);
        }

        if (queue[i].x >= 0)
            slicesDone++;
        if (queue[i].y >= numLevels || m_file->getSliceState(queue[i].x) != OctreeFile::SliceState_Complete)
            continue;

        // Read from file and load to runtime.

        OctreeSlice slice;
        m_file->readSlice(queue[i].x, slice);
        if (!m_runtime->loadSlice(slice))
            setError("OctreeRuntime ran out of memory!");

        // Queue children.

        for (int j = 0; j < slice.getNumChildEntries(); j++)
            queue.add(Vec2i(slice.getChildEntry(j), queue[i].y + 1));
    }

    // Done.

    m_numLevels = numLevels;
    failIfError();

    printf("Loaded %d levels from '%s' in %s.\n",
        numLevels,
        m_file->getName().getPtr(),
        formatTime(timer.getElapsed()).getPtr());

    printf("%s\n", m_runtime->getStats().getPtr());
}

//------------------------------------------------------------------------

Mat4f BenchmarkContext::getOctreeToWorld(int objectID) const
{
    if (!m_file)
        return Mat4f();

    const OctreeFile::Object& obj = m_file->getObject(objectID);
    return obj.objectToWorld * obj.octreeToObject;
}

//------------------------------------------------------------------------

Mat4f BenchmarkContext::getProjection(const Vec2i& frameSize) const
{
    return Mat4f::fitToView(Vec2f(-1.0f, -1.0f), Vec2f(2.0f, 2.0f), Vec2f(frameSize)) * m_camera.getCameraToClip();
}

//------------------------------------------------------------------------

void BenchmarkContext::renderOctree(Image& image, int objectID) const
{
    String error = m_renderer->renderObject(image, m_runtime, objectID,
        getOctreeToWorld(), getWorldToCamera(), getProjection(image.getSize()));

    if (error.getLength())
        fail("%s", error.getPtr());
}

//------------------------------------------------------------------------

void BenchmarkContext::showImage(Image& image)
{
    if (m_image)
        delete m_image;
    m_image = new Image(image.getSize(), ImageFormat::R8_G8_B8_A8);
    *m_image = image;

    m_window->setSize(image.getSize());
    m_window->setVisible(true);
    Window::pollMessages();
    for (int i = 0; i < 3; i++)
        m_window->repaintNow();
    Window::pollMessages();
}

//------------------------------------------------------------------------

void BenchmarkContext::showOctree(const Vec2i& frameSize, int objectID)
{
    Image image(frameSize, ImageFormat::ABGR_8888);
    renderOctree(image, objectID);
    showImage(image);
}

//------------------------------------------------------------------------

void BenchmarkContext::hideWindow(void)
{
    if (m_window)
        m_window->setVisible(false);
    Window::pollMessages();
}

//------------------------------------------------------------------------

bool BenchmarkContext::handleEvent(const Window::Event& ev)
{
    if (ev.type != Window::EventType_Paint)
        return false;

    GLContext* gl = m_window->getGL();
    gl->setVGXform(gl->xformMatchPixels());
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    gl->drawImage(*m_image, Vec2f(0.0f), Vec2f(0.0f), false);
    return false;
}

//------------------------------------------------------------------------

BuilderBase::Params BenchmarkContext::readBuildParams(const String& stateFile)
{
    StateDump dump;
    {
        File file(stateFile, File::Read);
        char tag[9];
        file.readFully(tag, 8);
        tag[8] = 0;
        if (String(tag) != "FWState ")
            setError("Invalid state file!");
        file >> dump;
    }

    dump.pushOwner("App");
    BuilderBase::Params params;
    dump.get(params.enableVariableResolution,   "m_variableResolution");
    dump.get(params.colorDeviation,             "m_colorDeviation");
    dump.get(params.normalDeviation,            "m_normalDeviation");
    dump.get(params.contourDeviation,           "m_contourDeviation");
    dump.get((S32&)params.filter,               "m_filterType");
    dump.get((S32&)params.shaper,               "m_shaperType");
    dump.popOwner();

    params.colorDeviation /= 256.0f;
    params.contourDeviation = pow(2.0f, (F32)OctreeFile::UnitScale - params.contourDeviation) * sqrt(3.0f);

    failIfError();
    return params;
}

//------------------------------------------------------------------------
