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

#include "App.hpp"
#include "base/Main.hpp"
#include "gpu/GLContext.hpp"
#include "3d/Mesh.hpp"
#include "io/StateDump.hpp"
#include "build/MeshBuilder.hpp"
#include "AmbientProcessor.hpp"
#include "Benchmark.hpp"

#include <stdio.h>
#include <conio.h>

using namespace FW;

//------------------------------------------------------------------------

static const char* const s_initialMeshDir       = "scenes/default";
static const char* const s_initialOctreeDir     = "octrees";
static const char* const s_defaultMeshFile      = "scenes/default/default.obj";
static const char* const s_defaultOctreeFile    = "octrees/default_11_ao.oct";
static const char* const s_tempOctreeFile       = "octrees/tmp.oct";
static const char* const s_defaultCamera        = "5O12/0p7Ayz/dlDNz13lQngy17ssby/8UJJJz////X108Qx7w/6//m100";

//------------------------------------------------------------------------

static const char* const s_aboutText =
    "Open-source implementation of\n"
    "\"Efficient Sparse Voxel Octrees\", presented at I3D 2010.\n"
    "\n"
    "Samuli Laine, Tero Karras\n"
    "Copyright 2009-2012 NVIDIA Corporation\n"
    "\n"
    "http://code.google.com/p/efficient-sparse-voxel-octrees/\n"
;

//------------------------------------------------------------------------

static const char* const s_commandHelpText =
    "\n"
    "Usage: octree <mode> [options]\n"
    "Implementation of \"Efficient Sparse Voxel Octrees\",\n"
    "presented at I3D 2010.\n"
    "\n"
    "The following values for <mode> are supported:\n"
    "\n"
    "   interactive             View octree files interactively.\n"
    "   build                   Build octree file.\n"
    "   inspect                 Print info about octree file.\n"
    "   ambient                 Augment octree file with ambient occlusion data.\n"
    "   optimize                Reconstruct octree file to improve performance.\n"
    "   benchmark               Run benchmarks.\n"
    "\n"
    "Common options:\n"
    "\n"
    "   --log=<file.log>        Log all output to file.\n"
    "\n"
    "Options for \"octree interactive\":\n"
    "\n"
    "   --size=<w>x<h>          Initial window size. Default is \"1024x768\".\n"
    "   --state=<file.dat>      Load state from the given file.\n"
    "   --in=<file.oct>         Load octree from the given file.\n"
    "   --max-threads=<num>     Maximum concurrent builder threads. Default is \"4\".\n"
    "\n"
    "Options for \"octree build\":\n"
    "\n"
    "   --in=<file.obj/oct>     Input mesh. Specify octree to reuse the mesh embedded in it.\n"
    "   --out=<file.oct>        Output octree file.\n"
    "   --levels=<value>        Max octree levels to build.\n"
    "   --contours=<1/0>        Enable/disable contours. Default is \"1\".\n"
    "   --color-error=<value>   Max color error. Default is \"16\" RGB values.\n"
    "   --normal-error=<value>  Max normal error. Default is \"0.01\" units.\n"
    "   --contour-error=<value> Max contour error. Default is \"15\" levels.\n"
    "   --max-threads=<num>     Maximum concurrent builder threads. Default is \"4\".\n"
    "\n"
    "Options for \"octree inspect\":\n"
    "\n"
    "   --in=<file.oct>         Input octree file.\n"
    "\n"
    "Options for \"octree ambient\":\n"
    "\n"
    "   --in=<file.oct>         Input octree file. Modified in place.\n"
    "   --ao-radius=<value>     AO ray length, relative to the scene. Default is \"0.05\".\n"
    "   --flip-normals=<1/0>    Enable/disable flipping of normals. Default is \"0\".\n"
    "\n"
    "Options for \"octree optimize\":\n"
    "\n"
    "   --in=<file.oct>         Input octree file.\n"
    "   --out=<file.oct>        Output octree file.\n"
    "   --levels=<value>        Include only the given number of levels.\n"
    "   --include-mesh=<1/0>    Include/exclude original mesh. Default is \"1\".\n"
    "\n"
    "Options for \"octree benchmark\":\n"
    "\n"
    "   --in=<file.oct>         Input octree file.\n"
    "   --levels=<value>        Max octree levels to load.\n"
    "   --size=<w>x<h>          Frame size. Default is \"1024x768\".\n"
    "   --frames-per-launch=<v> Frames to execute within a single launch. Default is \"10\".\n"
    "   --warmup-launches=<v>   Launches prior to starting the measurement. Default is \"4\".\n"
    "   --measure-frames=<v>    Total number of frames to measure. Default is \"2000\".\n"
    "   --camera=\"<v>\"        Camera signature. Can specify multiple times.\n"
;

//------------------------------------------------------------------------

static const char* const s_guiHelpText =
    "Click \"Show view/camera/build controls\" to enable advanced functionality, or press Tab.\n"
    "Press F1 to hide this message.\n"
    "\n"
    "General keys:\n"
    "\n"
    "\tEsc\tExit (also Alt-F4)\n"
    "\tNum\tLoad numbered state\n"
    "\tAlt-Num\tSave numbered state\n"
    "\tF9\tShow/hide FPS counter\n"
    "\tF10\tShow/hide GUI\n"
    "\tF11\tToggle fullscreen\n"
    "\tPrtScn\tSave screenshot\n"
    "\n"
    "Camera movement:\n"
    "\n"
    "\tDrag\tRotate (left), strafe (middle)\n"
    "\tArrows\tRotate\n"
    "\tW\tMove forward (also Alt-UpArrow)\n"
    "\tS\tMove back (also Alt-DownArrow)\n"
    "\tA\tStrafe left (also Alt-LeftArrow)\n"
    "\tD\tStrafe right (also Alt-RightArrow)\n"
    "\tR\tStrafe up (also PageUp)\n"
    "\tF\tStrafe down (also PageDown)\n"
    "\tWheel\tAdjust movement speed\n"
    "\tSpace\tMove faster (hold)\n"
    "\tCtrl\tMove slower (hold)\n"
    "\n"
    "Uncheck \"Retain camera alignment\" to enable:\n"
    "\n"
    "\tQ\tRoll counter-clockwise (also Insert)\n"
    "\tE\tRoll clockwise (also Home)\n"
;

//------------------------------------------------------------------------

App::App(void)
:   m_commonCtrl                    (CommonControls::Feature_Default & ~CommonControls::Feature_RepaintOnF5),
    m_cameraCtrl                    (&m_commonCtrl, CameraControls::Feature_Default & ~CameraControls::Feature_StereoControls),

    m_action                        (Action_None),

    m_activeView                    (View_Primary),
    m_disableContourTest            (false),
    m_disableBeamOptimization       (false),
    m_disablePostProcessFiltering   (false),
    m_enableAntialias               (false),
    m_enableLargeAAFilter           (false),
    m_maxVoxelSize                  (1.0f),
    m_brightness                    (1.7f),

    m_maxOctreeLevels               (16),
    m_buildWithoutContours          (false),
    m_buildColorError               (16.0f),
    m_buildNormalError              (0.01f),
    m_buildContourError             (15.0f),

    m_showHelp                      (false),
    m_showViewControls              (false),
    m_showCameraControls            (false),
    m_showManagementControls        (false),
    m_viewControlsVisible           (false),
    m_cameraControlsVisible         (false),
    m_managementControlsVisible     (false)
{
    m_commonCtrl.showFPS(true);
    m_commonCtrl.setStateFilePrefix("state_octree_");
    m_commonCtrl.addStateObject(this);

    m_cameraCtrl.setKeepAligned(true);

    m_window.setTitle("Octree");
    m_window.addListener(&m_cameraCtrl);
    m_window.addListener(this);
    m_window.addListener(&m_commonCtrl);

    rebuildGui();
}

//------------------------------------------------------------------------

App::~App(void)
{
}

//------------------------------------------------------------------------

bool App::handleEvent(const Window::Event& ev)
{
    // Window closed => destroy app.

    if (ev.type == Window::EventType_Close)
    {
        printf("Exiting...\n");
        m_window.showModalMessage("Exiting...");
        delete this;
        return true;
    }

    // Update GUI controls.

    if (ev.type == Window::EventType_KeyDown && ev.key == FW_KEY_TAB)
    {
        bool v = (!m_showViewControls || !m_showCameraControls || !m_showManagementControls);
        m_showViewControls = v;
        m_showCameraControls = v;
        m_showManagementControls = v;
    }

    if (m_showViewControls != m_viewControlsVisible ||
        m_showCameraControls != m_cameraControlsVisible ||
        m_showManagementControls != m_managementControlsVisible)
    {
        m_viewControlsVisible = m_showViewControls;
        m_cameraControlsVisible = m_showCameraControls;
        m_managementControlsVisible = m_showManagementControls;
        rebuildGui();
    }

    // Handle actions.

    Action action = m_action;
    m_action = Action_None;
    String name;

    switch (action)
    {
    case Action_None:
        break;

    case Action_About:
        m_window.showMessageDialog("About", s_aboutText);
        break;

    case Action_LoadState:
        m_commonCtrl.loadStateDialog();
        break;

    case Action_SaveState:
        m_commonCtrl.saveStateDialog();
        break;

    case Action_ResetCamera:
        resetCamera();
        break;

    case Action_ImportCameraSignature:
        {
            m_window.setVisible(false);
            printf("\nEnter camera signature:\n");

            char buf[1024];
            if (scanf_s("%s", buf, FW_ARRAY_SIZE(buf)))
                m_cameraCtrl.decodeSignature(buf);
            else
                setError("Signature too long!");

            if (!hasError())
                printf("Done.\n\n");
            else
            {
                printf("Error: %s\n", getError().getPtr());
                clearError();
                waitKey();
            }
        }
        break;

    case Action_ExportCameraSignature:
        m_window.setVisible(false);
        printf("\nCamera signature:\n");
        printf("%s\n", m_cameraCtrl.encodeSignature().getPtr());
        waitKey();
        break;

    case Action_LoadOctree:
        name = m_window.showFileLoadDialog("Load octree", "oct:Octree", s_initialOctreeDir);
        if (name.getLength() && loadOctree(name))
            resetCamera();
        break;

    case Action_SaveOctree:
        name = m_window.showFileSaveDialog("Save octree", "oct:Octree", s_initialOctreeDir);
        if (name.getLength())
        {
            m_window.showModalMessage(sprintf("Saving octree to '%s'...", name.getPtr()));
            m_manager.saveFile(name);
            m_octreeFileName = name;
            m_commonCtrl.message(sprintf("Saved octree to '%s'", name.getPtr()));
        }
        break;

    case Action_NewOctreeFromMesh:
        name = m_window.showFileLoadDialog("Import mesh", getMeshImportFilter(), s_initialMeshDir);
        if (name.getLength())
        {
            m_window.showModalMessage(sprintf("Importing mesh from '%s'...\nThis will take a few seconds.", name.getFileName().getPtr()));
            MeshBase* mesh = importMesh(name);
            failIfError();
            if (mesh)
            {
                m_window.showModalMessage("Initializing temporary octree file...\nThis will take a few seconds.");
                m_manager.newFile();
                m_manager.addMesh(mesh, OctreeManager::BuilderType_Mesh, getBuilderParams());
                resetCamera();
                m_octreeFileName = "";
                m_commonCtrl.message(sprintf("Imported mesh from '%s'", name.getPtr()));
            }
        }
        break;

    case Action_RebuildOctree:
        m_window.showModalMessage("Initializing temporary octree file...\nThis will take a few seconds.");
        m_manager.rebuildFile(OctreeManager::BuilderType_Mesh, getBuilderParams());
        break;

    case Action_PrintStats:
        m_window.setVisible(false);
        printf("\nStatistics for '%s':\n", m_manager.getFile()->getName().getPtr());
        m_manager.getFile()->printStats();
        waitKey();
        break;

    default:
        FW_ASSERT(false);
        break;
    }

    // Repaint.

    m_window.setVisible(true);
    if (ev.type == Window::EventType_Paint)
        render(m_window.getGL());
    m_window.repaint();
    return false;
}

//------------------------------------------------------------------------

void App::readState(StateDump& d)
{
    d.pushOwner("App");

    String octreeFileName;
    d.get(octreeFileName,                   "m_octreeFileName");
    if (m_octreeFileName != octreeFileName && octreeFileName.getLength())
        loadOctree(octreeFileName);

    d.get((S32&)m_activeView,               "m_activeView");
    d.get(m_disableContourTest,             "m_disableContourTest");
    d.get(m_disableBeamOptimization,        "m_disableBeamOptimization");
    d.get(m_disablePostProcessFiltering,    "m_disablePostProcessFiltering");
    d.get(m_enableAntialias,                "m_enableAntialias");
    d.get(m_enableLargeAAFilter,            "m_enableLargeAAFilter");
    d.get(m_maxVoxelSize,                   "m_maxVoxelSize");
    d.get(m_brightness,                     "m_brightness");

    d.get(m_maxOctreeLevels,                "m_maxOctreeLevels");
    d.get(m_buildWithoutContours,           "m_buildWithoutContours");
    d.get(m_buildColorError,                "m_buildColorError");
    d.get(m_buildNormalError,               "m_buildNormalError");
    d.get(m_buildContourError,              "m_buildContourError");

    d.popOwner();
}

//------------------------------------------------------------------------

void App::writeState(StateDump& d) const
{
    d.pushOwner("App");

    d.set(m_octreeFileName,                 "m_octreeFileName");

    d.set((S32)m_activeView,                "m_activeView");
    d.set(m_disableContourTest,             "m_disableContourTest");
    d.set(m_disableBeamOptimization,        "m_disableBeamOptimization");
    d.set(m_disablePostProcessFiltering,    "m_disablePostProcessFiltering");
    d.set(m_enableAntialias,                "m_enableAntialias");
    d.set(m_enableLargeAAFilter,            "m_enableLargeAAFilter");
    d.set(m_maxVoxelSize,                   "m_maxVoxelSize");
    d.set(m_brightness,                     "m_brightness");

    d.set(m_maxOctreeLevels,                "m_maxOctreeLevels");
    d.set(m_buildWithoutContours,           "m_buildWithoutContours");
    d.set(m_buildColorError,                "m_buildColorError");
    d.set(m_buildNormalError,               "m_buildNormalError");
    d.set(m_buildContourError,              "m_buildContourError");

    d.popOwner();
}

//------------------------------------------------------------------------

bool App::loadOctree(const String& fileName)
{
    m_window.showModalMessage(sprintf("Loading octree from '%s'...\nThis will take a few seconds.", fileName.getFileName().getPtr()));

    String oldError = clearError();
    m_manager.loadFile(fileName);
    String newError = getError();

    if (restoreError(oldError))
    {
        m_commonCtrl.message(sprintf("Error while loading '%s': %s", fileName.getPtr(), newError.getPtr()));
        return false;
    }

    m_commonCtrl.message(sprintf("Loaded octree from '%s'", fileName.getPtr()));
    m_octreeFileName = fileName;
    return true;
}

//------------------------------------------------------------------------

void App::resetCamera(void)
{
    if (m_manager.getFile()->getNumObjects())
    {
        MeshBase* mesh = m_manager.getFile()->getMesh(0);
        if (mesh)
        {
            m_cameraCtrl.initForMesh(mesh);
            m_commonCtrl.message("Camera reset");
        }
    }
}

//------------------------------------------------------------------------

void App::rebuildGui(void)
{
    CommonControls& cc = m_commonCtrl;
    cc.resetControls();

    cc.setControlVisibility(true);
    cc.addToggle(&m_showHelp,                                   FW_KEY_F1,          "Show help (F1)");
    cc.addButton((S32*)&m_action, Action_About,                 FW_KEY_NONE,        "About...");
    cc.addButton((S32*)&m_action, Action_LoadState,             FW_KEY_NONE,        "Load state... [0]");
    cc.addButton((S32*)&m_action, Action_SaveState,             FW_KEY_NONE,        "Save state... [Alt-0]");
    cc.addSeparator();

    cc.setControlVisibility(m_showViewControls);
    cc.addToggle((S32*)&m_activeView, View_Primary,             FW_KEY_F2,          "Default view [F2]");
    cc.addToggle((S32*)&m_activeView, View_PrimaryAndShadow,    FW_KEY_F3,          "View with shadows [F3]");
    cc.addToggle((S32*)&m_activeView, View_OriginalMesh,        FW_KEY_F4,          "View original mesh [F4]");
    cc.addToggle((S32*)&m_activeView, View_OctreeDepth,         FW_KEY_F5,          "View octree depth [F5]");
    cc.addToggle((S32*)&m_activeView, View_IterationCount,      FW_KEY_F6,          "View raycast iteration count [F6]");
    cc.addSeparator();

    cc.setControlVisibility(m_showViewControls);
    cc.addToggle(&m_disableContourTest,                         FW_KEY_Z,           "Disable contour test [Z]");
    cc.addToggle(&m_disableBeamOptimization,                    FW_KEY_X,           "Disable beam optimization [X]");
    cc.addToggle(&m_disablePostProcessFiltering,                FW_KEY_C,           "Disable post-process filtering [C]");
    cc.addToggle(&m_enableAntialias,                            FW_KEY_V,           "Enable 4x antialiasing [V]");
    cc.addToggle(&m_enableLargeAAFilter,                        FW_KEY_B,           "Enable large antialias filter [B]");
    cc.beginSliderStack();
    cc.addSlider(&m_maxVoxelSize, 0.1f, 10.0f, true, FW_KEY_NONE, FW_KEY_NONE,      "Maximum voxel size = %g pixels");
    cc.addSlider(&m_brightness, 0.0f, 5.0f, false, FW_KEY_NONE, FW_KEY_NONE,        "Brightness coefficient = %g");
    cc.endSliderStack();
    cc.addSeparator();

    cc.setControlVisibility(m_showCameraControls);
    cc.addButton((S32*)&m_action, Action_ResetCamera,           FW_KEY_NONE,        "Reset camera");
    cc.addButton((S32*)&m_action, Action_ImportCameraSignature, FW_KEY_NONE,        "Import camera signature...");
    cc.addButton((S32*)&m_action, Action_ExportCameraSignature, FW_KEY_NONE,        "Export camera signature...");
    m_cameraCtrl.addGUIControls();
    cc.addSeparator();

    cc.setControlVisibility(m_showManagementControls);
    cc.addButton((S32*)&m_action, Action_LoadOctree,            FW_KEY_I,           "Load octree... [I]");
    cc.addButton((S32*)&m_action, Action_SaveOctree,            FW_KEY_O,           "Save octree... [O]");
    cc.addButton((S32*)&m_action, Action_NewOctreeFromMesh,     FW_KEY_M,           "New octree from mesh... [M]");
    cc.addButton((S32*)&m_action, Action_RebuildOctree,         FW_KEY_BACKSPACE,   "Rebuild octree [Backspace]");
    cc.addButton((S32*)&m_action, Action_PrintStats,            FW_KEY_P,           "Print octree stats... [P]");
    cc.addSlider(&m_maxOctreeLevels, 1, OctreeFile::UnitScale, false, FW_KEY_NONE, FW_KEY_NONE, "Maximum octree levels to load/build = %d levels");

    cc.setControlVisibility(m_showManagementControls);
    cc.addToggle(&m_buildWithoutContours,                       FW_KEY_NONE,        "Build without contours (Backspace to rebuild)");
    cc.beginSliderStack();
    cc.addSlider(&m_buildColorError, 1.0f, 256.0f, true, FW_KEY_NONE, FW_KEY_NONE, "Allowed color approximation error (Backspace to rebuild) = %g RGB values");
    cc.addSlider(&m_buildNormalError, 1.0e-4f, 2.0f, true, FW_KEY_NONE, FW_KEY_NONE, "Allowed normal approximation error (Backspace to rebuild) = %g normalized units");
    cc.addSlider(&m_buildContourError, 0.0f, 32.0f, false, FW_KEY_NONE, FW_KEY_NONE, "Allowed contour approximation error (Backspace to rebuild) = equivalent of %g levels");
    cc.endSliderStack();
    cc.addSeparator();

    cc.setControlVisibility(true);
    cc.addToggle(&m_showViewControls,                           FW_KEY_NONE,        "Show visualization controls");
    cc.addToggle(&m_showCameraControls,                         FW_KEY_NONE,        "Show camera controls");
    cc.addToggle(&m_showManagementControls,                     FW_KEY_NONE,        "Show octree management controls");
}

//------------------------------------------------------------------------

void App::waitKey(void)
{
    printf("Press any key to continue . . . ");
    _getch();
    printf("\n\n");
}

//------------------------------------------------------------------------

void App::render(GLContext* gl)
{
    // Determine parameters.

    OctreeManager::RenderMode renderMode;

    CudaRenderer::Params cudaParams;
    cudaParams.enableContours               = (!m_disableContourTest);
    cudaParams.enableAntialias              = m_enableAntialias;
    cudaParams.enableLargeReconstruction    = m_enableLargeAAFilter;
    cudaParams.enableJitterLOD              = true;
    cudaParams.enableBeamOptimization       = (!m_disableBeamOptimization);
    cudaParams.enableBlur                   = (!m_disablePostProcessFiltering);
    cudaParams.maxVoxelSize                 = m_maxVoxelSize;
    cudaParams.brightness                   = m_brightness;

    switch (m_activeView)
    {
    case View_Primary:
        renderMode = OctreeManager::RenderMode_Cuda;
        cudaParams.visualization = CudaRenderer::Visualization_Primary;
        break;

    case View_PrimaryAndShadow:
        renderMode = OctreeManager::RenderMode_Cuda;
        cudaParams.visualization = CudaRenderer::Visualization_PrimaryAndShadow;
        break;

    case View_OriginalMesh:
        {
            renderMode = OctreeManager::RenderMode_Mesh;
            GLContext::Config glConfig = m_window.getGLConfig();
            glConfig.numSamples = (m_enableAntialias) ? 4 : 0;
            m_window.setGLConfig(glConfig);
        }
        break;

    case View_OctreeDepth:
        renderMode = OctreeManager::RenderMode_Cuda;
        cudaParams.visualization = CudaRenderer::Visualization_RaycastLevel;
        break;

    case View_IterationCount:
        renderMode = OctreeManager::RenderMode_Cuda;
        cudaParams.visualization = CudaRenderer::Visualization_IterationCount;
        break;

    default:
        FW_ASSERT(false);
        return;
    }

    // Set parameters.

    m_manager.setRenderMode(renderMode);
    m_manager.setMaxLevels(m_maxOctreeLevels);

    CudaRenderer* cuda = m_manager.getCudaRenderer();
    if (cuda)
    {
        cuda->setParams(cudaParams);
        cuda->setWindow(&m_window);
    }

    // Render.

    Mat4f projection = gl->xformFitToView(Vec2f(-1.0f, -1.0f), Vec2f(2.0f, 2.0f)) * m_cameraCtrl.getCameraToClip();
    Mat4f worldToCamera = m_cameraCtrl.getWorldToCamera();
    m_manager.renderObject(gl, 0, worldToCamera, projection);

    // Show statistics.

    String memoryStats = sprintf("Memory used: host %d megs, device %d megs",
        (S32)(getMemoryUsed() >> 20),
        (S32)(CudaModule::getMemoryUsed() >> 20));

    OctreeRuntime* runtime = m_manager.getRuntime();
    m_commonCtrl.message((runtime) ? runtime->getStats() : "", "OctreeRuntimeStats");
    m_commonCtrl.message((cuda) ? cuda->getStats() : "", "CudaRendererStats");
    m_commonCtrl.message(m_manager.getStats(), "OctreeManagerStats");

    // Show help.

    if (m_showHelp)
        renderGuiHelp(gl);
}

//------------------------------------------------------------------------

void App::renderGuiHelp(GLContext* gl)
{
    S32 fontSize = 16;
    F32 tabSize = 64.0f;

    Mat4f oldXform = gl->setVGXform(gl->xformMatchPixels());
    gl->setFont("Arial", fontSize, GLContext::FontStyle_Bold);
    Vec2f origin = Vec2f(8.0f, (F32)gl->getViewSize().y - 4.0f);

    String str = s_guiHelpText;
    int startIdx = 0;
    Vec2f pos = 0.0f;
    while (startIdx < str.getLength())
    {
        if (str[startIdx] == '\n')
            pos = Vec2f(0.0f, pos.y - (F32)fontSize);
        else if (str[startIdx] == '\t')
            pos.x += tabSize;

        int endIdx = startIdx;
        while (endIdx < str.getLength() && str[endIdx] != '\n' && str[endIdx] != '\t')
            endIdx++;

        gl->drawLabel(str.substring(startIdx, endIdx), pos + origin, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
        startIdx = max(endIdx, startIdx + 1);
    }

    gl->setVGXform(oldXform);
    gl->setDefaultFont();
}

//------------------------------------------------------------------------

BuilderBase::Params App::getBuilderParams(void)
{
    BuilderBase::Params params;
    params.colorDeviation = m_buildColorError / 256.0f;
    params.normalDeviation = m_buildNormalError;
    params.setContourDeviationForLevels(m_buildContourError);
    params.shaper = (m_buildWithoutContours) ? BuilderBase::Shaper_None : BuilderBase::Shaper_Hull;
    return params;
}

//------------------------------------------------------------------------

void App::firstTimeInit(void)
{
    // Error has occurred => skip.

    if (hasError())
        return;

    // CUDA not available => skip.

    if (!CudaModule::isAvailable())
        return;

    // Check that CudaCompiler is able to auto-detect paths.

    CudaCompiler::staticInit();

    // Mesh file does not exist => skip.
    {
        File file(s_defaultMeshFile, File::Read);
        if (hasError())
        {
            clearError();
            return;
        }
    }

    // Print header.

    printf("Performing first-time initialization.\n");
    printf("This will take a while.\n");
    printf("\n");

    // Populate CudaCompiler cache.

    CudaRenderer().populateCompilerCache();

    // Octree does not exist => build it.

    bool needToBuild = false;
    {
        OctreeFile file(s_defaultOctreeFile, File::Read);
        if (hasError())
        {
            clearError();
            needToBuild = true;
        }
    }

    if (needToBuild)
    {
        runBuild(s_defaultMeshFile, s_tempOctreeFile, 11, true, 16.0f, 0.01f, 15.0f, 4);
        runAmbient(s_tempOctreeFile, 0.15f, false);
        runOptimize(s_tempOctreeFile, s_defaultOctreeFile, 0, true);
    }

    // Setup default state.

    printf("Setting up default state...\n");
    loadOctree(s_defaultOctreeFile);
    m_cameraCtrl.decodeSignature(s_defaultCamera);
    m_commonCtrl.saveState(m_commonCtrl.getStateFileName(1));
    failIfError();

    // Print footer.

    printf("Done.\n");
    printf("\n");
}

//------------------------------------------------------------------------

void FW::runInteractive(const Vec2i& frameSize, const String& stateFile, const String& inFile, int maxThreads)
{
    if (hasError())
        return;

    // Launch.

    printf("Starting up...\n");
    App* app = new App;
    app->setWindowSize(frameSize);
    app->setMaxConcurrency(maxThreads);

    // Load state.

    if (!hasError() && !stateFile.getLength())
        app->loadDefaultState();
    else if (!hasError() && !app->loadState(stateFile))
        setError("Unable to load state from '%s'!", stateFile.getPtr());

    // Load octree.

    if (!hasError() && inFile.getLength())
    {
        if (!app->loadOctree(inFile))
            setError("Unable to load octree from '%s'!", inFile.getPtr());
        else if (!stateFile.getLength())
        {
            printf("Resetting camera...\n");
            app->resetCamera();
        }
    }

    // Error => close window.

    if (hasError())
        delete app;
    else
        app->flashButtonTitles();
}

//------------------------------------------------------------------------

void FW::runBuild(const String& inFile, const String& outFile, int numLevels, bool buildContours, F32 colorError, F32 normalError, F32 contourError, int maxThreads)
{
    if (hasError())
        return;

    // Import mesh.

    printf("Importing mesh from '%s'...\n", inFile.getPtr());
    MeshBase* mesh = NULL;
    if (!inFile.toLower().endsWith(".oct"))
        mesh = importMesh(inFile);
    else
    {
        OctreeFile src(inFile, File::Read);
        if (src.getNumObjects())
            mesh = src.getMeshCopy(0);
    }

    if (!mesh)
        setError("Unable to import mesh from '%s'!", inFile.getPtr());

    if (hasError())
    {
        delete mesh;
        return;
    }

    // Create octree.

    printf("Building octree to '%s'...\n", outFile.getPtr());
    OctreeFile file(outFile, File::Create);
    int objectID = file.addObject();

    if (!hasError())
        file.setMesh(objectID, mesh);
    else
        delete mesh;

    if (hasError())
        return;

    // Build.

    BuilderBase::Params params;
    params.colorDeviation = colorError / 256.0f;
    params.normalDeviation = normalError;
    params.setContourDeviationForLevels(contourError);
    params.shaper = (buildContours) ? BuilderBase::Shaper_Hull : BuilderBase::Shaper_None;

    MeshBuilder builder(&file);
    builder.setMaxConcurrency(maxThreads);
    builder.buildObject(objectID, numLevels, params);
}

//------------------------------------------------------------------------

void FW::runInspect(const String& inFile)
{
    if (!hasError())
    {
        OctreeFile file(inFile, File::Read);
        if (!hasError())
        {
            printf("\nStatistics for '%s':\n", inFile.getPtr());
            file.printStats();
        }
    }
}

//------------------------------------------------------------------------

void FW::runAmbient(const String& inFile, F32 aoRadius, bool flipNormals)
{
    if (hasError())
        return;

    // Open file.

    printf("Computing AO for '%s'...\n", inFile.getPtr());
    OctreeFile file(inFile, File::Modify);
    if (hasError())
        return;

    // Calculate AO.

    AmbientProcessor proc(&file, 0);
    proc.setRayLength(aoRadius);
    proc.setFlipNormals(flipNormals);
    proc.run();
}

//------------------------------------------------------------------------

void FW::runOptimize(const String& inFile, const String& outFile, int numLevels, bool includeMesh)
{
    if (hasError())
        return;

    // Open files.

    printf("Optimizing '%s' to '%s'...\n", inFile.getPtr(), outFile.getPtr());
    OctreeFile src(inFile, File::Read);
    if (hasError())
        return;

    OctreeFile dst(outFile, File::Create);
    if (hasError())
        return;

    // Print original size.

    printf("Source file size: %.0f megs\n", (F32)src.getFileSize() / (F32)(1 << 20));

    // Optimize.

    dst.set(src, (numLevels) ? numLevels : OctreeFile::UnitScale, includeMesh, true);
    if (hasError())
        return;

    printf("Flushing...\n");
    dst.flush();
    if (hasError())
        return;

    // Print optimized size.

    printf("Destination file size: %.0f megs\n", (F32)dst.getFileSize() / (F32)(1 << 20));
}

//------------------------------------------------------------------------

void FW::runBenchmark(const String& inFile, int numLevels, const Vec2i& frameSize, int framesPerLaunch, int warmupLaunches, int measureFrames, const Array<String>& cameras)
{
    if (hasError())
        return;

    // Set parameters.

    Benchmark bench;
    bench.setFrameSize(frameSize);
    bench.setFramesPerLaunch(framesPerLaunch);
    bench.setWarmupLaunches(warmupLaunches);
    bench.setMeasureFrames(measureFrames);
    bench.loadOctree(inFile, (numLevels) ? numLevels : OctreeFile::UnitScale);
    bench.setCameras(cameras);

    // Benchmark.

    CudaRenderer::Params params;
    params.enableBeamOptimization = false;
    bench.measure("Single pass", params);
    params.enableBeamOptimization = true;
    bench.measure("Beam opt.", params);

    // Print results.

    bench.printResults("Raycast performance", "");
}

//------------------------------------------------------------------------

void FW::init(void)
{
    // Parse mode.

    bool modeInteractive = false;
    bool modeBuild       = false;
    bool modeInspect     = false;
    bool modeAmbient     = false;
    bool modeOptimize    = false;
    bool modeBenchmark   = false;
    bool showHelp        = false;

    if (argc < 2)
    {
        printf("Run \"octree --help\" for command-line options.\n\n");
        modeInteractive = true;
    }
    else
    {
        String mode = argv[1];
        if (mode == "interactive")      modeInteractive = true;
        else if (mode == "build")       modeBuild = true;
        else if (mode == "inspect")     modeInspect = true;
        else if (mode == "ambient")     modeAmbient = true;
        else if (mode == "optimize")    modeOptimize = true;
        else if (mode == "benchmark")   modeBenchmark = true;
        else                            showHelp = true;
    }

    // Parse options.

    String  logFile;
    Vec2i   frameSize       = Vec2i(1024, 768);
    String  stateFile;
    String  inFile;
    S32     maxThreads      = 4;
    String  outFile;
    S32     numLevels       = 0;
    bool    buildContours   = true;
    F32     colorError      = 16.0f;
    F32     normalError     = 0.01f;
    F32     contourError    = 15.0f;
    F32     aoRadius        = 0.05f;
    bool    flipNormals     = false;
    bool    includeMesh     = true;
    S32     framesPerLaunch = 10;
    S32     warmupLaunches  = 4;
    S32     measureFrames   = 2000;
    Array<String> cameras;

    for (int i = 2; i < argc; i++)
    {
        const char* ptr = argv[i];

        if ((parseLiteral(ptr, "--help") || parseLiteral(ptr, "-h")) && !*ptr)
        {
            showHelp = true;
        }
        else if (parseLiteral(ptr, "--log="))
        {
            if (!*ptr)
                setError("Invalid log file '%s'!", argv[i]);
            logFile = ptr;
        }
        else if ((modeInteractive || modeBenchmark) && parseLiteral(ptr, "--size="))
        {
            if (!parseInt(ptr, frameSize.x) || !parseLiteral(ptr, "x") || !parseInt(ptr, frameSize.y) || *ptr || min(frameSize) <= 0)
                setError("Invalid frame size '%s'!", argv[i]);
        }
        else if (modeInteractive && parseLiteral(ptr, "--state="))
        {
            if (!*ptr)
                setError("Invalid state file '%s'!", argv[i]);
            stateFile = ptr;
        }
        else if ((modeInteractive || modeBuild || modeInspect || modeAmbient || modeOptimize || modeBenchmark) && parseLiteral(ptr, "--in="))
        {
            if (!*ptr)
                setError("Invalid input file '%s'!", argv[i]);
            inFile = ptr;
        }
        else if ((modeInteractive || modeBuild) && parseLiteral(ptr, "--max-threads="))
        {
            if (!parseInt(ptr, maxThreads) || *ptr || maxThreads < 1)
                setError("Invalid number of builder threads '%s'!", argv[i]);
        }
        else if ((modeBuild || modeOptimize) && parseLiteral(ptr, "--out="))
        {
            if (!*ptr)
                setError("Invalid input file '%s'!", argv[i]);
            outFile = ptr;
        }
        else if ((modeBuild || modeOptimize || modeBenchmark) && parseLiteral(ptr, "--levels="))
        {
            if (!parseInt(ptr, numLevels) || *ptr || numLevels < 1 || numLevels > OctreeFile::UnitScale)
                setError("Invalid number of levels '%s'!", argv[i]);
        }
        else if (modeBuild && parseLiteral(ptr, "--contours="))
        {
            int value = 0;
            if (!parseInt(ptr, value) || *ptr || value < 0 || value > 1)
                setError("Invalid contour enable/disable '%s'!", argv[i]);
            buildContours = (value != 0);
        }
        else if (modeBuild && parseLiteral(ptr, "--color-error="))
        {
            if (!parseFloat(ptr, colorError) || *ptr || colorError < 0.0f || colorError > 256.0f)
                setError("Invalid color error '%s'!", argv[i]);
        }
        else if (modeBuild && parseLiteral(ptr, "--normal-error="))
        {
            if (!parseFloat(ptr, normalError) || *ptr || normalError < 0.0f || normalError > 2.0f)
                setError("Invalid normal error '%s'!", argv[i]);
        }
        else if (modeBuild && parseLiteral(ptr, "--contour-error="))
        {
            if (!parseFloat(ptr, contourError) || *ptr || contourError < 0.0f)
                setError("Invalid contour error '%s'!", argv[i]);
        }
        else if (modeAmbient && parseLiteral(ptr, "--ao-radius="))
        {
            if (!parseFloat(ptr, aoRadius) || *ptr || aoRadius < 0.0f)
                setError("Invalid AO radius '%s'!", argv[i]);
        }
        else if (modeAmbient && parseLiteral(ptr, "--flip-normals="))
        {
            int value = 0;
            if (!parseInt(ptr, value) || *ptr || value < 0 || value > 1)
                setError("Invalid normal flip enable/disable '%s'!", argv[i]);
            flipNormals = (value != 0);
        }
        else if (modeOptimize && parseLiteral(ptr, "--include-mesh="))
        {
            int value = 0;
            if (!parseInt(ptr, value) || *ptr || value < 0 || value > 1)
                setError("Invalid contour include/exclude '%s'!", argv[i]);
            includeMesh = (value != 0);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--frames-per-launch="))
        {
            if (!parseInt(ptr, framesPerLaunch) || *ptr || framesPerLaunch < 1)
                setError("Invalid number of frames per launch '%s'!", argv[i]);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--warmup-launches="))
        {
            if (!parseInt(ptr, framesPerLaunch) || *ptr || warmupLaunches < 0)
                setError("Invalid number of warmup launches '%s'!", argv[i]);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--measure-frames="))
        {
            if (!parseInt(ptr, measureFrames) || *ptr || measureFrames < 1)
                setError("Invalid number of frames to measure '%s'!", argv[i]);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--camera="))
        {
            if (!*ptr)
                setError("Invalid camera signature '%s'!", argv[i]);
            cameras.add(ptr);
        }
        else
        {
            setError("Invalid option '%s'!", argv[i]);
        }
    }

    // Show help.

    if (showHelp)
    {
        printf("%s\n", s_commandHelpText);
        exitCode = 1;
        clearError();
        return;
    }

    // Log file specified => start logging.

    if (logFile.getLength())
        pushLogFile(logFile);

    // Validate options.

    if ((modeBuild || modeInspect || modeAmbient || modeOptimize || modeBenchmark) && !inFile.getLength())
        setError("Input file (--in) not specified!");
    if ((modeBuild || modeOptimize) && !outFile.getLength())
        setError("Output file (--out) not specified!");
    if (modeBuild && !numLevels)
        setError("Number of levels (--levels) not specified!");
    if (modeBenchmark && !cameras.getSize())
        setError("No camera signatures specified!");

    // Run.

    if (modeInteractive)
        runInteractive(frameSize, stateFile, inFile, maxThreads);

    if (modeBuild)
        runBuild(inFile, outFile, numLevels, buildContours, colorError, normalError, contourError, maxThreads);

    if (modeInspect)
        runInspect(inFile);

    if (modeAmbient)
        runAmbient(inFile, aoRadius, flipNormals);

    if (modeOptimize)
        runOptimize(inFile, outFile, numLevels, includeMesh);

    if (modeBenchmark)
        runBenchmark(inFile, numLevels, frameSize, framesPerLaunch, warmupLaunches, measureFrames, cameras);

    // Handle errors.

    if (hasError())
    {
        printf("Error: %s\n", getError().getPtr());
        exitCode = 1;
        clearError();
        return;
    }

    // Command-line mode => print footer.

    if (!modeInteractive)
        printf("Done.\n\n");
}

//------------------------------------------------------------------------
