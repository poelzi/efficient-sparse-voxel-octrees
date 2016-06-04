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
#include "gui/Window.hpp"
#include "gui/CommonControls.hpp"
#include "3d/CameraControls.hpp"
#include "OctreeManager.hpp"

namespace FW
{
//------------------------------------------------------------------------

class App : public Window::Listener, public CommonControls::StateObject
{
private:
    enum Action
    {
        Action_None,

        Action_About,
        Action_LoadState,
        Action_SaveState,

        Action_ResetCamera,
        Action_ImportCameraSignature,
        Action_ExportCameraSignature,

        Action_LoadOctree,
        Action_SaveOctree,
        Action_NewOctreeFromMesh,
        Action_RebuildOctree,
        Action_PrintStats,
    };

    enum View
    {
        View_Primary = 0,
        View_PrimaryAndShadow,
        View_OriginalMesh,
        View_OctreeDepth,
        View_IterationCount,
    };

public:
                                App                 (void);
    virtual                     ~App                (void);

    virtual bool                handleEvent         (const Window::Event& ev);
    virtual void                readState           (StateDump& d);
    virtual void                writeState          (StateDump& d) const;

    void                        setWindowSize       (const Vec2i& size)         { m_window.setSize(size); }
    void                        setMaxConcurrency   (int maxBuilderThreads)     { m_manager.setMaxConcurrency(maxBuilderThreads); }

    bool                        loadState           (const String& fileName)    { return m_commonCtrl.loadState(fileName); }
    void                        loadDefaultState    (void)                      { if (!m_commonCtrl.loadState(m_commonCtrl.getStateFileName(1))) firstTimeInit(); }
    bool                        loadOctree          (const String& fileName);
    void                        resetCamera         (void);
    void                        flashButtonTitles   (void)                      { m_commonCtrl.flashButtonTitles(); }

private:
    void                        rebuildGui          (void);
    void                        waitKey             (void);
    void                        render              (GLContext* gl);
    void                        renderGuiHelp       (GLContext* gl);
    BuilderBase::Params         getBuilderParams    (void);

    void                        firstTimeInit       (void);

private:
                                App                 (const App&); // forbidden
    App&                        operator=           (const App&); // forbidden

private:
    Window                      m_window;
    CommonControls              m_commonCtrl;
    CameraControls              m_cameraCtrl;
    OctreeManager               m_manager;

    Action                      m_action;
    String                      m_octreeFileName;

    View                        m_activeView;
    bool                        m_disableContourTest;
    bool                        m_disableBeamOptimization;
    bool                        m_disablePostProcessFiltering;
    bool                        m_enableAntialias;
    bool                        m_enableLargeAAFilter;
    F32                         m_maxVoxelSize;
    F32                         m_brightness;

    S32                         m_maxOctreeLevels;
    bool                        m_buildWithoutContours;
    F32                         m_buildColorError;
    F32                         m_buildNormalError;
    F32                         m_buildContourError;

    bool                        m_showHelp;
    bool                        m_showViewControls;
    bool                        m_showCameraControls;
    bool                        m_showManagementControls;
    bool                        m_viewControlsVisible;
    bool                        m_cameraControlsVisible;
    bool                        m_managementControlsVisible;
};

//------------------------------------------------------------------------

void    runInteractive  (const Vec2i& frameSize, const String& stateFile, const String& inFile, int maxThreads);
void    runBuild        (const String& inFile, const String& outFile, int numLevels, bool buildContours, F32 colorError, F32 normalError, F32 contourError, int maxThreads);
void    runInspect      (const String& inFile);
void    runAmbient      (const String& inFile, F32 aoRadius, bool flipNormals);
void    runOptimize     (const String& inFile, const String& outFile, int numLevels, bool includeMesh);
void    runBenchmark    (const String& inFile, int numLevels, const Vec2i& frameSize, int framesPerLaunch, int warmupLaunches, int measureFrames, const Array<String>& cameras);

//------------------------------------------------------------------------
}
