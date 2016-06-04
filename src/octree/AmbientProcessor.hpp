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
#include "base/Timer.hpp"
#include "io/OctreeFile.hpp"
#include "io/OctreeRuntime.hpp"
#include "gpu/CudaCompiler.hpp"
#include "gpu/CudaModule.hpp"
#include "cuda/Ambient.hpp"

namespace FW
{
//------------------------------------------------------------------------

class AmbientProcessor
{
public:
    enum
    {
            DefaultRaysPerNode  = 256,
            MinRaysPerBatch     = 512 << 10,
            MaxRaysPerBatch     = 2048 << 10,
            NumSlicesToPrefetch = 8
    };

public:
                        AmbientProcessor    (OctreeFile* file, int objectID);
                        ~AmbientProcessor   (void);

    void                setRayLength        (F32 length)    { m_rayLength = length; }
    void                setFlipNormals      (bool enable)   { m_flipNormals = enable; }

    void                run                 (void);

private:
    struct NodeInfo
    {
        Vec3i           pos;
        U32             validMask;
        bool            lastInStrip;
        bool            secondInPair;
    };

    struct SliceTask
    {
        OctreeSlice*    slice;
        Array<NodeInfo> nodes;
        S32*            attachData;
    };

    void                        processSlice        (OctreeSlice* slice, const Array<NodeInfo>& nodes);
    void                        initiateProcessing  (void);
    void                        finishProcessing    (void);

    S32&                        getWarpCounter      (void) { return *(S32*)m_module->getGlobal("g_warpCounter").getMutablePtr(); }
    AmbientInput&               getInput            (void) { return *(AmbientInput*)m_module->getGlobal("c_input").getMutablePtr(); }

private:
                                AmbientProcessor    (AmbientProcessor&); // forbidden
    AmbientProcessor&           operator=           (AmbientProcessor&); // forbidden

private:
    CudaCompiler                m_compiler;
    CudaModule*                 m_module;

    OctreeFile*                 m_file;
    int                         m_objectID;
    int                         m_raysPerNode;
    float                       m_rayLength;
    bool                        m_flipNormals;
    OctreeRuntime*              m_runtime;
    int                         m_numWarps;
    Buffer                      m_activeWarps;
    Buffer                      m_requestBuffer;
    Buffer                      m_resultBuffer;
    double                      m_kernelTime;
    CUevent                     m_kernelStartEvent;
    CUevent                     m_kernelEndEvent;
    S64                         m_requestsProcessed;
    Array<SliceTask*>           m_sliceTasks;
    int                         m_sliceTaskTotal;
    Array<SliceTask*>           m_procTasks;
    int                         m_procTaskTotal;
};

//------------------------------------------------------------------------
}
