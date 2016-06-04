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
#include "../io/OctreeFile.hpp"
#include "../Util.hpp"
#include "base/Thread.hpp"

namespace FW
{
//------------------------------------------------------------------------

class BuilderBase
{
public:
    enum
    {
        AvgNodesPerSlice        = 48 << 10,
        MaxNodesPerBlock        = (1 << 24) / 17, // dictated by ColorNormalPaletteAttach
        ForceSplitLevels        = 2,

        // buildObject()
        MaxPrefetchSlices       = OctreeFile::MaxPrefetchSlices,
        MaxPrefetchBytesTotal   = OctreeFile::MaxPrefetchBytesTotal,
        MaxAsyncBuildSlices     = 8
    };

    enum FilterType
    {
        Filter_Nearest = 0, // Uncompressed attributes.
        Filter_NearestDXT,  // Compressed attributes.
        Filter_Box,         // Interpolated attributes, constructed using a box filter.
        Filter_Pyramid,     // Interpolated attributes, constructed using a pyramid filter.

        Filter_Max
    };

    enum ShaperType
    {
        Shaper_None = 0,    // Cubical voxels.
        Shaper_Hull = 3,    // Contours.

        Shaper_Max
    };

    struct Params
    {
        bool                enableVariableResolution;
        F32                 colorDeviation;             // 0.0f - 1.0f
        F32                 normalDeviation;            // 0.0f - 2.0f
        F32                 contourDeviation;           // relative to exp2(OctreeFile::UnitScale)
        FilterType          filter;
        ShaperType          shaper;

        Params(void)
        {
            enableVariableResolution    = true;
            colorDeviation              = 2.0f / 256.0f;
            normalDeviation             = 0.01f;
            contourDeviation            = exp2(OctreeFile::UnitScale - 15) * sqrt(3.0f);
            filter                      = Filter_NearestDXT;
            shaper                      = Shaper_Hull;
        }

        void setContourDeviationForLevels(F32 levels)
        {
            contourDeviation = pow(2.0f, (F32)OctreeFile::UnitScale - levels) * sqrt(3.0f);
        }
    };

protected:
    struct ChildSlice
    {
        S32                 idx;
        Vec3i               cubePos;
        S32                 cubeScale;
        S32                 nodeScale;
        bool                isSplit;

        Array<Vec3i>        nodes;
        Array<S32>          buildData;
        BitWriter           bitWriter;

        void                init                (int idx, const Vec3i& cubePos, int cubeScale, int nodeScale, bool isSplit);
        void                writeBits           (int num, S32 value)    { bitWriter.write(num, value); }
    };

    struct Task
    {
        OctreeSlice*        slice;
        String              idString;
        const S32*          buildData;
        S32                 objectID;
        S32                 numNodes;
        Array<AttachIO::AttachType> attachTypes;

        Array<OctreeSlice>  children;
        F32                 workIn;
        F32                 workOut;
    };

    //------------------------------------------------------------------------

    class ThreadState
    {
    public:
                            ThreadState         (void);
        virtual             ~ThreadState        (void);

        void                runTask             (Task& task);

    protected:
        virtual void        beginParentSlice    (const Vec3i& cubePos, int cubeScale, int nodeScale, int objectID, int numNodes) = 0;
        virtual void        endParentSlice      (void)                  {}
        virtual Vec3i       readParentNode      (const Vec3i& cubePos, int cubeScale, int nodeScale) = 0; // negative if not split

        virtual void        beginChildSlice     (ChildSlice& cs)        { FW_UNREF(cs); }
        virtual void        endChildSlice       (ChildSlice& cs)        { FW_UNREF(cs); }
        virtual bool        buildChildNode      (ChildSlice& cs, const Vec3i& nodePos) = 0;

        BitReader&          getBitReader        (void)                  { return m_bitReader; }
        S32                 readBits            (int num)               { return m_bitReader.read(num); }
        void                attachNodeValue     (AttachIO::AttachType type, const S32* value) { m_attachIO->exportNodeValue(type, value); }
        void                attachNodeSubValue  (AttachIO::AttachType type, int idxInNode, const S32* value) { m_attachIO->exportNodeSubValue(type, idxInNode, value); }
        void                addWorkIn           (F32 work)              { m_workIn += work; }
        void                addWorkOut          (F32 work)              { m_workOut += work; }

    private:
        void                initChildSlices     (const OctreeSlice& slice);
        void                selectSliceSplits   (Array<S32>& childEntries, const ChildSlice& cs);

    private:
                            ThreadState         (ThreadState&); // forbidden
        ThreadState&        operator=           (ThreadState&); // forbidden

    private:
        Array<ChildSlice>   m_childSlices;
        AttachIO*           m_attachIO;

        BitReader           m_bitReader;
        F32                 m_workIn;
        F32                 m_workOut;

        Array<U32>          m_nodeSplit;
        Array<U8>           m_nodeValidMask;
    };

    //------------------------------------------------------------------------

private:
    struct ThreadEntry
    {
        BuilderBase*        builder;
        ThreadState*        state;
        Thread*             thread;
    };

public:
                            BuilderBase         (OctreeFile* file);
    virtual                 ~BuilderBase        (void);

    void                    setMaxConcurrency   (int maxThreads)        { FW_ASSERT(maxThreads > 0); m_maxThreads = maxThreads; }

    OctreeFile*             getFile             (void) const            { return m_file; }
    virtual String          getClassName        (void) const = 0;
    virtual String          getIDString         (void) const = 0;
    virtual bool            supportsConcurrency (void) const            { return false; }

    void                    buildObject         (int objectID, int numLevels, const Params& params, bool enablePrints = true);
    bool                    buildSlice          (OctreeSlice* slice, F32* workIn = NULL, F32* workOut = NULL);

    bool                    asyncBuildSlice     (OctreeSlice* slice);
    OctreeSlice*            asyncFinishSlice    (bool wait, int sliceID = -1, F32* workIn = NULL, F32* workOut = NULL);
    int                     asyncGetNumPending  (void) const            { return m_tasks.getSize(); }
    bool                    asyncIsPending      (int sliceID) const     { return m_tasks.contains(sliceID); }
    void                    asyncAbort          (void);

protected:
    virtual bool            createRootSlice     (ChildSlice&                    cs,
                                                 Array<AttachIO::AttachType>&   attach,
                                                 Mat4f&                         octreeToObject,
                                                 int                            objectID,
                                                 const Params&                  params) = 0;

    virtual ThreadState*    createThreadState   (int idx) = 0;
    virtual void            prepareTask         (Task& task)            { FW_UNREF(task); }

private:
    bool                    createRootSlice     (OctreeSlice& slice, int objectID, const Params& params);

    static void             threadFunc          (void* param);

private:
                            BuilderBase         (BuilderBase&); // forbidden
    BuilderBase&            operator=           (BuilderBase&); // forbidden

private:
    OctreeFile*             m_file;
    Monitor                 m_monitor;
    S32                     m_maxThreads;

    Array<ThreadEntry>      m_threads;
    ThreadState*            m_serialState;
    volatile S32            m_numRunning;
    volatile bool           m_abort;

    Hash<S32, Task*>        m_tasks;
    Array<Task*>            m_pendingTasks;
    Array<Task*>            m_finishedTasks;
};

//------------------------------------------------------------------------
/*

BuildDataAttach format v1
-------------------------

writeBits(32, version); // must be 1
writeBits(32, objectID);
writeBits(32, numNodes);

for (int i = 0; i < 8; i++)
    writeBits(8, subclassIDString[i]);

writeBits(padding, 0); // pad to dword boundary
<subclass specific data>
writeBits(padding, 0); // pad to dword boundary

*/
//------------------------------------------------------------------------
}
