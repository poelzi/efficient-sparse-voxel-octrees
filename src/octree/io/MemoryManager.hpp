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
#include "gpu/Buffer.hpp"

namespace FW
{
//------------------------------------------------------------------------

class MemoryManager
{
public:

    //------------------------------------------------------------------------

    enum
    {
        MaxCpuBytes     = 512 << 20,
        MaxTextureBytes = 512 << 20,
        CudaMemReserve  = 32 << 20,
        CudaMemMaxPct   = 90,
        CompactMinCopy  = 8 << 20,
        CompactMaxTemp  = 16 << 20
    };

    enum Mode
    {
        Mode_CPU,
        Mode_Cuda,
    };

    //------------------------------------------------------------------------

    class Relocation
    {
    public:
                            Relocation              (void)                      : m_initStep(1) {}
                            Relocation              (const Relocation& other)   { operator=(other); }
                            ~Relocation             (void)                      {}

        bool                isEmpty                 (void) const                { return (m_oldOfs.getSize() == 0); }
        S64                 getByteDelta            (S64 oldOfs) const;
        S64                 getDWordDelta           (S64 oldOfs) const          { return getByteDelta(oldOfs << 2) >> 2; }

        void                clear                   (void)                      { m_initStep = 1; m_oldOfs.clear(); m_delta.clear(); }
        void                add                     (S64 oldOfs, S64 delta);
        Relocation&         operator=               (const Relocation& other);

    private:
        S32                 m_initStep;
        Array<S64>          m_oldOfs;
        Array<S64>          m_delta;
    };

    //------------------------------------------------------------------------

private:
    struct FreeRange // tracks allocation status of pages
    {
        FreeRange*          prev;
        FreeRange*          next;

        S64                 startPage;              // inclusive
        S64                 endPage;                // exclusive
    };

public:
                            MemoryManager           (Mode mode, int align);
                            ~MemoryManager          (void);

    Buffer&                 getBuffer               (void)                      { return m_buffer; }
    Mode                    getMode                 (void) const                { return m_mode; }
    S64                     getTotalBytes           (void) const                { return m_numPages << m_pageBytesLog; }
    S64                     getFreeBytes            (void) const                { return m_numFreePages << m_pageBytesLog; }

    void                    clear                   (void);
    S64                     alloc                   (S64 numBytes, Relocation* reloc = NULL); // -1 if out of memory
    void                    free                    (S64 ofs, S64 numBytes);

private:
    FreeRange*              addFreeRange            (FreeRange* prev, S64 startPage, S64 endPage);
    void                    removeFreeRange         (FreeRange* range);
    void                    compact                 (FreeRange* startRange, FreeRange* endRange, Relocation* reloc);

    void                    allocMaximalCudaBuffer  (void);

private:
                            MemoryManager           (MemoryManager&); // forbidden
    MemoryManager&          operator=               (MemoryManager&); // forbidden

private:
    Mode                    m_mode;
    S32                     m_pageBytes;
    S32                     m_pageBytesLog;

    Buffer                  m_buffer;
    Array<CUdeviceptr>      m_cudaBlocks;
    S64                     m_numPages;
    S64                     m_numFreePages;
    FreeRange               m_freeRanges;

    Buffer                  m_compactTemp;
};

//------------------------------------------------------------------------
}
