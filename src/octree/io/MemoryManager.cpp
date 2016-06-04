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

#include "MemoryManager.hpp"
#include "base/BinaryHeap.hpp"
#include "gpu/CudaModule.hpp"

using namespace FW;

//------------------------------------------------------------------------

S64 MemoryManager::Relocation::getByteDelta(S64 oldOfs) const
{
    if (!m_oldOfs.getSize() || oldOfs < m_oldOfs[0])
        return 0;

    int idx = 0;
    for (int step = m_initStep; step > 0; step >>= 1)
    {
        int probe = idx + step;
        if (probe < m_oldOfs.getSize() && oldOfs >= m_oldOfs[probe])
            idx = probe;
    }
    return m_delta[idx];
}

//------------------------------------------------------------------------

void MemoryManager::Relocation::add(S64 oldOfs, S64 delta)
{
    FW_ASSERT(!m_oldOfs.getSize() || oldOfs > m_oldOfs.getLast());

    if ((m_initStep << 1) == m_oldOfs.getSize())
        m_initStep <<= 1;

    m_oldOfs.add(oldOfs);
    m_delta.add(delta);
}

//------------------------------------------------------------------------

MemoryManager::Relocation& MemoryManager::Relocation::operator=(const Relocation& other)
{
    m_initStep = other.m_initStep;
    m_oldOfs = other.m_oldOfs;
    m_delta = other.m_delta;
    return *this;
}

//------------------------------------------------------------------------

MemoryManager::MemoryManager(Mode mode, int align)
:   m_mode          (mode),
    m_buffer        (NULL, 0, Buffer::Hint_None, align)
{
    // Determine page size.

    FW_ASSERT(align > 0 && (align & (align - 1)) == 0);
    m_pageBytes = 1;
    m_pageBytesLog = 0;
    while (m_pageBytes != align)
    {
        m_pageBytes <<= 1;
        m_pageBytesLog++;
    }

    // Allocate buffer.

    switch (mode)
    {
    case Mode_CPU:
        m_numPages = MaxCpuBytes >> m_pageBytesLog;
        m_buffer.resize(m_numPages << m_pageBytesLog);
        break;

    case Mode_Cuda:
        allocMaximalCudaBuffer();
        break;

    default:
        FW_ASSERT(false);
        break;
    }

    printf("MemoryManager: Allocated %.0f megabytes.\n", (F64)(m_numPages << m_pageBytesLog) * exp2(-20));

    // Setup allocator.

    m_freeRanges.prev       = &m_freeRanges;
    m_freeRanges.next       = &m_freeRanges;
    m_freeRanges.startPage  = FW_S64_MAX;
    m_freeRanges.endPage    = FW_S64_MAX;

    clear();
}

//------------------------------------------------------------------------

MemoryManager::~MemoryManager(void)
{
    clear();
    removeFreeRange(m_freeRanges.next);
    for (int i = 0; i < m_cudaBlocks.getSize(); i++)
        CudaModule::checkError("cuMemFree", cuMemFree(m_cudaBlocks[i]));
}

//------------------------------------------------------------------------

void MemoryManager::clear(void)
{
    while (m_freeRanges.next != &m_freeRanges)
        removeFreeRange(m_freeRanges.next);
    addFreeRange(&m_freeRanges, 0, m_numPages);
    m_numFreePages = m_numPages;
}

//------------------------------------------------------------------------

S64 MemoryManager::alloc(S64 numBytes, Relocation* reloc)
{
    FW_ASSERT(numBytes >= 0);
    if (reloc)
        reloc->clear();

    // Out of memory => fail.

    S64 numPages = (max(numBytes - 1, (S64)0) >> m_pageBytesLog) + 1;
    if (numPages > m_numFreePages)
        return -1;

    // Of all memory ranges containing enough free pages,
    // find the one with the least amount of allocated pages.

    FreeRange*  currStart       = m_freeRanges.next;
    S64         currFreePages   = 0;
    S64         currAllocPages  = 0;

    FreeRange*  bestStart       = NULL;
    FreeRange*  bestEnd         = NULL;
    S64         bestAllocPages  = FW_S64_MAX;

    for (FreeRange* currEnd = m_freeRanges.next; currEnd != &m_freeRanges; currEnd = currEnd->next)
    {
        // Grow current range towards the end.

        currFreePages += currEnd->endPage - currEnd->startPage;
        if (currEnd->prev != &m_freeRanges)
            currAllocPages += currEnd->startPage - currEnd->prev->endPage;

        // Shrink current range from the start.

        for (;;)
        {
            S64 probe = currFreePages - (currStart->endPage - currStart->startPage);
            if (probe < numPages)
                break;
            currFreePages = probe;
            currAllocPages -= currStart->next->startPage - currStart->endPage;
            currStart = currStart->next;
        }

        // Best so far?

        if (currFreePages >= numPages && currAllocPages < bestAllocPages)
        {
            bestStart       = currStart;
            bestEnd         = currEnd;
            bestAllocPages  = currAllocPages;

            if (!currAllocPages)
                break;
        }
    }

    // The range contains allocated pages => fail or compact.

    FW_ASSERT(bestStart);
    if (bestStart != bestEnd)
    {
        if (!reloc)
            return -1;
        compact(bestStart, bestEnd, reloc);
    }

    // Allocate.

    S64 ofs = bestStart->startPage << m_pageBytesLog;
    bestStart->startPage += numPages;
    m_numFreePages -= numPages;
    if (bestStart->startPage == bestStart->endPage)
        removeFreeRange(bestStart);
    return ofs;
}

//------------------------------------------------------------------------

void MemoryManager::free(S64 ofs, S64 numBytes)
{
    FW_ASSERT(ofs >= 0);
    FW_ASSERT((ofs & (m_pageBytes - 1)) == 0);
    FW_ASSERT(numBytes >= 0);
    S64 startPage = ofs >> m_pageBytesLog;
    S64 endPage = startPage + (max(numBytes - 1, (S64)0) >> m_pageBytesLog) + 1;
    FW_ASSERT(startPage >= 0 && endPage <= m_numPages);

    // Find previous range.

    FreeRange* range = &m_freeRanges;
    while (range->next->startPage <= startPage)
        range = range->next;

    // Add range for the block.

    addFreeRange(range, startPage, endPage);
    m_numFreePages += endPage - startPage;

    // Union with the existing ranges.

    if (range == &m_freeRanges || range->endPage < startPage)
        range = range->next;
    while (range->next != &m_freeRanges && range->next->startPage <= range->endPage)
    {
        m_numFreePages -= range->endPage - range->next->startPage;
        range->endPage = max(range->endPage, range->next->endPage);
        removeFreeRange(range->next);
    }
}

//------------------------------------------------------------------------

MemoryManager::FreeRange* MemoryManager::addFreeRange(FreeRange* prev, S64 startPage, S64 endPage)
{
    FW_ASSERT(prev);
    FW_ASSERT(startPage >= 0 && startPage <= endPage && endPage <= m_numPages);

    FreeRange* range    = new FreeRange;
    range->prev         = prev;
    range->next         = prev->next;
    range->startPage    = startPage;
    range->endPage      = endPage;
    prev->next->prev    = range;
    prev->next          = range;
    return range;
}

//------------------------------------------------------------------------

void MemoryManager::removeFreeRange(FreeRange* range)
{
    FW_ASSERT(range);
    range->prev->next = range->next;
    range->next->prev = range->prev;
    delete range;
}

//------------------------------------------------------------------------

void MemoryManager::compact(FreeRange* startRange, FreeRange* endRange, Relocation* reloc)
{
    FW_ASSERT(reloc);
    FW_ASSERT(startRange && startRange != &m_freeRanges);
    FW_ASSERT(endRange && endRange != &m_freeRanges);
    FW_ASSERT(startRange != endRange);

    // Move all allocated pages to the beginning of the range.

    for (;;)
    {
        FreeRange* next = startRange->next;
        S64 src = startRange->endPage << m_pageBytesLog;
        S64 dst = startRange->startPage << m_pageBytesLog;
        S64 size = (next->startPage - startRange->endPage) << m_pageBytesLog;

        // Non-overlapping or delta is large enough => copy directly.

        if (src - dst >= min(size, (S64)CompactMinCopy))
        {
            S64 ofs = 0;
            while (ofs < size)
            {
                S64 num = min(size - ofs, src - dst);
                m_buffer.setRange(dst + ofs, m_buffer, src + ofs, num);
                ofs += num;
            }
        }

        // Otherwise => copy through temporary buffer.

        else
        {
            if (!m_compactTemp.getSize())
            {
                m_compactTemp.resizeDiscard(CompactMaxTemp);
                m_compactTemp.setOwner(m_buffer.getOwner(), true);
            }

            S64 ofs = 0;
            while (ofs < size)
            {
                S64 num = min(size - ofs, (S64)CompactMaxTemp);
                m_compactTemp.setRange(0, m_buffer, src + ofs, num);
                m_buffer.setRange(dst + ofs, m_compactTemp, 0, num);
                ofs += num;
            }
        }

        // Add relocation.

        S64 delta = startRange->startPage - startRange->endPage;
        reloc->add(startRange->endPage << m_pageBytesLog, delta << m_pageBytesLog);

        // Update ranges.

        startRange->startPage = next->startPage + delta;
        startRange->endPage = next->endPage;
        removeFreeRange(next);
        if (next == endRange)
            break;
    }

    reloc->add(startRange->endPage << m_pageBytesLog, 0);
}

//------------------------------------------------------------------------

void MemoryManager::allocMaximalCudaBuffer(void)
{
    struct Block
    {
        U64 ptr;
        U64 size;
    };

    U32 minBufSize = 1 << 20;

    // Try to allocate a single block.

    CudaModule::staticInit();
    Array<Block> blocks;
    {
        CUsize_t free = 0, total = 0;
        cuMemGetInfo(&free, &total);

        U64 size = (U64)max((S64)free - CudaMemReserve, (S64)0);
        size = min(size, (U64)free * CudaMemMaxPct / 100);

        CUdeviceptr ptr = NULL;
        cuMemAlloc(&ptr, (U32)size);
        if (ptr)
        {
            Block& block = blocks.add();
            block.ptr = (U64)ptr;
            block.size = size;
        }
    }

    // Failure => allocate multiple blocks.

    if (!blocks.getSize())
    {
        U64 totalSize = 0;
        for (;;)
        {
            CUsize_t free = 0, total = 0;
            cuMemGetInfo(&free, &total);

            U64 size = 1;
            while (size + CudaMemReserve <= free && (totalSize + size) * 100 <= (totalSize + free) * CudaMemMaxPct)
                size <<= 1;
            size >>= 1;

            CUdeviceptr ptr = NULL;
            while (size >= minBufSize)
            {
                cuMemAlloc(&ptr, (U32)size);
                if (ptr)
                    break;
                size >>= 1;
            }
            if (!ptr)
                break;

            Block& block = blocks.add();
            block.ptr = (U64)ptr;
            block.size = size;
            totalSize += size;
        }
    }

    // Find longest run of consecutive blocks.

    U64 runStart = 0;
    U64 runEnd = 0;
    {
        BinaryHeap<U64> heap;
        for (int i = 0; i < blocks.getSize(); i++)
            heap.add(i, blocks[i].ptr);

        U64 currStart = 0;
        U64 currEnd = 0;
        while (!heap.isEmpty())
        {
            const Block& block = blocks[heap.getMinIndex()];
            heap.removeMin();

            if (block.ptr != currEnd)
                currStart = block.ptr;
            currEnd = block.ptr + block.size;
//            printf("%08x  %08x  %08x\n", (U32)block.ptr, (U32)block.size, (U32)(currEnd - currStart));

            if (currEnd - currStart > runEnd - runStart)
            {
                runStart = currStart;
                runEnd = currEnd;
            }
        }
    }

    // Discard blocks outside the run.

    for (int i = 0 ; i < blocks.getSize(); i++)
    {
        U64 ptr = blocks[i].ptr;
        if (ptr >= runStart && ptr < runEnd)
            m_cudaBlocks.add((CUdeviceptr)ptr);
        else
            cuMemFree((CUdeviceptr)ptr);
    }

    // Setup the block.

    runStart = (runStart + m_pageBytes - 1) & (U64)-m_pageBytes;
    m_numPages = (runEnd - runStart) >> m_pageBytesLog;
    if (m_numPages <= 0)
        fail("MemoryManager: Cannot allocate CUDA memory!");

    m_buffer.wrapCuda((CUdeviceptr)runStart, m_numPages << m_pageBytesLog);
}

//------------------------------------------------------------------------
