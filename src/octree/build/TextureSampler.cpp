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

#include "TextureSampler.hpp"

using namespace FW;

//------------------------------------------------------------------------

TextureSampler::TextureSampler(const Texture& tex)
:   m_size  (0),
    m_data  (NULL)
{
    const Image* image = tex.getImage();
    FW_ASSERT(image);
    m_size = image->getSize();
    FW_ASSERT(m_size.min() > 0);

    // Layout levels.

    Vec2i size = m_size;
    int numBytes = 0;
    do
    {
        Level& level = m_levels.add();
        level.size = size;
        level.numBytes = size.x * size.y * ((m_levels.getSize() == 1) ? sizeof(Color) : sizeof(Range));
        numBytes += level.numBytes;
        size.x = (size.x + 1) >> 1;
        size.y = (size.y + 1) >> 1;
    }
    while (size.max() > 1);

    // Allocate data.
    {
        m_data = new U8[numBytes];
        U8* ptr = m_data;
        for (int i = 0; i < m_levels.getSize(); i++)
        {
            Level& level = m_levels[i];
            level.colors = (i == 0) ? (Color*)ptr : NULL;
            level.ranges = (i == 0) ? NULL : (Range*)ptr;
            ptr += level.numBytes;
        }
    }

    // Convert base level.

    image->read(ImageFormat::R8_G8_B8_A8, m_levels[0].colors, m_size.x * sizeof(Color));

    // Generate mipmaps.

    for (int i = 1; i < m_levels.getSize(); i++)
    {
        const Level& level = m_levels[i];
        const Level& prev = m_levels[i - 1];
        Range* dst = level.ranges;

        for (int y = 0; y < level.size.y * 2; y += 2)
        {
            const Color* srcColor = prev.colors + prev.size.x * y;
            const Range* srcRange = prev.ranges + prev.size.x * y;
            int yinc = (y == prev.size.y - 1) ? -prev.size.x * y : prev.size.x;
            Sample s00, s10, s01, s11;

            for (int x = 0; x < level.size.x * 2; x += 2)
            {
                int xinc = (x == prev.size.x - 1) ? -x : 1;
                if (prev.colors)
                {
                    decode(s00, srcColor[0]);
                    decode(s10, srcColor[xinc]);
                    decode(s01, srcColor[yinc]);
                    decode(s11, srcColor[xinc + yinc]);
                    srcColor += 2;
                }
                else
                {
                    decode(s00, srcRange[0]);
                    decode(s10, srcRange[xinc]);
                    decode(s01, srcRange[yinc]);
                    decode(s11, srcRange[xinc + yinc]);
                    srcRange += 2;
                }
                lerpMinMax(s00, s10, 0.5f);
                lerpMinMax(s01, s11, 0.5f);
                lerpMinMax(s00, s01, 0.5f);
                encode(*dst++, s00);
            }
        }
    }
}

//------------------------------------------------------------------------

TextureSampler::~TextureSampler(void)
{
    delete[] m_data;
}

//------------------------------------------------------------------------

void TextureSampler::samplePoint(Sample& res, const Vec2f& pos, F32 sizeInTexels) const
{
    // Wrap position and compute level of detail.

    Vec2f pp = Vec2f(pos.x - floor(pos.x), pos.y - floor(pos.y)) * Vec2f(m_size);
    F32 lod = log2(max(sizeInTexels, 1.0f));
    int li = (int)lod;
    F32 lf = clamp(lod - (F32)li, 0.0f, 1.0f);

    // Lookup the first level.

    Vec2f pf0;
    Sample& s000 = res;
    Sample s100, s010, s110;
    lookup(pf0, s000, s100, s010, s110, pp, li);

    // Magnification => bilinear filter on base level, with diminished lo/hi difference.

    if (sizeInTexels <= 1.0f)
    {
        lerpShrink(s000, s100, pf0.x, sizeInTexels);
        lerpShrink(s010, s110, pf0.x, sizeInTexels);
        lerpShrink(s000, s010, pf0.y, sizeInTexels);
    }

    // Minification => trilinear filter between two mipmap levels.

    else
    {
        Vec2f pf1;
        Sample s001, s101, s011, s111;
        lookup(pf1, s001, s101, s011, s111, pp, li + 1);

        F32 kernelSize = sizeInTexels * exp2(-li - 1);
        lerpMinMax(s000, s100, pf0.x);
        lerpMinMax(s010, s110, pf0.x);
        lerpMinMax(s000, s010, pf0.y);
        lerpShrink(s001, s101, pf1.x, kernelSize);
        lerpShrink(s011, s111, pf1.x, kernelSize);
        lerpShrink(s001, s011, pf1.y, kernelSize);
        lerpMinMax(s000, s001, lf);
    }

    // Normalize the result.

    res.avg *= 1.0f / 255.0f;
    res.lo *= 1.0f / 255.0f;
    res.hi *= 1.0f / 255.0f;
}

//------------------------------------------------------------------------

void TextureSampler::lerpMinMax(Sample& a, const Sample& b, F32 t)
{
    a.avg = a.avg + (b.avg - a.avg) * t;
    a.lo.x = fastMin(a.lo.x, b.lo.x);
    a.lo.y = fastMin(a.lo.y, b.lo.y);
    a.lo.z = fastMin(a.lo.z, b.lo.z);
    a.lo.w = fastMin(a.lo.w, b.lo.w);
    a.hi.x = fastMax(a.hi.x, b.hi.x);
    a.hi.y = fastMax(a.hi.y, b.hi.y);
    a.hi.z = fastMax(a.hi.z, b.hi.z);
    a.hi.w = fastMax(a.hi.w, b.hi.w);
}

//------------------------------------------------------------------------

void TextureSampler::lerpShrink(Sample& a, const Sample& b, F32 t, F32 kernelSize)
{
    F32 half = kernelSize * 0.5f;
    F32 bias = (fastMax(half - t, 0.0f) - fastMax(half + t - 1.0f, 0.0f)) * 0.5f;
    F32 tc = t + bias;
    F32 td = half - abs(bias);

    a.avg = a.avg + (b.avg - a.avg) * t;
    Vec4f dlo = b.lo - a.lo;
    a.lo = a.lo + dlo * tc - dlo.abs() * td;
    Vec4f dhi = b.hi - a.hi;
    a.hi = a.hi + dhi * tc + dhi.abs() * td;
}

//------------------------------------------------------------------------

void TextureSampler::lookup(Vec2f& posFrac, Sample& s00, Sample& s10, Sample& s01, Sample& s11, const Vec2f& pos, int lod) const
{
    const Level& level = m_levels[clamp(lod, 0, m_levels.getSize() - 1)];

    // Compute position.

    Vec2f pp = pos * exp2(-lod) - 0.5f;
    int sx = level.size.x - 1;
    int sy = level.size.y - 1;
    int tx = clamp((int)(pp.x + 1.0f) - 1, -1, sx);
    int ty = clamp((int)(pp.y + 1.0f) - 1, -1, sy);
    posFrac.x = clamp(pp.x - (F32)tx, 0.0f, 1.0f);
    posFrac.y = clamp(pp.y - (F32)ty, 0.0f, 1.0f);

    int x0 = (tx == -1) ? sx : tx;
    int x1 = (tx == sx) ? 0 : tx + 1;
    int y0 = ((ty == -1) ? sy : ty) * level.size.x;
    int y1 = ((ty == sy) ? 0 : ty + 1) * level.size.x;

    // Lookup four neighboring texels.

    if (level.colors)
    {
        decode(s00, level.colors[x0 + y0]);
        decode(s10, level.colors[x1 + y0]);
        decode(s01, level.colors[x0 + y1]);
        decode(s11, level.colors[x1 + y1]);
    }
    else
    {
        decode(s00, level.ranges[x0 + y0]);
        decode(s10, level.ranges[x1 + y0]);
        decode(s01, level.ranges[x0 + y1]);
        decode(s11, level.ranges[x1 + y1]);
    }
}

//------------------------------------------------------------------------
