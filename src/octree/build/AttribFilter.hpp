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
#include "base/Math.hpp"
#include "3d/Mesh.hpp"
#include "TextureSampler.hpp"
#include "DisplacementMap.hpp"

namespace FW
{
//------------------------------------------------------------------------

class BuilderMeshAccessor;

//------------------------------------------------------------------------

class AttribFilter
{
public:
    enum Component // scalar
    {
        Component_Weight = 0,
        Component_ColorR,
        Component_ColorG,
        Component_ColorB,
        Component_ColorA,
        Component_NormalX,
        Component_NormalY,
        Component_NormalZ,

        Component_Max
    };

    enum DataItem // encoded dword
    {
        DataItem_Color = 0,
        DataItem_Normal,

        DataItem_Max
    };

    //------------------------------------------------------------------------

    class Value
    {
    public:
                            Value               (void)                  { clear(); }
                            Value               (const Value& v)        { operator=(v); }
        explicit            Value               (const S32* data)       { decode(data); }
                            ~Value              (void)                  {}

        F32                 get                 (int idx) const         { FW_ASSERT(idx >= 0 && idx < Component_Max); return m_components[idx]; }
        F32&                get                 (int idx)               { FW_ASSERT(idx >= 0 && idx < Component_Max); return m_components[idx]; }
        F32                 operator[]          (int idx) const         { return get(idx); }
        F32&                operator[]          (int idx)               { return get(idx); }

        F32                 getWeight           (void) const            { return get(Component_Weight); }
        void                setWeight           (F32 v)                 { get(Component_Weight) = v; }
        void                addWeight           (F32 v)                 { get(Component_Weight) += v; }

        Vec4f               getColor            (void) const            { return Vec4f(get(Component_ColorR), get(Component_ColorG), get(Component_ColorB), get(Component_ColorA)); }
        void                setColor            (const Vec4f& v)        { get(Component_ColorR) = v.x; get(Component_ColorG) = v.y; get(Component_ColorB) = v.z; get(Component_ColorA) = v.w; }
        void                addColor            (const Vec4f& v)        { get(Component_ColorR) += v.x; get(Component_ColorG) += v.y; get(Component_ColorB) += v.z; get(Component_ColorA) += v.w; }

        Vec3f               getNormal           (void) const            { return Vec3f(get(Component_NormalX), get(Component_NormalY), get(Component_NormalZ)); }
        void                setNormal           (const Vec3f& v)        { get(Component_NormalX) = v.x; get(Component_NormalY) = v.y; get(Component_NormalZ) = v.z; }
        void                addNormal           (const Vec3f& v)        { get(Component_NormalX) += v.x; get(Component_NormalY) += v.y; get(Component_NormalZ) += v.z; }

        void                clear               (void)                  { memset(m_components, 0, sizeof(m_components)); }
        Value&              operator=           (const Value& v)        { memcpy(m_components, v.m_components, sizeof(m_components)); return *this; }
        Value&              operator+=          (const Value& v)        { for (int i = 0; i < Component_Max; i++) m_components[i] += v.m_components[i]; return *this; }
        Value&              operator*=          (F32 v)                 { for (int i = 0; i < Component_Max; i++) m_components[i] *= v; return *this; }
        Value               operator+           (const Value& v) const  { return Value(*this) += v; }
        Value               operator*           (F32 v) const           { return Value(*this) *= v; }

        void                encode              (S32* dataOut, Vec4f& colorOut, Vec3f& normalOut) const;
        void                decode              (const S32* data);

    private:
        F32                 m_components[Component_Max];
    };

    //------------------------------------------------------------------------

public:
                            AttribFilter        (void);
    virtual                 ~AttribFilter       (void);

    virtual void            init                (const BuilderMeshAccessor* mesh, F32 voxelSize, int initialCapacity) = 0;
    virtual Vec2i           getExtent           (void) const = 0; // [min,max] of the voxel neighborhood

    // Input data one voxel at a time.

    virtual void            inputBegin          (int voxelIdx, const Vec3f& voxelPos) = 0;

    virtual void            inputTriangle       (int                                triIdx,
                                                 const TextureSampler::Sample&      colorSample,
                                                 const DisplacedTriangle::Normal*   normalSample,
                                                 F32                                dispArea,
                                                 int                                numBary,
                                                 const Vec2f*                       bary) = 0;

    virtual void            inputEnd            (void) = 0;

    // Compute the filtered value for a voxel.

    virtual void            outputBegin         (void) = 0;
    virtual void            outputAccumulate    (int voxelIdx, const Vec3i& posInNeighborhood) = 0; // must be within getExtent()
    virtual const Value&    outputEnd           (void) = 0;

private:
                            AttribFilter        (AttribFilter&); // forbidden
    AttribFilter&           operator=           (AttribFilter&); // forbidden
};

//------------------------------------------------------------------------

class BoxFilter : public AttribFilter
{
public:
                            BoxFilter           (const Vec2i& extent);
    virtual                 ~BoxFilter          (void);

    virtual void            init                (const BuilderMeshAccessor* mesh, F32 voxelSize, int initialCapacity);
    virtual Vec2i           getExtent           (void) const            { return m_extent; }

    virtual void            inputBegin          (int voxelIdx, const Vec3f& voxelPos);

    virtual void            inputTriangle       (int                                triIdx,
                                                 const TextureSampler::Sample&      colorSample,
                                                 const DisplacedTriangle::Normal*   normalSample,
                                                 F32                                dispArea,
                                                 int                                numBary,
                                                 const Vec2f*                       bary);

    virtual void            inputEnd            (void)                  {}

    virtual void            outputBegin         (void)                  { m_output.clear(); }
    virtual void            outputAccumulate    (int voxelIdx, const Vec3i& posInNeighborhood) { FW_UNREF(posInNeighborhood); m_output += m_voxels[voxelIdx]; }
    virtual const Value&    outputEnd           (void)                  { return m_output; }

private:
                            BoxFilter           (BoxFilter&); // forbidden
    BoxFilter&              operator=           (BoxFilter&); // forbidden

private:
    const BuilderMeshAccessor*  m_mesh;
    Vec2i                       m_extent;
    Array<Value>                m_voxels;
    Value*                      m_input;
    Value                       m_output;
};

//------------------------------------------------------------------------

class PyramidFilter : public AttribFilter
{
private:
    struct Voxel
    {
        F32                 v[8][Component_Max];
    };

public:
                            PyramidFilter       (void);
    virtual                 ~PyramidFilter      (void);

    virtual void            init                (const BuilderMeshAccessor* mesh, F32 voxelSize, int initialCapacity);
    virtual Vec2i           getExtent           (void) const            { return Vec2i(0, 1); }

    virtual void            inputBegin          (int voxelIdx, const Vec3f& voxelPos);

    virtual void            inputTriangle       (int                                triIdx,
                                                 const TextureSampler::Sample&      colorSample,
                                                 const DisplacedTriangle::Normal*   normalSample,
                                                 F32                                dispArea,
                                                 int                                numBary,
                                                 const Vec2f*                       bary);

    virtual void            inputEnd            (void)                  {}

    virtual void            outputBegin         (void)                  { m_output.clear(); }
    virtual void            outputAccumulate    (int voxelIdx, const Vec3i& posInNeighborhood);
    virtual const Value&    outputEnd           (void)                  { return m_output; }

private:
    void                    setTriangle         (F32 weight, const Vec3f& p, const Vec3f& pu, const Vec3f& pv);

private:
                            PyramidFilter       (PyramidFilter&); // forbidden
    PyramidFilter&          operator=           (PyramidFilter&); // forbidden

private:
    const BuilderMeshAccessor*  m_mesh;
    Array<Voxel>                m_voxels;
    F32                         m_voxelSize;
    F32                         m_weightCoef;
    Vec3f                       m_voxelPos;
    Voxel*                      m_input;
    Value                       m_output;

    F32                         m_w[5][5];
    F32                         m_zw[4][4];
    F32                         m_yw[3][3];
    F32                         m_yzw[3][3];
    Vec3f                       m_coefs[8];
};

//------------------------------------------------------------------------
}
