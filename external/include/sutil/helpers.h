/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2025  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <vector_functions.h>
#include <vector_types.h>
#include <sutil/vec_math.h>


#define float3_as_args( u )                      \
    reinterpret_cast<unsigned int&>( ( u ).x ),  \
    reinterpret_cast<unsigned int&>( ( u ).y ),  \
    reinterpret_cast<unsigned int&>( ( u ).z )


namespace sutil
{

__forceinline__ __device__ float3 toSRGB( const float3& c )
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}


__forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x )
{
    x = clamp( x, 0.0f, 1.0f );
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}


__forceinline__ __device__ uchar4 make_color( const float3& c )
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
}

__forceinline__ __device__ uchar4 make_color( const float4& c )
{
    return make_color( make_float3( c.x, c.y, c.z ) );
}


__forceinline__ __device__ float luminance( const float3& rgb )
{
    const float3 ntsc_luminance = { 0.30f, 0.59f, 0.11f };
    return dot( rgb, ntsc_luminance );
}


__forceinline__ __device__ float fresnel_schlick( const float cos_theta,
                                                  const float exponent = 5.0f,
                                                  const float minimum  = 0.0f,
                                                  const float maximum  = 1.0f )
{
    /**
      Clamp the result of the arithmetic due to floating point precision:
      the result should lie strictly within [minimum, maximum]
      return clamp(minimum + (maximum - minimum) * powf(1.0f - cos_theta, exponent),
                   minimum, maximum);
    */

    /** The max doesn't seem like it should be necessary, but without it you get
        annoying broken pixels at the center of reflective spheres where cos_theta ~ 1.
    */
    return clamp( minimum + ( maximum - minimum ) * powf( fmaxf( 0.0f, 1.0f - cos_theta ), exponent ), minimum, maximum );
}


__forceinline__ __device__ float3 fresnel_schlick( const float cos_theta, const float exponent, const float3& minimum, const float3& maximum )
{
    return make_float3( fresnel_schlick( cos_theta, exponent, minimum.x, maximum.x ),
                        fresnel_schlick( cos_theta, exponent, minimum.y, maximum.y ),
                        fresnel_schlick( cos_theta, exponent, minimum.z, maximum.z ) );
}


__forceinline__ __device__ float3 linearize( float3 c )
{
    return make_float3(
            powf( c.x, 2.2f ),
            powf( c.y, 2.2f ),
            powf( c.z, 2.2f )
            );
}


__forceinline__ __device__ bool refract( float3& r, const float3& i, const float3& n, const float ior )
{
    float3 nn       = n;
    float  negNdotV = dot( i, nn );
    float  eta;

    if( negNdotV > 0.0f )
    {
        eta      = ior;
        nn       = -n;
        negNdotV = -negNdotV;
    }
    else
    {
        eta = 1.f / ior;
    }

    const float k = 1.f - eta * eta * ( 1.f - negNdotV * negNdotV );

    if( k < 0.0f )
    {
        // Initialize this value, so that r always leaves this function initialized.
        r = make_float3( 0.f );
        return false;
    }
    else
    {
        r = normalize( eta * i - ( eta * negNdotV + sqrtf( k ) ) * nn );
        return true;
    }
}


__forceinline__ __device__ float3 exp( const float3& x )
{
    return make_float3( expf( x.x ), expf( x.y ), expf( x.z ) );
}

} // namespace sutil
