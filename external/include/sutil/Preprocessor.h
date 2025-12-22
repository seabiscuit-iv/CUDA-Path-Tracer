/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SUTIL_HOSTDEVICE __host__ __device__
#    define SUTIL_INLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
#else
#    define SUTIL_HOSTDEVICE
#    define SUTIL_INLINE inline
#    define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#endif



