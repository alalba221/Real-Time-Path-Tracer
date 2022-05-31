// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once
#include "math/vec.h"
#include <vector_types.h>
#include"random/random.h"
////
//#include <sal.h>
//#include <thrust/device_vector.h>
//#include <device_launch_parameters.h>
//#include <curand_kernel.h>
namespace Alalba {

  struct TriangleMeshSBTData {
    gdt::vec3f* vertex;
    gdt::vec3i* index;
    gdt::vec3f  albedo;
    gdt::vec3f emission;
    gdt::vec3f kd;

    float eta;
    float roughness;
    float metallic;
    int pdf_id, sample_id, eval_id;
  };

  struct MissData {
    gdt::vec3f bg_color;
  };

  enum {
    LAMBERTIAN_SAMPLE = 0,
    LAMBERTIAN_PDF,
    LAMBERTIAN_EVAL,
    MICROFACET_SAMPLE,
    MICROFACET_PDF,
    MICROFACET_EVAL,
    BSDF_SAMPLE,
    BSDF_PDF,
    BSDF_EVAL,
    CALLABLE_PGS,
  }; // callable id

  enum
  {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
  };

  struct ParallelogramLight {
    gdt::vec3f emission;
    gdt::vec3f corner;
    gdt::vec3f v1;
    gdt::vec3f v2;
    gdt::vec3f normal;
  };
  struct LaunchParams
  {
    struct {
      float4* colorBuffer;
      // uint32_t* accum_buffer;
      // uint32_t* colorBuffer;
      // uint32_t* accum_buffer;
      gdt::vec2i size;
      int       frameID = 0;
    } frame;

    struct {
      gdt::vec3f position;
      gdt::vec3f direction;
      gdt::vec3f horizontal;
      gdt::vec3f vertical;
    } camera;
    
    ParallelogramLight light;
    int light_samples;
    unsigned int subframe_index;
    unsigned int samples_per_launch;
    OptixTraversableHandle traversable;
  };

} 