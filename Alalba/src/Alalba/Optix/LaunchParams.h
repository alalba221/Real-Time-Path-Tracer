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
namespace Alalba {
  //using namespace gdt;
  struct TriangleMeshSBTData {
    gdt::vec3f  color;
    gdt::vec3f* vertex;
    gdt::vec3i* index;
    //float3  color;
    //float3* vertex;
    //float3* index;
  };
  struct SphereSBTData {
    //gdt::vec3f  center;
    float3  center;
    float radius;
  };
  struct SphereShellMeshSBTData {
    //gdt::vec3f  center;
    float3  center;
    float radius1;
    float radius2;
  };
  struct ParalelogramSBTData {
    float4	plane;
    float3 	v1;
    float3 	v2;
    float3 	anchor;
  };

  struct HitGroupData
  {
    union
    {
      SphereSBTData sphere;
      SphereShellMeshSBTData          sphere_shell;
      ParalelogramSBTData        parallelogram;
      TriangleMeshSBTData  triangle_mesh;
    } geometry;

    union 
    {
      //Phong           metal;
      //Glass           glass;
      //CheckerPhong    checker;
      float3     color;
    } shading;
  };

  struct LaunchParams
  {
    struct {
      uint32_t* colorBuffer;
      gdt::vec2i size;
    } frame;

    struct {
      gdt::vec3f position;
      gdt::vec3f direction;
      gdt::vec3f horizontal;
      gdt::vec3f vertical;
    } camera;

    OptixTraversableHandle traversable;
  };

} // ::osc