#include <optix_device.h>
#include "Alalba/Optix/LaunchParams.h"

//#include <crt/host_defines.h>
/*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */


using namespace Alalba;
namespace Alalba {
  /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

  static __forceinline__ __device__
    void* unpackPointer(uint32_t i0, uint32_t i1)
  {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
  }
  
  static __forceinline__ __device__
    void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T* getPRD()
  {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
  }

  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------

  extern "C" __global__ void __closesthit__radiance()
  { 
    const TriangleMeshSBTData& sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // compute normal: local!!!
    const int   primID = optixGetPrimitiveIndex();
    const gdt::vec3i index = sbtData.index[primID];
    const gdt::vec3f& A = sbtData.vertex[index.x];
    const gdt::vec3f& B = sbtData.vertex[index.y];
    const gdt::vec3f& C = sbtData.vertex[index.z];
    const gdt::vec3f Ng = normalize(cross(B - A, C - A));

    const gdt::vec3f rayDir = optixGetWorldRayDirection();
    const float cosDN = 0.2f + .8f * fabsf(dot(rayDir, Ng));
    gdt::vec3f& prd = *(gdt::vec3f*)getPRD<gdt::vec3f>();
    prd = cosDN *sbtData.color;

    //const HitGroupData& sbtData
    //  = *(const HitGroupData*)optixGetSbtDataPointer();

    //// compute normal: local!!!
    //const int   primID = optixGetPrimitiveIndex();
    //const gdt::vec3i index = sbtData.geometry.triangle_mesh.index[primID];
    //const gdt::vec3f& A = sbtData.geometry.triangle_mesh.vertex[index.x];
    //const gdt::vec3f& B = sbtData.geometry.triangle_mesh.vertex[index.y];
    //const gdt::vec3f& C = sbtData.geometry.triangle_mesh.vertex[index.z];
    //const gdt::vec3f Ng = normalize(cross(B - A, C - A));

    //const gdt::vec3f rayDir = optixGetWorldRayDirection();
    //const float cosDN = 0.2f + .8f * fabsf(dot(rayDir, Ng));
    //gdt::vec3f& prd = *(gdt::vec3f*)getPRD<gdt::vec3f>();
    //prd = cosDN * sbtData.shading.color;
  }

  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */
  }



  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------

  extern "C" __global__ void __miss__radiance()
  { 
    gdt::vec3f& prd = *(gdt::vec3f*)getPRD<gdt::vec3f>();
    // set to constant white as background color
    prd = gdt::vec3f(1.f);
  }



  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto& camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    gdt::vec3f pixelColorPRD = gdt::vec3f(0.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&pixelColorPRD, u0, u1);

    // normalized screen plane position, in [0,1]^2
    const gdt::vec2f screen(gdt::vec2f(ix + .5f, iy + .5f)
      / gdt::vec2f(optixLaunchParams.frame.size));

    // generate ray direction
    gdt::vec3f rayDir = normalize(camera.direction
      + (screen.x - 0.5f) * camera.horizontal
      + (screen.y - 0.5f) * camera.vertical);

    optixTrace(optixLaunchParams.traversable,
      camera.position,
      rayDir,
      0.f,    // tmin
      1e20f,  // tmax
      0.0f,   // rayTime
      OptixVisibilityMask(255),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
      SURFACE_RAY_TYPE,             // SBT offset
      RAY_TYPE_COUNT,               // SBT stride
      SURFACE_RAY_TYPE,             // missSBTIndex 
      u0, u1);

    const int r = int(255.99f * pixelColorPRD.x);
    const int g = int(255.99f * pixelColorPRD.y);
    const int b = int(255.99f * pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r << 0) | (g << 8) | (b << 16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }
}
