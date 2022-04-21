#include <optix_device.h>
#include "Alalba/Optix/LaunchParams.h"

/*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */


using namespace Alalba;
namespace Alalba {

  typedef gdt::LCG<16> Random;
  /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;
  //------------------------------------------------------------------------------
  //
  //
  //
  //------------------------------------------------------------------------------
  struct RadiancePRD {
    gdt::vec3f       emitted;
    gdt::vec3f       radiance;
    gdt::vec3f       attenuation;
    gdt::vec3f       origin;
    gdt::vec3f       direction;
    int          countEmitted;
    int          done;
    int          pad;
    Random random;
  };

  struct Onb
  {
    __forceinline__ __device__ Onb(const gdt::vec3f& normal)
    {
      m_normal = normal;

      if (fabs(m_normal.x) > fabs(m_normal.z))
      {
        m_binormal.x = -m_normal.y;
        m_binormal.y = m_normal.x;
        m_binormal.z = 0;
      }
      else
      {
        m_binormal.x = 0;
        m_binormal.y = -m_normal.z;
        m_binormal.z = m_normal.y;
      }

      m_binormal = normalize(m_binormal);
      m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(gdt::vec3f& p) const
    {
      p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    gdt::vec3f m_tangent;
    gdt::vec3f m_binormal;
    gdt::vec3f m_normal;
  };


  //------------------------------------------------------------------------------
  //
  //
  //
  //------------------------------------------------------------------------------
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


  static __forceinline__ __device__
    gdt::vec3f faceforward( const gdt::vec3f& n, const gdt::vec3f& i, const gdt::vec3f& nref)
  {
    //float sign = dot(i, nerf) > 0 ? 1.0f : -1.0f;
    //return  n*sign;
    return n * copysignf(1.0f, dot(i, nref));
  }

  static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    gdt::vec3f                 ray_origin,
    gdt::vec3f                 ray_direction,
    float                  tmin,
    float                  tmax,
    RadiancePRD* prd
  )
  {
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    optixTrace(
      handle,
      ray_origin,
      ray_direction,
      tmin,
      tmax,
      0.0f,                // rayTime
      OptixVisibilityMask(1),
      OPTIX_RAY_FLAG_NONE,
      RAY_TYPE_RADIANCE,        // SBT offset
      RAY_TYPE_COUNT,           // SBT stride
      RAY_TYPE_RADIANCE,        // missSBTIndex
      u0, u1);
  }

  static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    gdt::vec3f                 ray_origin,
    gdt::vec3f                 ray_direction,
    float                  tmin,
    float                  tmax
  )
  {
    //printf("A");
    unsigned int occluded = 0u;
    optixTrace(
      handle,
      ray_origin,
      ray_direction,
      tmin,
      tmax,
      0.0f,                    // rayTime
      OptixVisibilityMask(1),
      OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
      RAY_TYPE_OCCLUSION,      // SBT offset
      RAY_TYPE_COUNT,          // SBT stride
      RAY_TYPE_OCCLUSION,      // missSBTIndex
      occluded);
    return occluded;
  }

  static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, gdt::vec3f& p)
  {
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PI * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
  }

  static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
  {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
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
  extern "C" __global__ void __closesthit__occlusion()
  {
    setPayloadOcclusion(true);
  }
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
    const gdt::vec3f N_0 = normalize(cross(B - A, C - A));
    
    const gdt::vec3f rayDir = optixGetWorldRayDirection();
    
    const gdt::vec3f N = faceforward(N_0, -rayDir, N_0);
    const gdt::vec3f P = (gdt::vec3f)optixGetWorldRayOrigin() + (gdt::vec3f)optixGetRayTmax() * rayDir;

    // const float cosDN = 0.2f + .8f * (dot(-rayDir, N));
    RadiancePRD* prd = (RadiancePRD*)getPRD<RadiancePRD>();
    // prd->radiance = cosDN *sbtData.albedo;

    if (prd->countEmitted)
      prd->emitted = sbtData.emission;
    else
      prd->emitted = gdt::vec3f(0.f);

    // sample on hemisphere
    {
      const float z1 = prd->random();
      const float z2 = prd->random();
      
      gdt::vec3f w_in;
      cosine_sample_hemisphere(z1, z2, w_in);// local
      Onb onb(N);
      onb.inverse_transform(w_in);// local to global
      
      prd->direction = w_in;
      prd->origin = P;

      // prd->radiance = sbtData.albedo;
      // prd->countEmitted = false;
      prd->attenuation *= sbtData.albedo;
    }

    const float z1 = prd->random();
    const float z2 = prd->random();
    
    ParallelogramLight light = optixLaunchParams.light;
    const gdt::vec3f light_pos = light.corner + light.v1 * z1 + light.v2 * z2;
    const gdt::vec3f Li = light.emission;
    // Calculate properties of light sample (for area based pdf)
    const float L_distance = length(light_pos - P);
    const gdt::vec3f L_dir = normalize(light_pos - P);
    
    const float  LDotN = dot(N, L_dir);
    const float  LnDotL = -dot(light.normal, L_dir);
    
    gdt::vec3f L_light = gdt::vec3f(0.f);
    if (LDotN > 0 && LnDotL > 0) 
    {
      const bool occluded = traceOcclusion(
        optixLaunchParams.traversable,
        P,
        L_dir,
        0.01f,
        L_distance - 0.01f
      );
      if (!occluded)
      {
        float A = length(cross(light.v1, light.v2));
        L_light = (A * LDotN * LnDotL) / (L_distance * L_distance * M_PI);
      }
    }

    float weight = 0.0f;
    prd->radiance += L_light * Li;
  }

  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */
  }
  extern "C" __global__ void __anyhit__occlusion()
  { /*! for this simple example, this will remain empty */
  }


  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  extern "C" __global__ void __miss__occlusion()
  {
    /// Dont try to getPRD here!!!
    //RadiancePRD* prd = (RadiancePRD*)getPRD<RadiancePRD>();
    //prd->radiance = gdt::vec3f(0.f);
  }
  extern "C" __global__ void __miss__radiance()
  { 
    RadiancePRD* prd = (RadiancePRD*)getPRD<RadiancePRD>();
    // set to constant white as background color
    prd->radiance = gdt::vec3f(0.f);
    
    // prd->done = true;
  }



  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const int accumID = optixLaunchParams.frame.accumID;


    const auto& camera = optixLaunchParams.camera;
    const unsigned int subframe_index = optixLaunchParams.subframe_index;
    int spp = optixLaunchParams.samples_per_launch;
    
    // unsigned long long seed = blockIdx.x * blockDim.x + threadIdx.x;
    gdt::vec3f result(0.f);
    RadiancePRD prd;

    prd.random.init(ix + accumID * optixLaunchParams.frame.size.x,
      iy + accumID * optixLaunchParams.frame.size.y);
    do
    {
      const gdt::vec2f screen(gdt::vec2f(ix + prd.random(), iy + prd.random())
        / gdt::vec2f(optixLaunchParams.frame.size));

      gdt::vec3f  ray_origin = camera.position;

      gdt::vec3f ray_direction = normalize(camera.direction
        + (screen.x - 0.5f) * camera.horizontal
        + (screen.y - 0.5f) * camera.vertical);

      prd.emitted = gdt::vec3f(0.f);
      prd.radiance = gdt::vec3f(0.f);
      prd.attenuation = gdt::vec3f(1.f);
      prd.countEmitted = true;
      prd.done = false;

      int depth = 0;
      for (;;)
      {
        traceRadiance(
          optixLaunchParams.traversable,
          ray_origin,
          ray_direction,
          0.01f,  // tmin       // TODO: smarter offset
          1e16f,  // tmax
          &prd);

        //result += prd.emitted;
        result += prd.radiance * (prd.attenuation);
        
        if (prd.done || depth >= 3)
          break;
        
        ray_origin = prd.origin;
        ray_direction = prd.direction;
        
        depth++;
      }

    } while (--spp);

    const gdt ::vec3i    launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * optixLaunchParams.frame.size.x + launch_index.x;
    gdt::vec3f  accum_color = result / static_cast<float>(optixLaunchParams.samples_per_launch);
    
    if (subframe_index > 0)
    {
      const float                 a = 1.0f / static_cast<float>(subframe_index + 1);
      const gdt::vec3f accum_color_prev = optixLaunchParams.frame.colorBuffer[image_index];
      //accum_color = lerp(accum_color_prev, accum_color, a);
    }
    

    const int r = int(255.99f * accum_color.x);
    const int g = int(255.99f * accum_color.y);
    const int b = int(255.99f * accum_color.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r << 0) | (g << 8) | (b << 16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }
}
