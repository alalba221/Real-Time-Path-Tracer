#include <optix_device.h>
#include "Alalba/Optix/LaunchParams.h"

/*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */


using namespace Alalba;
namespace Alalba {

  typedef gdt::LCG<32> Random;
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

 /* struct Onb
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
  };*/


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

  __forceinline__ __device__ gdt::vec3f lerp(
    const gdt::vec3f& a, const gdt::vec3f& b, const float t
  )
  {
    return (1-t)*a + t * b;
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
    const gdt::vec3f Ng = normalize(cross(B - A, C - A));
    const gdt::vec3f ray_dir = normalize((gdt::vec3f)optixGetWorldRayDirection());
    const gdt::vec3f wo = -ray_dir;
    
    const gdt::vec3f ray_origin = optixGetWorldRayOrigin();
    const gdt::vec3f P = ray_origin + optixGetRayTmax() * ray_dir;
    const gdt::vec3f N = dot(Ng, wo) < 0 ? -Ng : Ng; // faceforward(Ng, -ray_dir, Ng);
    
    // const float cosDN = 0.2f + .8f * (dot(-rayDir, N));
    RadiancePRD* prd = (RadiancePRD*)getPRD<RadiancePRD>();

    if (prd->countEmitted)
    {
      prd->emitted = sbtData.emission;
      //printf("sbtData.emission: %d", sbtData.emission.x);
    }
    else 
    {
      prd->emitted = gdt::vec3f(0.f);
    }
    prd->countEmitted = false;
    // sample wi
    auto wi = optixDirectCall<gdt::vec3f, RadiancePRD*, const gdt::vec3f&, const TriangleMeshSBTData&, const gdt::vec3f&>(
      sbtData.sample_id,
      prd,
      N,
      sbtData,
      wo
      );

    prd->origin = P;
    prd->direction = normalize(wi);

    // sample light
    const float e1 = prd->random();
    const float e2 = prd->random();
    
    ParallelogramLight light = optixLaunchParams.light;
    const gdt::vec3f light_pos = light.corner + light.v1 * e1 + light.v2 * e2;
    const gdt::vec3f Li = light.emission;
    // Calculate properties of light sample (for area based pdf)
    const float L_distance = length(light_pos - P);
    const gdt::vec3f L_dir = normalize(light_pos - P);
    
    const float  LDotN = dot(N, L_dir);
    const float  LnDotL = dot(light.normal, -L_dir);
    gdt::vec3f L_light = gdt::vec3f(0.f);

    auto pdf = optixDirectCall<float, RadiancePRD*, const gdt::vec3f&, const gdt::vec3f&, const TriangleMeshSBTData&, const gdt::vec3f&>(
      sbtData.pdf_id,
      prd,
      wi,
      N,
      sbtData,
      wo
      );
    auto eval = optixDirectCall<gdt::vec3f, RadiancePRD*, const gdt::vec3f&, const gdt::vec3f&, const TriangleMeshSBTData&, const gdt::vec3f&>(
      sbtData.eval_id,
      prd,
      wi,
      N,
      sbtData,
      wo
      );

    /// shade(q, -wi) * {f_r * cosine / pdf(wi)}
    /// cos(N,wo) 
    const float costheta = dot(N, wi);
    prd->attenuation *= (eval * costheta / pdf);


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
        float pdf_light = 1.f / A;
        // -----------------------------------------------
        // if hit the light
        //  {L_i * f_r * cosine} / pdf(wi)
        // L_direct = L_i * {f_r * cos θ * cos θ’ / |x’ - p|^2 / pdf_light}
        // LnDotL : cos(theta')
        // LDotN : cos(theta)
        // ----------------------------------------------------
        L_light = (LDotN * LnDotL * eval) / (L_distance * L_distance) / pdf_light;
      }
    }
    /// ----------------------------
    /// + light emmision
    /// -----------------------------
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
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD* prd = (RadiancePRD*)getPRD<RadiancePRD>();
   
    prd->radiance = data->bg_color;
    prd->done = true;
    
  }



  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const int accumID = optixLaunchParams.frame.frameID;


    const auto& camera = optixLaunchParams.camera;
    const unsigned int subframe_index = optixLaunchParams.subframe_index;
    int spp = optixLaunchParams.samples_per_launch;
    
    // unsigned long long seed = blockIdx.x * blockDim.x + threadIdx.x;
    gdt::vec3f result(0.f);

    RadiancePRD prd;
    prd.random.init(ix + accumID * optixLaunchParams.frame.size.x,
      iy + accumID * optixLaunchParams.frame.size.y);

    gdt::vec3f shade(0.f);
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

        /// L_dir = L_i * f_r * cos θ * cos θ’ / |x’ - p|^2 / pdf_light
        /// L_indir = shade(q, -wi) *{ f_r * cos θ / pdf_hemi / P_RR}
        /// Return L_dir + L_indir
        /// radiance = emmit + shade
        shade += prd.emitted;
        shade += prd.radiance * (prd.attenuation);
        
        if (prd.done || depth >= 3)
          break;
        
        ray_origin = prd.origin;
        ray_direction = prd.direction;
        depth++;
      }

    } while (--spp);

  
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    gdt::vec3f  accum_color = shade / static_cast<float>(optixLaunchParams.samples_per_launch);
    gdt::vec4f rgba(accum_color, 1.f);
    // and write to frame buffer ...
    if (optixLaunchParams.frame.frameID > 0) {
      rgba
        += float(optixLaunchParams.frame.frameID)
        * gdt::vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
      rgba /= (optixLaunchParams.frame.frameID + 1.f);
    }
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }

  /// Helper
  __forceinline__ __device__ gdt::vec3f toWorld(
    const gdt::vec3f& N, const gdt::vec3f& ray
  )
  {
    gdt::vec3f m_tangent;
    gdt::vec3f m_binormal;
    gdt::vec3f m_normal = N;
    if (fabs(N.x) > fabs(N.z)) {
      m_binormal.x = -m_normal.y;
      m_binormal.y = m_normal.x;
      m_binormal.z = 0;
    }
    else {
      m_binormal.x = 0;
      m_binormal.y = -m_normal.z;
      m_binormal.z = m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross(m_binormal, m_normal);
    return ray.x * m_tangent + ray.y * m_binormal + ray.z * m_normal;
  }

  __forceinline__ __device__ gdt::vec3f reflect(
    const gdt::vec3f& I, const gdt::vec3f& N
  )
  {
    return (I - 2 * dot(I, N) * N);
  }

  __forceinline__ __device__ gdt::vec3f sampleHemiSphere(
    const gdt::vec3f& wo, const gdt::vec3f& N, RadiancePRD* prd
  )
  {
    
    //float x1 = getRandomFloat(state), x2 = getRandomFloat(state);
    float x1 = prd->random(), x2 = prd->random();
    float phi = 2.f * M_PI * x2;
    float r = sqrtf(x1);
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    gdt::vec3f wi(x, y, sqrt(fmaxf(0.f, 1.f - x * x - y * y)));
    return toWorld(N, wi);
  }

  __forceinline__ __device__ float DistributionGGX(
    const gdt::vec3f& N, const gdt::vec3f& H, float roughness
)
  {
    float a2 = roughness * roughness;
    float NdotH = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    denom = denom * denom * M_PI;
    return nom / denom;
  }
  __forceinline__ __device__ float GeometrySchlickGGX(
    const float NdotV, const float k
  )
  {
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
  }

  __forceinline__ __device__ float GeometrySmith(
    const gdt::vec3f& N, const gdt::vec3f& V, const gdt::vec3f& L, float k
  )
  {
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx1 = GeometrySchlickGGX(NdotL, k);
    float ggx2 = GeometrySchlickGGX(NdotV, k);
    return ggx1 * ggx2;
  }

  __forceinline__ __device__ gdt::vec3f fresnelSchlick(
    const gdt::vec3f& wi, const gdt::vec3f& h, const gdt::vec3f& albedo, float metallic
  )
  {
    // vec3f F0 = mix(vec3f(0.04f), albedo, metallic);
    gdt::vec3f F0 = gdt::vec3f(0.04f) * (1 - metallic) + albedo * metallic;
    float HoWi = dot(h, wi);
    return F0 + (gdt::vec3f(1.0f) - F0) * pow(2.0f, (-5.55473f * HoWi - 6.98316f) * HoWi);
  }

  /// --------------------------------
  /// callable
  extern "C" __device__ gdt::vec3f __direct_callable__lambertian_sample(
    RadiancePRD * prd,
    const gdt::vec3f & surface_noraml,
    const TriangleMeshSBTData & sbt,
    const gdt::vec3f & wo
  )
  {
    auto scattered = sampleHemiSphere(wo, surface_noraml, prd);
    return scattered;//toWorld(surface_noraml, scattered);
  }
  // pdf
  extern "C" __device__ float __direct_callable__lambertian_pdf(
    RadiancePRD * prd,
    const gdt::vec3f & wi,
    const gdt::vec3f & surface_noraml,
    const TriangleMeshSBTData & sbt,
    const gdt::vec3f & wo
  )
  {
    return 0.5f / M_PI;
  }
  // brdf
  extern "C" __device__ gdt::vec3f __direct_callable__lambertian_eval(
    RadiancePRD * prd,
    const gdt::vec3f & wi,
    const gdt::vec3f & surface_noraml,
    const TriangleMeshSBTData & sbt,
    const gdt::vec3f & wo
  )
  {
    //return sbt.albedo;
    return sbt.albedo / M_PI;
  }

  extern "C" __device__ gdt::vec3f __direct_callable__microfacet_sample(
    RadiancePRD * prd,
    const gdt::vec3f & surface_noraml,
    const TriangleMeshSBTData & sbt,
    const gdt::vec3f & wo
  )
  {
    //curandState_t* state = prd->state;
    float a = sbt.roughness * sbt.roughness;
    float a2 = a * a;
 
    float e0 = prd->random();
    float e1 = prd->random();
    float cosTheta2 = (1.0f - e0) / (e0 * (a2 - 1.0f) + 1.0f);
    float cosTheta = sqrt(cosTheta2);
    float sinTheta = sqrt(1 - cosTheta2);

    float phi = 2.0f * M_PI * e1;
    gdt::vec3f h(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    h = normalize(h);
    h = toWorld(surface_noraml, h);

    return reflect(-wo, h);
  }

  extern "C" __device__ gdt::vec3f __direct_callable__microfacet_pdf(
    RadiancePRD * prd,
    const gdt::vec3f & wi,
    const gdt::vec3f & surface_noraml,
    const TriangleMeshSBTData & sbt,
    const gdt::vec3f & wo
  )
  {
    gdt::vec3f N = surface_noraml;
    gdt::vec3f wh = normalize(wo + wi);
    float D = DistributionGGX(N, wh, sbt.roughness);
    return (D * dot(wh, N)) / (4.0f * dot(wo, wh));
    // return D / 4.0f;
  }

  extern "C" __device__ gdt::vec3f __direct_callable__microfacet_eval(
    RadiancePRD * prd,
    const gdt::vec3f & wi,
    const gdt::vec3f & surface_noraml,
    const TriangleMeshSBTData & sbt,
    const gdt::vec3f & wo
  )
  {

    gdt::vec3f N = surface_noraml;
    float roughness = sbt.roughness;
    auto wh = normalize(wi + wo);

    float cosalpha = dot(N, wi);
    float whDotwo = dot(wi, wh);
    // if (cosalpha > 0.0f && whDotwo > 0.0f) 
    if (cosalpha > 0.0f)
    {
      gdt::vec3f Fr = fresnelSchlick(wi, wh, sbt.albedo, sbt.metallic);
      float D = DistributionGGX(N, wh, roughness);
      float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
      float G = GeometrySmith(N, wi, wo, k); //dotProduct(N, wi) / (dotProduct(N, wi) * (1.0f - k) + k);
      gdt::vec3f mirofacet = Fr * G * D / (4.0f * dot(wo, N) * dot(wi, N));
      return (mirofacet * (gdt::vec3f(1.0f) - sbt.kd)) + (sbt.kd * sbt.albedo / M_PI);
    }
    else {
      return gdt::vec3f(0.0f);
    }
  }

  extern "C" __device__ gdt::vec3f __direct_callable__metal_sample(
    RadiancePRD * prd,
    const gdt::vec3f & surface_noraml,
    const TriangleMeshSBTData & sbt,
    const gdt::vec3f & ray_dir
  )
  {
    auto scattered = reflect(ray_dir, surface_noraml);
    return scattered;//toWorld(surface_noraml, scattered);
  }

  extern "C" __device__ float __direct_callable__metal_pdf(
    RadiancePRD * prd,
    const gdt::vec3f & scattered,
    const gdt::vec3f & surface_noraml,
    const TriangleMeshSBTData & sbt,
    const gdt::vec3f & ray_dir
  )
  {
    return 1.0f;
  }

  extern "C" __device__ gdt::vec3f __direct_callable__metal_eval(
    RadiancePRD * prd,
    const gdt::vec3f & scattered,
    const gdt::vec3f & surface_noraml,
    const TriangleMeshSBTData & sbt,
    const gdt::vec3f & ray_dir
  )
  {
    return 1.0f;
  }
}
