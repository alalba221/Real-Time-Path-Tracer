// ======================================================================== //
//            Later will be merged with Render/Mesh                         //
// ======================================================================== //

#pragma once

#include "math/AffineSpace.h"
#include <vector>

/*! \namespace osc - Optix Siggraph Course */
namespace Alalba {
  using namespace gdt;

  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  struct TriangleMesh {
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
    vec3f              diffuse;
  };

  struct Sphere
  {
    gdt::vec3f center;
    float  radius;
  };

  struct SphereShell
  {
    gdt::vec3f 	center;
    float 	radius1;
    float 	radius2;
  };

  struct Parallelogram
  {
    Parallelogram() = default;
    Parallelogram(gdt::vec3f v1, gdt::vec3f v2, gdt::vec3f anchor) :
      v1(v1), v2(v2), anchor(anchor)
    {
      gdt::vec3f normal = normalize(cross(v1, v2));
      float d = dot(normal, anchor);
      this->v1 *= 1.0f / dot(v1, v1);
      this->v2 *= 1.0f / dot(v2, v2);
      plane = gdt::vec4f(normal, d);
    }
    gdt::vec4f	plane;
    gdt::vec3f 	v1;
    gdt::vec3f 	v2;
    gdt::vec3f 	anchor;
  };

  /// Material
  struct Phong
  {
    gdt::vec3f Ka;
    gdt::vec3f Kd;
    gdt::vec3f Ks;
    gdt::vec3f Kr;
    float  phong_exp;
  };


  struct Glass
  {
    float  importance_cutoff;
    gdt::vec3f cutoff_color;
    float  fresnel_exponent;
    float  fresnel_minimum;
    float  fresnel_maximum;
    float  refraction_index;
    gdt::vec3f refraction_color;
    gdt::vec3f reflection_color;
    gdt::vec3f extinction_constant;
    gdt::vec3f shadow_attenuation;
    int    refraction_maxdepth;
    int    reflection_maxdepth;
  };


  struct CheckerPhong
  {
    gdt::vec3f Kd1, Kd2;
    gdt::vec3f Ka1, Ka2;
    gdt::vec3f Ks1, Ks2;
    gdt::vec3f Kr1, Kr2;
    float  phong_exp1, phong_exp2;
    gdt::vec2f inv_checker_size;
  };


  struct Model {
    Model()
    {}
    ~Model()
    {
      for (auto mesh : meshes) delete mesh;
      for (auto sphere : spheres) delete sphere;
      for (auto sphereshell : sphereshells) delete sphereshell;
      for (auto parallelogram : parallelograms) delete parallelogram;
    }

    std::vector<TriangleMesh*> meshes;
    std::vector<Sphere*> spheres;
    std::vector<SphereShell*> sphereshells;
    std::vector<Parallelogram*> parallelograms;
    //! bounding box of all vertices in the model
    box3f bounds;

    void addUnitCube(const gdt::affine3f& xfm);
    //! add aligned cube aith front-lower-left corner and size
    void addCube(const gdt::vec3f& center, const gdt::vec3f& size);
  };

  Model* loadOBJ(const std::string& objFile);
}

