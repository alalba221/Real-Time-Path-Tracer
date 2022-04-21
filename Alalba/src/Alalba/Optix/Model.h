// ======================================================================== //
//            Later will be merged with Render/Mesh                         //
// ======================================================================== //

#pragma once

#include "math/AffineSpace.h"
#include "Material.h"
/*! \namespace osc - Optix Siggraph Course */
namespace Alalba {
  using namespace gdt;

  /*! a simple indexed triangle mesh that our sample renderer will
      render */
  // each trianglemesh represent an obj(geometry) file
  struct TriangleMesh {
    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;
    
    // material data:
    //vec3f              diffuse;
    Material* material;
  };

  struct Model {
    ~Model()
    {
      for (auto mesh : meshes) delete mesh;
    }
    // for cornell box, each TriangleMesh* points to an obj
    std::vector<TriangleMesh*> meshes;
    //! bounding box of all vertices in the model
    void MergeModel(Model* model);
    box3f bounds;
  };

  Model* loadOBJ(const std::string& objFile, Material* material);

  struct Scene
  {
    std::vector<Model*> models;
    
    void Add(Model* model) {
      models.push_back(model);
    }
  };
}
