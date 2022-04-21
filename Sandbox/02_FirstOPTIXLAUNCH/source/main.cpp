#include "Alalba.h"
#include "stb/include/stb_image_write.h"
#include "Alalba/Core/Application.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace Alalba;
using namespace gdt;
using namespace glm;


class OptixLayer : public Alalba::Layer 
{
public:
  OptixLayer(const Model* model)
    :Layer("OptixLayer"),
    m_Camera(glm::vec3(278, 278, -600), glm::vec3(278, 278, 0), glm::vec3(0.f, 1.f, 0.f)),
    sample(model)
  {}
  virtual ~OptixLayer() {}
  virtual void OnAttach() override
  {
    // optix
    fbSize = vec2i(1280.0f * 2, 720.0f * 2);
    sample.resize(fbSize);
    pixels.resize(1280 * 720 * 2 * 2);
    sample.render();
    sample.downloadPixels(pixels.data());
    // opengl
    m_FinalPresentBuffer.reset(Alalba::FrameBuffer::Create(1280 * 2, 720 * 2, Alalba::FrameBufferFormat::RGBA8));
    m_Shader.reset(Alalba::Shader::Create("assets/shaders/shader.glsl"));
    // Create Quad
    float x = -1, y = -1;
    float width = 2, height = 2;
    struct QuadVertex
    {
      glm::vec3 Position;
      glm::vec2 TexCoord;
    };
    QuadVertex* data = new QuadVertex[4];
    data[0].Position = glm::vec3(x, y, 0);
    data[0].TexCoord = glm::vec2(0, 0);

    data[1].Position = glm::vec3(x + width, y, 0);
    data[1].TexCoord = glm::vec2(1, 0);

    data[2].Position = glm::vec3(x + width, y + height, 0);
    data[2].TexCoord = glm::vec2(1, 1);

    data[3].Position = glm::vec3(x, y + height, 0);
    data[3].TexCoord = glm::vec2(0, 1);

    m_VertexBuffer.reset(Alalba::VertexBuffer::Create());
    m_VertexBuffer->SetData(data, 4 * sizeof(QuadVertex));

    uint32_t* indices = new uint32_t[6]{ 0, 1, 2, 2, 3, 0, };
    m_IndexBuffer.reset(Alalba::IndexBuffer::Create());
    m_IndexBuffer->SetData(indices, 6 * sizeof(unsigned int));

    // create an empty texture
    m_OptixResult.reset(Alalba::Texture2D::Create(TextureFormat::RGBA, 1280 * 2, 720 * 2));
    

  }
  virtual void OnDetach() override
  {
    //dump
  }
  void OnUpdate() override
  {
    /// Cam test
    m_Camera.Update();

    sample.setCamera(m_Camera);
   

    // update render data
    sample.render();
    /*pixels.clear();
    pixels.resize(fbSize.x * fbSize.y);*/
    sample.downloadPixels(pixels.data());

   // 

    Alalba::Renderer::Clear(0.2f, 0.3f, 0.8f, 1.0f);
    //if (m_Camera.m_Changed)
    //{
     m_OptixResult->ReloadFromMemory((unsigned char*)pixels.data(), 1280 * 2, 720 * 2);
    //  m_Camera.m_Changed = false;
    //}
    m_Shader->Bind();
    m_OptixResult->Bind(0);
    m_VertexBuffer->Bind();
    m_IndexBuffer->Bind();
    Alalba::Renderer::DrawIndexed(m_IndexBuffer->GetCount(), false);



  }
  virtual void OnImGuiRender() override
  {
  }
  void OnEvent(Alalba::Event& event) override
  {
    // dump
  }
private:
  Alalba::Camera m_Camera;
  std::unique_ptr<Alalba::Texture2D> m_OptixResult;
  //std::unique_ptr<Alalba::SampleRenderer> m_sample;

  std::unique_ptr<Alalba::Shader> m_Shader;
  std::unique_ptr<Alalba::FrameBuffer> m_FrameBuffer, m_FinalPresentBuffer;

  std::unique_ptr<Alalba::VertexBuffer> m_VertexBuffer;
  std::unique_ptr<Alalba::IndexBuffer> m_IndexBuffer;
  SampleRenderer sample;

  vec2i fbSize;
  std::vector<uint32_t> pixels;
};


class Sandbox02 : public Alalba::Application {

public:

	Sandbox02()
	{

    Lambertian* white = new Lambertian(gdt::vec3f(.73f, .73f, .73f));
    Lambertian* red   = new Lambertian(gdt::vec3f(.65f, .05f, .05f));
    Lambertian* green = new Lambertian(gdt::vec3f(.12f, .45f, .15f));
    Diffuse_light* light = new Diffuse_light(gdt::vec3f(1.f, 1.f, 1.f));
   
    Model* model = new Model;
    model->MergeModel(loadOBJ("assets/cornellbox/floor.obj",white));
    model->MergeModel(loadOBJ("assets/cornellbox/left.obj", red));
    model->MergeModel(loadOBJ("assets/cornellbox/right.obj", green));
    model->MergeModel(loadOBJ("assets/cornellbox/light.obj", light));
    model->MergeModel(loadOBJ("assets/cornellbox/shortbox.obj", white));
    model->MergeModel(loadOBJ("assets/cornellbox/tallbox.obj", white));
    model->MergeModel(loadOBJ("assets/bunny/untitled.obj", white));
		PushLayer(new OptixLayer(model));
	};
	~Sandbox02() {};
  void RenderImGui() override
  {
    
  }
};
Alalba::Application* Alalba::CreateApplication() {
	return new Sandbox02();
}