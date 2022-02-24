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
  OptixLayer()
    :Layer("OptixLayer"),
    m_Camera(glm::perspectiveFov(glm::radians(45.0f), 1280 * 2.0f, 720 * 2.0f, 0.1f, 10000.0f))
  {}
  virtual ~OptixLayer() {}
  virtual void OnAttach() override
  {
  }
  virtual void OnDetach() override
  {
    //dump
  }
  void OnUpdate() override
  {
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
  std::unique_ptr<Alalba::SampleRenderer> m_sample;

};


class Sandbox02 : public Alalba::Application {


public:
	Sandbox02()
	{
		// PushLayer(new ExampLayer());
		//PushLayer(new Alalba::OptixLayer());
	};
	~Sandbox02() {};
	void OnInit() override
	{
    SampleRenderer sample;

    const vec2i fbSize(vec2i(1200, 1024));
    sample.resize(fbSize);
    sample.render();

    std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
    sample.downloadPixels(pixels.data());

    const std::string fileName = "osc_example2.png";
    stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
      pixels.data(), fbSize.x * sizeof(uint32_t));
    std::cout << GDT_TERMINAL_GREEN
      << std::endl
      << "Image rendered, and saved to " << fileName << " ... done." << std::endl
      << GDT_TERMINAL_DEFAULT
      << std::endl;
  }
};
Alalba::Application* Alalba::CreateApplication() {
	return new Sandbox02();
}