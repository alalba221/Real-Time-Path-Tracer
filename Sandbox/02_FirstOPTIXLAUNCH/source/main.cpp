#include "Alalba.h"
#include "stb/include/stb_image_write.h"
#include "Alalba/Core/Application.h"

using namespace Alalba;
using namespace gdt;

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