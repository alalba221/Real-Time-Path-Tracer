#include "Alalba.h"
//#include "stb/include/stb_image_write.h"
#include "Alalba/Core/Application.h"


class Sandbox02 : public Alalba::Application {
public:
	Sandbox02()
	{
		// PushLayer(new ExampLayer());
		PushLayer(new Alalba::OptixLayer());
	};
	~Sandbox02() {};
};
Alalba::Application* Alalba::CreateApplication() {
	return new Sandbox02();
}