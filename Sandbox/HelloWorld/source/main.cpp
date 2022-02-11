#include "Alalba.h"
#include "imgui/imgui.h"
//#include "Alalba/Core/Application.h"


class Sandbox : public Alalba::Application {
public:
	Sandbox()
	{
		// PushLayer(new ExampLayer());
		//PushLayer(new Alalba::OptixLayer());
	};
	~Sandbox() {};
};
Alalba::Application* Alalba::CreateApplication() {
	return new Sandbox();
}