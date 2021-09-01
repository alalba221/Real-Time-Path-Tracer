#include "Alalba.h"
#include "imgui/imgui.h"
class ExampleLayer : public Alalba::Layer
{
public:
	ExampleLayer()
		: Layer("Example")
	{
	}

	void OnUpdate() override
	{
		if (Alalba::Input::IsMouseButtonPressed(ALALBA_MOUSE_BUTTON_1))
			ALALBA_APP_TRACE("Mouse key is pressed (poll)!");
	}

	virtual void OnImGuiRender() override
	{
		ImGui::Begin("Test");
		ImGui::Text("Hello World");
		ImGui::End();
	}

	void OnEvent(Alalba::Event& event) override
	{
		if (event.GetEventType() == Alalba::EventType::KeyPressed)
		{
			Alalba::KeyPressedEvent& e = (Alalba::KeyPressedEvent&)event;
			if (e.GetKeyCode() == ALALBA_KEY_TAB)
					ALALBA_APP_TRACE("Tab key is pressed (event)!");
			//	ALALBA_APP_TRACE("{0}", (char)e.GetKeyCode());
		}
	}

};

class Sandbox : public Alalba::Application{
public:
  Sandbox()
  {
    PushLayer(new ExampleLayer());
  };
  ~Sandbox(){};
};
Alalba::Application* Alalba::CreateApplication(){
  return new Sandbox();
}