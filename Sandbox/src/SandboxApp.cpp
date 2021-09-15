#include "Alalba.h"

static void ImGuiShowHelpMarker(const char* desc)
{
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}


class EditorLayer : public Alalba::Layer
{
public:
	EditorLayer()
		: Layer("Example"),m_ClearColor{0.2f,0.3f,0.8f,1.0f}
	{
	}

	void OnUpdate() override
	{
		if (Alalba::Input::IsMouseButtonPressed(ALALBA_MOUSE_BUTTON_1))
			ALALBA_APP_TRACE("Mouse key is pressed (poll)!");
		Alalba::Renderer::Clear(m_ClearColor[0], m_ClearColor[1], m_ClearColor[2], m_ClearColor[3]);
	}

	virtual void OnImGuiRender() override
	{
		static bool show_demo_window = true;
		if (show_demo_window)
			ImGui::ShowDemoWindow(&show_demo_window);

		ImGui::Begin("GameLayer");
		ImGui::ColorEdit4("Clear Color", m_ClearColor);
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
	private: 
		float m_ClearColor[4];
};

class Sandbox : public Alalba::Application{
public:
  Sandbox()
  {
    PushLayer(new EditorLayer());
  };
  ~Sandbox(){};
};
Alalba::Application* Alalba::CreateApplication(){
  return new Sandbox();
}