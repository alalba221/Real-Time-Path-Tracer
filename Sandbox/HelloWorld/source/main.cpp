#include "Alalba.h"
#include "imgui/imgui.h"
#include "Alalba/Core/Application.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
class ExampLayer : public Alalba::Layer
{
public:
	ExampLayer()
		: Layer("Example")
	{
	}
	void OnAttach() override
	{
		cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    ALALBA_APP_ASSERT(numDevices != 0, "NO CUDA devices found");
    OptixResult res=optixInit();
    ALALBA_APP_ASSERT(res == OPTIX_SUCCESS, "Optix Init failed");
		ALALBA_APP_INFO("successfully initialized optix... yay!");
	}
	void OnUpdate() override
	{
		//if (Alalba::Input::IsMouseButtonPressed(ALALBA_MOUSE_BUTTON_1))
		//	ALALBA_APP_TRACE("Mouse key is pressed (poll)!");
	}

	virtual void OnImGuiRender() override
	{
		// migui demo line 7663
		//ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
		static bool p_open = true;

		static bool opt_fullscreen_persistant = true;
		static bool opt_padding = false;
		static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
		static bool opt_fullscreen = true;

		// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
		// because it would be confusing to have two docking targets within each others.
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
		if (opt_fullscreen)
		{
			ImGuiViewport* viewport = ImGui::GetMainViewport();
			ImGui::SetNextWindowPos(viewport->Pos);
			ImGui::SetNextWindowSize(viewport->Size);
			ImGui::SetNextWindowViewport(viewport->ID);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
			window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
			window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
		}
		//else
		//{
		//	dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
		//}
		// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
		// and handle the pass-thru hole, so we ask Begin() to not render a background.
		if (ImGuiDockNodeFlags_PassthruCentralNode)
			window_flags |= ImGuiWindowFlags_NoBackground;

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("DockSpace Demo", &p_open, window_flags);
		ImGui::PopStyleVar();

		if (opt_fullscreen)
			ImGui::PopStyleVar(2);

		// Dockspace
		ImGuiIO& io = ImGui::GetIO();
		if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
		{
			ImGuiID dockspace_id = ImGui::GetID("MyDockspace");
			ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
		}
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("Options"))
			{
				// Disabling fullscreen would allow the window to be moved to the front of other windows,
				// which we can't undo at the moment without finer window depth/z control.
				ImGui::MenuItem("Fullscreen", NULL, &opt_fullscreen);
				ImGui::MenuItem("Padding", NULL, &opt_padding);
				ImGui::Separator();

				if (ImGui::MenuItem("Flag: NoSplit", "", (dockspace_flags & ImGuiDockNodeFlags_NoSplit) != 0)) { dockspace_flags ^= ImGuiDockNodeFlags_NoSplit; }
				if (ImGui::MenuItem("Flag: NoResize", "", (dockspace_flags & ImGuiDockNodeFlags_NoResize) != 0)) { dockspace_flags ^= ImGuiDockNodeFlags_NoResize; }
				if (ImGui::MenuItem("Flag: NoDockingInCentralNode", "", (dockspace_flags & ImGuiDockNodeFlags_NoDockingInCentralNode) != 0)) { dockspace_flags ^= ImGuiDockNodeFlags_NoDockingInCentralNode; }
				if (ImGui::MenuItem("Flag: AutoHideTabBar", "", (dockspace_flags & ImGuiDockNodeFlags_AutoHideTabBar) != 0)) { dockspace_flags ^= ImGuiDockNodeFlags_AutoHideTabBar; }
				if (ImGui::MenuItem("Flag: PassthruCentralNode", "", (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) != 0, opt_fullscreen)) { dockspace_flags ^= ImGuiDockNodeFlags_PassthruCentralNode; }
				ImGui::Separator();

				if (ImGui::MenuItem("Close", NULL, false, p_open != NULL))
					p_open = false;
				ImGui::EndMenu();
			}
			//HelpMarker(
			//	"When docking is enabled, you can ALWAYS dock MOST window into another! Try it now!" "\n"
			//	"- Drag from window title bar or their tab to dock/undock." "\n"
			//	"- Drag from window menu button (upper-left button) to undock an entire node (all windows)." "\n"
			//	"- Hold SHIFT to disable docking (if io.ConfigDockingWithShift == false, default)" "\n"
			//	"- Hold SHIFT to enable docking (if io.ConfigDockingWithShift == true)" "\n"
			//	"This demo app has nothing to do with enabling docking!" "\n\n"
			//	"This demo app only demonstrate the use of ImGui::DockSpace() which allows you to manually create a docking node _within_ another window." "\n\n"
			//	"Read comments in ShowExampleAppDockSpace() for more details.");

			ImGui::EndMenuBar();
		}

		ImGui::End();
		//ImGui::Begin("Test");
		//ImGui::Text("Hello World");
		//ImGui::End();
	}

	void OnEvent(Alalba::Event& event) override
	{
		if (event.GetEventType() == Alalba::EventType::KeyPressed)
		{
			Alalba::KeyPressedEvent& e = (Alalba::KeyPressedEvent&)event;
			if (e.GetKeyCode() == ALALBA_TAB)
				ALALBA_APP_TRACE("Tab key is pressed (event)!");
			//	ALALBA_APP_TRACE("{0}", (char)e.GetKeyCode());
		}
	}

};

class Sandbox : public Alalba::Application {
public:
	Sandbox()
	{
		PushLayer(new ExampLayer());
		//PushLayer(new Alalba::OptixLayer());
	};
	~Sandbox() {};
};
Alalba::Application* Alalba::CreateApplication() {
	return new Sandbox();
}