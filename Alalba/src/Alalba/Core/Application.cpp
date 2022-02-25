#include "alalbapch.h"
#include "Application.h"
#include "Alalba/Core/Events/ApplicationEvent.h"

#include "Alalba/Renderer/Renderer.h"
#include "Alalba/Renderer/Framebuffer.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
//
//#define GLFW_EXPOSE_NATIVE_WIN32
//#include <GLFW/glfw3native.h>
//#include <Windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <Windows.h>

namespace Alalba{
  #define BIND_ENVENT_FN(x) std::bind(&x, this, std::placeholders::_1)

	Application* Application::s_Instance = nullptr;

  Application::Application(){
		s_Instance = this;
		
	  m_Window = std::unique_ptr<Window>(Window::Create());
    m_Window->SetEventCallback(BIND_ENVENT_FN(Application::OnEvent));

		m_ImGuiLayer = new ImGuiLayer();
		PushOverlayer(m_ImGuiLayer);
		
		Renderer::Init();
  }
  Application::~Application(){

  }

  void Application::PushLayer(Layer* layer)
	{
		m_LayerStack.PushLayer(layer);
		layer->OnAttach();
	}
	void Application::PushOverlayer(Layer* overlayer)
	{
		m_LayerStack.PushOverlayer(overlayer);
		overlayer->OnAttach();
	}
	

  void Application::OnEvent(Event& e){
    EventDispatcher dispatcher(e);
    dispatcher.Dispatch<WindowCloseEvent>(BIND_ENVENT_FN(Application::OnWindowClose));
    for (auto it = m_LayerStack.end(); it != m_LayerStack.begin();)
		{
			(*--it)->OnEvent(e);
			if (e.Handled)
			{
				break;
			}
		}
  }
	void Application::RenderImGui()
	{
		m_ImGuiLayer->Begin();
		ImGui::Begin("Renderer");
		auto& caps = RendererAPI::GetCapabilities();
		ImGui::Text("Vendor: %s", caps.Vendor.c_str());
		ImGui::Text("Renderer: %s", caps.Renderer.c_str());
		ImGui::Text("Version: %s", caps.Version.c_str());
		ImGui::End();

		for (Layer* layer : m_LayerStack)
			layer->OnImGuiRender();

		m_ImGuiLayer->End();
	}

	bool Application::OnWindowResize(WindowResizeEvent& e)
	{
		int width = e.GetWidth(), height = e.GetHeight();
		if (width == 0 || height == 0)
		{
			m_Minimized = true;
			return false;
		}
		m_Minimized = false;

		ALALBA_RENDER_2(width, height, { glViewport(0, 0, width, height); });
		auto& fbs = FrameBufferPool::GetGlobal()->GetAll();
		for (auto& fb : fbs)
			fb->Resize(width, height);
		return false;

	}

  bool Application::OnWindowClose(WindowCloseEvent& e)
	{
		m_Running = false;	
		return true;
	}
  void Application::Run(){
		OnInit();
    while(m_Running){
			if (!m_Minimized)
			{
				// Maybe we should put the clear command into the Que Here
				// Not in the each layer->OnUpdate() function
				for (Layer* layer : m_LayerStack)
					layer->OnUpdate();

				Application* app = this;
				ALALBA_RENDER_1(app, { app->RenderImGui(); });
				Renderer::Get().WaitAndRender();
			}
			m_Window->OnUpdate();
    }
		OnShutdown();
  }

	std::string Application::OpenFile(const std::string& filter) const
	{
		OPENFILENAMEA ofn;       // common dialog box structure
		CHAR szFile[260] = { 0 };       // if using TCHAR macros

		// Initialize OPENFILENAME
		ZeroMemory(&ofn, sizeof(OPENFILENAME));
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.hwndOwner = glfwGetWin32Window((GLFWwindow*)m_Window->GetNativeWindow());
		ofn.lpstrFile = szFile;
		ofn.nMaxFile = sizeof(szFile);
		ofn.lpstrFilter = "All\0*.*\0";
		ofn.nFilterIndex = 1;
		ofn.lpstrFileTitle = NULL;
		ofn.nMaxFileTitle = 0;
		ofn.lpstrInitialDir = NULL;
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

		if (GetOpenFileNameA(&ofn) == TRUE)
		{
			return ofn.lpstrFile;
		}
		return std::string();
	}

}