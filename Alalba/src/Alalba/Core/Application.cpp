#include "alalbapch.h"
#include "Application.h"
#include "Alalba/Core/Events/ApplicationEvent.h"
#include "Alalba/Core/Log.h"
#include <glad/glad.h>
namespace Alalba{
  #define BIND_ENVENT_FN(x) std::bind(&x, this, std::placeholders::_1)

	Application* Application::s_Instance = nullptr;

  Application::Application(){
		s_Instance = this;
		
	  m_Window = std::unique_ptr<Window>(new Window(WindowProps()));
    m_Window->SetEventCallback(BIND_ENVENT_FN(Application::OnEvent));

		m_ImGuiLayer = new ImGuiLayer();
		PushOverlayer(m_ImGuiLayer);

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
		for (Layer* layer : m_LayerStack)
			layer->OnImGuiRender();

		m_ImGuiLayer->End();

	}
  bool Application::OnWindowClose(WindowCloseEvent& e)
	{
		m_Running = false;	
		return true;
	}
  void Application::Run(){
    Event* e	= new WindowResizeEvent(1220, 970);
		if (e->IsInCategory(EventCategoryApplication))
		{
			ALALBA_APP_TRACE(e->ToString());
		}
		if (e->IsInCategory(EventCategoryInput))
		{
			ALALBA_APP_TRACE(e->ToString() );
		}
    while(m_Running){
      glClearColor(1, 0, 1, 1);
			glClear(GL_COLOR_BUFFER_BIT);

			for (Layer* layer : m_LayerStack)
				layer->OnUpdate();

			Application* app = this;
			this->RenderImGui();

			m_Window->OnUpdate();

    }
  }
}