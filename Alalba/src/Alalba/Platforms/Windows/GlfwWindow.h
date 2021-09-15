#pragma once
#include "Alalba/Core/Window.h"
#include <GLFW/glfw3.h>
namespace Alalba{

  class ALALBA_API GlfwWindow :public Window
	{
	public:
		GlfwWindow(const WindowProps& props);

		virtual ~GlfwWindow();

		void OnUpdate();

		inline unsigned int GetWidth()const { return m_Data.Width; };
		inline unsigned int GetHeight() const { return m_Data.Height; };

		//Window Attribute
		inline void SetEventCallback(const EventCallbackFn& callback) { m_Data.EventCallback = callback; };
		void SetVSync(bool enabled);
		bool IsVSync()const;
		inline virtual void* GetNativeWindow() const { return m_Window; };
	
		static Window* Create(const WindowProps& props = WindowProps());
  private:
		virtual void Init(const WindowProps& props);
		virtual void Shutdown();
	private:
		GLFWwindow* m_Window;
		
		struct WindowData 
		{
			std::string Title;
			unsigned int Width, Height;

			bool VSync;
			EventCallbackFn EventCallback;
		};
		WindowData m_Data;
	};
}