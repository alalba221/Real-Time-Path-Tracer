#pragma once

#include "alalbapch.h"
#include "Alalba/Core/Base.h"
#include "Alalba/Core/Events/Event.h"

#include <GLFW/glfw3.h>
namespace Alalba
{

	class WindowProps 
	{
	public:
		std::string Title;
		unsigned int Width;
		unsigned int Height;

		WindowProps(const std::string& title="Alalba Engine",
			unsigned int width = 1280,
			unsigned int height = 720)
			: Title(title),Width(width),Height(height)
		{

		}
	};


	class ALALBA_API Window
	{
	public:
	
  	using EventCallbackFn = std::function<void(Event&)>;
		Window(const WindowProps& props);

		virtual ~Window();

		void OnUpdate();

		inline unsigned int GetWidth()const { return m_Data.Width; };
		inline unsigned int GetHeight() const { return m_Data.Height; };

		//Window Attribute
		inline void SetEventCallback(const EventCallbackFn& callback) { m_Data.EventCallback = callback; };
		void SetVSync(bool enabled);
		bool IsVSync()const;
		inline virtual GLFWwindow* GetNativeWindow() const { return m_Window; };
	
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