#pragma once
#include "Alalba/Core/Window.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
namespace Alalba{

  class ALALBA_API  SDLWindow :public Window
	{
	public:
		SDLWindow(const WindowProps& props);

		virtual ~SDLWindow();

		void OnUpdate();

		inline unsigned int GetWidth()const { return m_Data.Width; };
		inline unsigned int GetHeight() const { return m_Data.Height; };

		//Window Attribute
		inline void SetEventCallback(const EventCallbackFn& callback) { m_Data.EventCallback = callback; };
		void SetVSync(bool enabled);
		bool IsVSync()const;
		inline virtual void* GetNativeWindow() const { return m_Window; };
	
		static Window* Create(const WindowProps& props = WindowProps());
		//inline SDL_GLContext GetCurContext() const {return m_Context;};
  private:
		virtual void Init(const WindowProps& props);
		virtual void Shutdown() override;
	private:
		SDL_Window* m_Window;
		SDL_GLContext m_Context;
		
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