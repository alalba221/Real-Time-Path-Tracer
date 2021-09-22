#include "alalbapch.h"
#include "Window.h"
#include "Alalba/Platforms/Windows/GlfwWindow.h"
#include "Alalba/Platforms/Windows/SDLWindow.h"
namespace Alalba{
  //WindowAPI Window::s_WindowAPI = WindowAPI::SDL;
  Window* Window::Create(const WindowProps& props)
	{
		// switch(GetAPI())
    // {
    //   case WindowAPI::GLFW:
    //     return new GlfwWindow(props);
    //   case WindowAPI::SDL:
    //     return new SDLWindow(props);
    // }
    #ifdef SDL_WINDOW_API
      return new SDLWindow(props);
    #else
      return new GlfwWindow(props);
    #endif
	}
}