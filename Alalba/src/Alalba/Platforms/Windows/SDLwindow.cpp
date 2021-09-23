#include "alalbapch.h"
#include "SDLWindow.h"
#include "Alalba/Core/Events/Event.h"
#include "Alalba/Core/Events/ApplicationEvent.h"
#include "Alalba/Core/Events/KeyEvent.h"
#include "Alalba/Core/Events/MouseEvent.h"
#include "Alalba/Core/Log.h"

#include <glad/glad.h>
namespace Alalba 
{
	static bool s_SDLInitialized = false;

	SDLWindow::SDLWindow(const WindowProps& prop)
	{
		ALALBA_CORE_INFO("SDL WINDOW API");
		Init(prop);
	}
	SDLWindow::~SDLWindow()
	{
		Shutdown();
	}
	void SDLWindow::OnUpdate()
	{
		SDL_GL_SwapWindow(m_Window);
		/// Event loop
		SDL_Event sdlevent;
		//SDL_WaitEvent(&sdlevent);
		do {
				WindowData* data = (WindowData*)SDL_GetWindowData(m_Window,"data");
				switch (sdlevent.type) 
				{
					// Window event
					case SDL_WINDOWEVENT:
					{
						switch(sdlevent.window.event)
						{
							case	SDL_WINDOWEVENT_CLOSE:
							{
								WindowCloseEvent event;
								data->EventCallback(event);
								break;
							}
							case SDL_WINDOWEVENT_RESIZED:
							{
								data->Width = sdlevent.window.data1;
								data->Height = sdlevent.window.data2;
								WindowResizeEvent event(sdlevent.window.data1, sdlevent.window.data2);
								data->EventCallback(event);
							}								
						}
						break;
					}
					//Key board
					case SDL_KEYDOWN:
					{
						KeyPressedEvent event(sdlevent.key.keysym.scancode, sdlevent.key.repeat);
						data->EventCallback(event);
						break;
					}
					case SDL_KEYUP:
					{
						KeyReleasedEvent event(sdlevent.key.keysym.scancode);
						data->EventCallback(event);
						break;
					}
					// Mouse Event
					case SDL_MOUSEMOTION:
					{
						MouseMovedEvent event((float)sdlevent.motion.x, (float)sdlevent.motion.y);
						//ALALBA_CORE_INFO("Mouse {0},{1}",(float)sdlevent.motion.xrel,(float)sdlevent.motion.yrel);
						data->EventCallback(event);
						break;
					}
					case SDL_MOUSEBUTTONDOWN:
					{
						MouseButtonPressedEvent event(sdlevent.button.button);
						data->EventCallback(event);
						break;
					}
					case SDL_MOUSEBUTTONUP:
					{
						MouseButtonReleasedEvent event(sdlevent.button.button);
						data->EventCallback(event);
						
						break;
					}
					case SDL_MOUSEWHEEL:
					{
						MouseScrolledEvent event((float)sdlevent.wheel.x, (float)sdlevent.wheel.y);
						data->EventCallback(event);
						break;
					}
					default:
						break;
				}
		} while (SDL_PollEvent(&sdlevent));
		
	}
	void SDLWindow::SetVSync(bool enabled)
	{
		if (enabled)
		{
			SDL_GL_SetSwapInterval(1);
		}
		else
		{
			SDL_GL_SetSwapInterval(0);
		}
		m_Data.VSync = enabled;
	}
	bool SDLWindow::IsVSync() const
	{
		return m_Data.VSync;
	}

	void SDLWindow::Init(const WindowProps& props)
	{
		m_Data.Title = props.Title;
		m_Data.Height = props.Height;
		m_Data.Width = props.Width;

		ALALBA_CORE_INFO("Create window{0} ({1},{2})", props.Title, props.Height, props.Width);

		if (!s_SDLInitialized)
		{
			int sucess = SDL_Init(SDL_INIT_EVERYTHING);
			//glfwSetErrorCallback(GLFWErrorCallback);
			if(sucess<0)
				ALALBA_CORE_ERROR("Could not intialize SDL!");
			s_SDLInitialized = true;
		}
		
    m_Window = SDL_CreateWindow(props.Title.c_str(),
                              SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED,
                              (int)props.Width,
                              (int)props.Height,
                              SDL_WINDOW_OPENGL|SDL_WINDOW_SHOWN);
		
		if (!m_Window)
    {
      ALALBA_CORE_ERROR("Cannot create window {0}",SDL_GetError());
      SDL_Quit();
    }
    m_Context = SDL_GL_CreateContext(m_Window);
		SDL_GL_MakeCurrent(m_Window, m_Context);
    int status = gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress);
		SDL_SetWindowData(m_Window,"data",&m_Data);
		SetVSync(true);
	}
	void SDLWindow::Shutdown()
	{
		SDL_GL_DeleteContext(m_Context);
		SDL_DestroyWindow(m_Window);
    SDL_Quit();
	}
}
