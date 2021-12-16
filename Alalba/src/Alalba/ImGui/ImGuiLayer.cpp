#include "alalbapch.h"
#include "ImGuiLayer.h"
#include "Alalba/Core/Application.h"

#ifdef SDL_WINDOW_API
  #include "backends/imgui_impl_sdl.h"
  #include <SDL2/SDL.h>
#else
  #include "backends/imgui_impl_glfw.h"
  #include <GLFW/glfw3.h>
#endif


#include "backends/imgui_impl_opengl3.h"

#include "imgui.h"
// TEMPORARY

#include <glad/glad.h>


// From imgui example 
namespace Alalba {
  ImGuiLayer::ImGuiLayer()
	{

	}

	ImGuiLayer::ImGuiLayer(const std::string& name)
	{

	}

	ImGuiLayer::~ImGuiLayer()
	{

	}


	void ImGuiLayer::OnAttach()
	{
    // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
        //io.ConfigViewportsNoAutoMerge = true;
        //io.ConfigViewportsNoTaskBarIcon = true;

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

         //When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
         ImGuiStyle& style = ImGui::GetStyle();
         if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
         {
             style.WindowRounding = 0.0f;
             style.Colors[ImGuiCol_WindowBg].w = 1.0f;
         }
        Application& app = Application::Get();

#ifdef SDL_WINDOW_API
        SDL_Window* window = static_cast<SDL_Window*>(app.GetWindow().GetNativeWindow());
        SDL_GLContext gl_context = SDL_GL_GetCurrentContext();
        ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
#else
        GLFWwindow* window = static_cast<GLFWwindow*>(app.GetWindow().GetNativeWindow());

            // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
#endif
        ImGui_ImplOpenGL3_Init("#version 410");
	}
	void ImGuiLayer::OnDetach()
	{
        ImGui_ImplOpenGL3_Shutdown();
#ifdef SDL_WINDOW_API
        ImGui_ImplSDL2_Shutdown();
#else
        ImGui_ImplGlfw_Shutdown();
#endif
        ImGui::DestroyContext();

	}

    void ImGuiLayer::Begin()
    {
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
#ifdef SDL_WINDOW_API
        ImGui_ImplSDL2_NewFrame();
#else
        ImGui_ImplGlfw_NewFrame();
#endif
        ImGui::NewFrame();
    }
    void ImGuiLayer::End()
    {
        ImGuiIO& io = ImGui::GetIO();
        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
          GLFWwindow* backup_current_context = glfwGetCurrentContext();
          ImGui::UpdatePlatformWindows();
          ImGui::RenderPlatformWindowsDefault();
          glfwMakeContextCurrent(backup_current_context);
        }
    }
    void ImGuiLayer::OnImGuiRender()
    {
    }
	
	
}
