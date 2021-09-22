#include"alalbapch.h"
#define IMGUI_IMPL_OPENGL_LOADER_GLAD

#include "backends/imgui_impl_opengl3.cpp"
#ifdef SDL_WINDOW_API
  #include "backends/imgui_impl_sdl.cpp"
#else
  #include "backends/imgui_impl_glfw.cpp"
#endif