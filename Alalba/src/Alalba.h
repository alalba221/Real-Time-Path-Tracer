#pragma once

//For use by user application
#include "Alalba/Core/Application.h"
#include "Alalba/Core/Log.h"
#include "Alalba/Core/Layer.h"

#include "Alalba/Core/Events/Event.h"
#include "Alalba/Core/Events/ApplicationEvent.h"
#include "Alalba/Core/Events/KeyEvent.h"
#include "Alalba/Core/Events/MouseEvent.h"

#include "Alalba/Core/Input.h"
#include "Alalba/Core/KeyCodes.h"
#include "Alalba/Core/MouseButtonCodes.h"

#include "Alalba/ImGui/ImGuiLayer.h"
#include "imgui/imgui.h"

// ---  Render API ------------------------------
#include "Alalba/Renderer/Renderer.h"
#include "Alalba/Renderer/FrameBuffer.h"
#include "Alalba/Renderer/Buffer.h"
#include "Alalba/Renderer/Texture.h"
#include "Alalba/Renderer/Shader.h"
//Entry Point
#include "Alalba/EntryPoint.h"