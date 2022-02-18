workspace "Alalba"
	architecture "x64"
	configurations
	{
		"Debug",
		"Release",
		"Dist"
	}
	
outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

--Include directories relatice to the script current running
--IncludeDir = {}
--IncludeDir["GLFW"] = "Alalba/vendor/GLFW/include"
--IncludeDir["Glad"] = "Alalba/vendor/Glad/include"
--IncludeDir["ImGui"] = "Alalba/vendor/imgui"
--IncludeDir["glm"] = "Alalba/vendor/glm"

	--Include directories relatice to the script current running
	IncludeDir = {}
	IncludeDir["GLFW"] = "%{wks.location}/Alalba/vendor/GLFW/include"
	IncludeDir["Glad"] = "%{wks.location}/Alalba/vendor/Glad/include"
	IncludeDir["ImGui"] = "%{wks.location}/Alalba/vendor/imgui"
	IncludeDir["glm"] = "%{wks.location}/Alalba/vendor/glm"
	IncludeDir["gdt"] = "%{wks.location}/Alalba/vendor/gdt/source"

group"Dependencies"
	include "Alalba/vendor/GLFW"
	include "Alalba/vendor/Glad"
	include "Alalba/vendor/imgui"
	include "Alalba/vendor/gdt"
group""

project "Alalba"
	location "Alalba"
	kind "StaticLib"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	pchheader "alalbapch.h"
	pchsource "Alalba/src/alalbapch.cpp"
	
	files
	{
		"%{prj.name}/src/**.h",
		"%{prj.name}/vendor/glm/glm/**.hpp",
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/vendor/glm/glm/**.inl"
	}
	defines
	{
		"ALALBA_PLATFORM_WINDOWS",
		"_CRT_SECURE_NO_WARNINGS"
	}
	
	local CUDA_INCLUDE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/include"
	local CUDA_EXTRA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/extras/CUPTI/include"
	local CUDA_LIB_DIR =  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/lib/x64"
	local OPTIX_ROOT = "C:/ProgramData/NVIDIA Corporation"
	local OPTIX7_INCLUDE_DIR = OPTIX_ROOT .. "/OptiX SDK 7.4.0/include"
	includedirs
	{
		"%{prj.name}/src",
		"%{prj.name}/vendor/spdlog/include",
		"%{IncludeDir.Glad}",
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.ImGui}",
		"%{IncludeDir.glm}",
		"%{IncludeDir.gdt}",
		"%{prj.location}/vendor/stb/include",
		"%{prj.location}/vendor/assimp/include",
		CUDA_INCLUDE_DIR,
		CUDA_EXTRA_DIR,
		OPTIX7_INCLUDE_DIR
	}
	links 
	{ 
		"GLFW",
		"Glad",
		"ImGui",
		"gdt",
		"opengl32.lib",
		--"cuda",
		--"nvrtc",
	}

	filter "system:windows"
		
		systemversion "latest"

		defines
		{
			"ALALBA_PLATFORM_WINDOWS",
			"ALALBA_BUILD_DLL",
			"GLFW_INCLUDE_NONE"
		}
		-- Directories relatice to project folder
		postbuildcommands
    {
			--("{COPY} %{cfg.buildtarget.relpath} ../bin/"..outputdir.."/Sandbox")
			("{COPY} vendor/assimp/win64/assimp.dll  ../bin/" ..outputdir .. "/RealTime")
    }
	filter "configurations:Debug"
		defines "ALALBA_DEBUG"
		runtime "Debug"
		symbols "on"
	filter "configurations:Release"
		defines "ALALBA_RELEASE"
		runtime "Release"
		optimize "on"
	filter "configurations:Dist"
		defines "ALALBA_Dist"
		runtime "Release"
		optimize "on"
		
-- add settings common to all project


include "Sandbox/RealTime"
include "Sandbox/HelloWorld"
include "Sandbox/02_FirstOPTIXLAUNCH"