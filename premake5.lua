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
IncludeDir = {}
IncludeDir["GLFW"] = "Alalba/vendor/GLFW/include"
IncludeDir["Glad"] = "Alalba/vendor/Glad/include"
IncludeDir["ImGui"] = "Alalba/vendor/imgui"
IncludeDir["glm"] = "Alalba/vendor/glm"
startproject "Sandbox"

group"Dependencies"
	include "Alalba/vendor/GLFW"
	include "Alalba/vendor/Glad"
	include "Alalba/vendor/imgui"
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
	includedirs
	{
		"%{prj.name}/src",
		"%{prj.name}/vendor/spdlog/include",
		"%{IncludeDir.Glad}",
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.ImGui}",
		"%{IncludeDir.glm}",
		"%{prj.location}/vendor/stb/include",
		"%{prj.location}/vendor/assimp/include"
	}
	links 
	{ 
		"GLFW",
		"Glad",
		"ImGui",
		"opengl32.lib"
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
			("{COPY} vendor/assimp/win64/assimp.dll  ../bin/" ..outputdir .. "/Sandbox")
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
	

project "Sandbox"
	location "Sandbox"
	kind "ConsoleApp"
	staticruntime "on"
	language "C++"
	cppdialect "C++17"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")
	
	files
	{
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp"
	}
	defines
	{
		"ALALBA_PLATFORM_WINDOWS"
	}
	includedirs
	{
		"Alalba/vendor/spdlog/include",
		"Alalba/src",
		"Alalba/vendor",
		"%{IncludeDir.glm}"

	}
	links
	{
		"Alalba",
		"Alalba/vendor/assimp/win64/assimp.lib"
	}
	
	filter "system:windows"
		
		systemversion "latest"

		defines
		{
			"ALALBA_PLATFORM_WINDOWS"
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