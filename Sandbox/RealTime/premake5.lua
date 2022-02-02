project "RealTime"
	kind "ConsoleApp"
	staticruntime "on"
	language "C++"
	cppdialect "C++17"

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin-int/" .. outputdir .. "/%{prj.name}")
	
local SOURCE_DIR = "source/*"
    files
    { 
      SOURCE_DIR .. "**.h", 
      SOURCE_DIR .. "**.hpp", 
      SOURCE_DIR .. "**.c",
      SOURCE_DIR .. "**.cpp",
    }
	
	defines
	{
		"ALALBA_PLATFORM_WINDOWS"
	}
	

	
	includedirs
	{
		"%{wks.location}/Alalba/vendor/spdlog/include",
		"%{wks.location}/Alalba/src",
		"%{wks.location}/Alalba/vendor",
		"%{IncludeDir.glm}"

	}
	links
	{
		"Alalba",
		"%{wks.location}/Alalba/vendor/assimp/win64/assimp.lib"
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