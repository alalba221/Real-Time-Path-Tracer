project "gdt"
	kind "StaticLib"
	language "C++"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

	includedirs
	{
		
		"source",
    }

	files
	{
       "source/**.h", 
       "source/**.hpp", 
       "source/**.c",
       "source/**.cpp",
    }

	filter "system:windows"
		systemversion "latest"
		cppdialect "C++17"
		staticruntime "On"
		disablewarnings { "4244", "5030" }
		characterset "MBCS"
       
		defines 
		{ 
			"_WIN32",
			"_WINDOWS",
            "_CRT_SECURE_NO_WARNINGS",
			"NOMINMAX"
		}

	filter "system:linux"
		pic "On"
		systemversion "latest"
		cppdialect "C++17"
		staticruntime "On"

	filter "configurations:Debug"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"
