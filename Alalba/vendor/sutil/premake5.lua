project "sutil"
	kind "StaticLib"
	language "C++"

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")
	
	local CUDA_INCLUDE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/include"
	local CUDA_EXTRA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/extras/CUPTI/include"
	local CUDA_LIB_DIR =  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/lib/x64"
	local OPTIX_ROOT = "C:/ProgramData/NVIDIA Corporation"
	local OPTIX7_INCLUDE_DIR = OPTIX_ROOT .. "/OptiX SDK 7.4.0/include"
	
	local VENDOR_DIR = "%{wks.location}/Alalba/vendor/"
	
	includedirs
	{
		
		CUDA_INCLUDE_DIR,
		CUDA_EXTRA_DIR,
		OPTIX7_INCLUDE_DIR,
		VENDOR_DIR,
		VENDOR_DIR.."sutil"
    }
	links 
	{ 
		"cudart_static",
		"cuda",
		"nvrtc",
    }
	files
	{
       "*.h", 
       "*.hpp", 
       "*.c",
       "*.cpp",
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
		
	filter "configurations:Debug"
		runtime "Debug"
		symbols "on"
		libdirs { 
				  CUDA_LIB_DIR
		}

	filter "configurations:Release"
		runtime "Release"
		optimize "on"
		libdirs {
				  CUDA_LIB_DIR
		}
