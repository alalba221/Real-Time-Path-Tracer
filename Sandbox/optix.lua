
	language "C++"

--	defines{
--		"NANOGUI_GLAD", "JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED", "POCO_NO_AUTOMATIC_LIBS",
--		"_USE_MATH_DEFINES", "_ENABLE_EXTENDED_ALIGNED_STORAGE",
--		"TINYGLTF_NO_INCLUDE_JSON",
--			"TINYGLTF_NO_INCLUDE_STB_IMAGE", 
--			"TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE",
--			"TINYGLTF_USE_CPP14",
--	}

	flags { "MultiProcessorCompile", "NoMinimalRebuild" }
	


	local CUDA_INCLUDE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/include"
	local CUDA_EXTRA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/extras/CUPTI/include"
	local CUDA_LIB_DIR =  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/lib/x64"
	local OPTIX_ROOT = "C:/ProgramData/NVIDIA Corporation"
	local OPTIX7_INCLUDE_DIR = OPTIX_ROOT .. "/OptiX SDK 7.4.0/include"
	
	local VENDOR_DIR = "%{wks.location}/Alalba/vendor/"
	
	includedirs
	{
		--JAHLEY_DIR,
		--MODULE_DIR,
		CUDA_INCLUDE_DIR,
		CUDA_EXTRA_DIR,
		OPTIX7_INCLUDE_DIR,
		VENDOR_DIR.."gdt/source"
		
		-- THIRD_PARTY_DIR,
		-- THIRD_PARTY_DIR .. "nanogui/ext/glfw/include",
		-- THIRD_PARTY_DIR .. "nanogui/include",
		-- THIRD_PARTY_DIR .. "nanogui/ext/eigen",
		-- THIRD_PARTY_DIR .. "nanogui/ext/glad/include",
		-- THIRD_PARTY_DIR .. "nanogui/ext/nanovg/src",
		-- THIRD_PARTY_DIR .. "g3log/src",
		-- THIRD_PARTY_DIR .. "PocoFoundationLite/include",
		-- THIRD_PARTY_DIR .. "openexr/source",
		-- THIRD_PARTY_DIR .. "openexr/source/*",
		-- THIRD_PARTY_DIR .. "blosc/include",
		-- THIRD_PARTY_DIR .. "taskflow",
		-- THIRD_PARTY_DIR .. "stb/include",
		-- THIRD_PARTY_DIR .. "libigl/include",
		-- THIRD_PARTY_DIR .. "cs_signal/source",
		-- THIRD_PARTY_DIR .. "doctest",
		-- THIRD_PARTY_DIR .. "concurrent/include",
		-- THIRD_PARTY_DIR .. "gdt/source",	
		-- THIRD_PARTY_DIR .. "json/single_include/nlohmann",
		-- THIRD_PARTY_DIR .. "tinygltf/include",
	}
	
	links 
	{ 
		"Alalba",
		-- "Nanogui",
		-- "GLFW",
		-- "CoreLibrary",
		-- "g3log",
		-- "PocoFoundationLite",
		-- "advapi32",
		-- "IPHLPAPI",
		-- "openexr",
		-- "zlib",
		-- "zstd_static",
		"cudart_static",
		"cuda",
		"nvrtc",
		-- "stb",
		-- "cs_signal",
		"gdt",
    }
	
	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	
	filter { "system:windows"}
		systemversion "latest"
--		disablewarnings { 
--			"5030", "4244", "4267", "4667", "4018", "4101", "4305", "4316",
--		} 
		defines 
		{ 
--			"POCO_OS_FAMILY_WINDOWS",
--			"NOMINMAX",
			"ALALBA_PLATFORM_WINDOWS"
		}
		
	filter "configurations:Debug"
		defines { "ALALBA_DEBUG" }
		runtime "Debug"
		symbols "On"
		libdirs { --THIRD_PARTY_DIR .. "builds/bin/" .. outputdir .. "/**",
				  --THIRD_PARTY_DIR .. "precompiled/bin/" .. outputdir .. "/**",
				  CUDA_LIB_DIR
		}
		
	filter "configurations:Release"
		defines "ALALBA_RELEASE"
		runtime "Release"
		optimize "On"
		libdirs { --THIRD_PARTY_DIR .. "builds/bin/" .. outputdir .. "/**",
				  --THIRD_PARTY_DIR .. "precompiled/bin/" .. outputdir .. "/**",
				  CUDA_LIB_DIR
		}
