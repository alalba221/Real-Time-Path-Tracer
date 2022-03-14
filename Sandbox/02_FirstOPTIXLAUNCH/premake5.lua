project "02_FirstOPTIXLAUNCH"
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
	--  SOURCE_DIR .. "**.cu"
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
		"%{IncludeDir.glm}",
		"%{IncludeDir.eigen}",

		--"%{wks.location}/Alalba/vendor",

	}
	-- Directories relatice to project folder
	postbuildcommands
    {
			--("{COPY} %{cfg.buildtarget.relpath} ../bin/"..outputdir.."/Sandbox")
			--("{COPY} assets  %{wks.location}/bin/" ..outputdir .. "/%{prj.name}")
    }


-- add settings common to all project

os.execute('nvcc assets\\deviceCode.cu -ptx -ccbin \z
			"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.30.30705\\bin\\Hostx64\\x64" \z 
			--gpu-architecture=compute_52 \z
			--use_fast_math \z
			--relocatable-device-code=true \z
			--expt-relaxed-constexpr \z
			-I"C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 7.4.0\\include" \z
			-I"C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 7.4.0\\SDK" \z
			-I"C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 7.4.0\\SDK\\cuda" \z
			-I"D:\\Study\\Alalba\\Alalba\\src\\Alalba\\Optix" \z
			-I"D:\\Study\\Alalba\\Alalba\\vendor\\gdt\\source\\math" \z
			-I"D:\\Study\\Alalba\\Alalba\\vendor\\gdt\\source" \z
			-I"D:\\Study\\Alalba\\Alalba\\src" \z
			-I"D:\\Study\\Alalba\\Alalba\\vendor\\eigen" \z
			-o assets\\deviceCode.ptx')
--os.execute('nvcc --version')
dofile("../optix.lua")