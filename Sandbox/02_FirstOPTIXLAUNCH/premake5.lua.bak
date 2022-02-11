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
		"%{IncludeDir.glm}"

	}
	-- Directories relatice to project folder
	postbuildcommands
    {
			--("{COPY} %{cfg.buildtarget.relpath} ../bin/"..outputdir.."/Sandbox")
			("{COPY} assets  %{wks.location}/bin/" ..outputdir .. "/%{prj.name}")
    }
			
-- add settings common to all project
dofile("../optix.lua")