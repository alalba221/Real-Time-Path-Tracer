#pragma once

namespace Alalba {

	using RendererID = uint32_t;

	enum class RendererAPIType
	{
		None,
		OpenGL
	};
	
	struct RenderAPICapabilities
	{
		std::string Vendor;
		std::string Renderer;
		std::string Version;

		int MaxSamples;
		float MaxAnisotropy;
	};

	class RendererAPI
	{
	private:

	public:
		static void Init();
		static void Shutdown();

		static void Clear(float r, float g, float b, float a);
		static void SetClearColor(float r, float g, float b, float a);

		static void DrawIndexed(unsigned int count, bool depthTest = true);
		static RendererAPIType Current() { return s_CurrentRendererAPI; }
		static RenderAPICapabilities& GetCapabilities()
		{
			static RenderAPICapabilities capabilities;
			return capabilities;
		}
 
	private:
		static RendererAPIType s_CurrentRendererAPI;

	};

}
