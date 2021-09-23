#include "alalbapch.h"
#include "Texture.h"

#include "Alalba/Renderer/RendererAPI.h"
#include "Alalba/Platforms/OpenGL/OpenGLTexture.h"

namespace Alalba {

	Texture2D* Texture2D::Create(TextureFormat format, unsigned int width, unsigned int height)
	{
		switch (RendererAPI::Current())
		{
			case RendererAPIType::None: return nullptr;
			case RendererAPIType::OpenGL: return new OpenGLTexture2D(format, width, height);
		}
		return nullptr;
	}

}
