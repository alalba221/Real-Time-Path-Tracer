#include "alalbapch.h"
#include "OpenGLTexture.h"

#include "Alalba/Renderer/RendererAPI.h"
#include "Alalba/Renderer/Renderer.h"

#include <glad/glad.h>

namespace Alalba {

	static GLenum AlalbaToOpenGLTextureFormat(TextureFormat format)
	{
		switch (format)
		{
			case Alalba::TextureFormat::RGB:     return GL_RGB;
			case Alalba::TextureFormat::RGBA:    return GL_RGBA;
		}
		return 0;
	}

	OpenGLTexture2D::OpenGLTexture2D(TextureFormat format, unsigned int width, unsigned int height)
		: m_Format(format), m_Width(width), m_Height(height)
	{
		auto self = this;
		ALALBA_RENDER_1(self, {
			glGenTextures(1, &self->m_RendererID);
			glBindTexture(GL_TEXTURE_2D, self->m_RendererID);
			glTexImage2D(GL_TEXTURE_2D, 0, AlalbaToOpenGLTextureFormat(self->m_Format), self->m_Width, self->m_Height, 0, AlalbaToOpenGLTextureFormat(self->m_Format), GL_UNSIGNED_BYTE, nullptr);
			glBindTexture(GL_TEXTURE_2D, 0);
		});
	}

	OpenGLTexture2D::~OpenGLTexture2D()
	{
		auto self = this;
		ALALBA_RENDER_1(self, {
			glDeleteTextures(1, &self->m_RendererID);
		});
	}

}
