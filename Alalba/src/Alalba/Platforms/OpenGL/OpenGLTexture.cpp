#include "alalbapch.h"
#include "OpenGLTexture.h"

#include "Alalba/Renderer/RendererAPI.h"
#include "Alalba/Renderer/Renderer.h"

#include <glad/glad.h>
#include "stb_image.h"

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

	static int CalculateMipMapCount(int width, int height)
	{
		int levels = 1;
		while ((width | height) >> levels) {
			levels++;
		}
		return levels;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// Texture2D
	//////////////////////////////////////////////////////////////////////////////////

	OpenGLTexture2D::OpenGLTexture2D(TextureFormat format, unsigned int width, unsigned int height)
		: m_Format(format), m_Width(width), m_Height(height)
	{
		auto self = this;
		ALALBA_RENDER_1(self, {
			glGenTextures(1, &self->m_RendererID);
			glBindTexture(GL_TEXTURE_2D, self->m_RendererID);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTextureParameterf(self->m_RendererID, GL_TEXTURE_MAX_ANISOTROPY, RendererAPI::GetCapabilities().MaxAnisotropy);


			glTexImage2D(GL_TEXTURE_2D, 0, AlalbaToOpenGLTextureFormat(self->m_Format), self->m_Width, self->m_Height, 0, AlalbaToOpenGLTextureFormat(self->m_Format), GL_UNSIGNED_BYTE, nullptr);
			glGenerateMipmap(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, 0);
		});
	}

	OpenGLTexture2D::OpenGLTexture2D(const std::string& path, bool srgb)
		: m_FilePath(path)
	{
		int width, height, channels;
		ALALBA_CORE_INFO("Loading texture {0}, srgb={1}", path, srgb);
		m_ImageData = stbi_load(path.c_str(), &width, &height, &channels, srgb ? STBI_rgb : STBI_rgb_alpha);

		m_Width = width;
		m_Height = height;
		m_Format = TextureFormat::RGBA;

		ALALBA_RENDER_S1(srgb, {
			// TODO: Consolidate properly
			if (srgb)
			{	// DSA version, not sure if it works
				glCreateTextures(GL_TEXTURE_2D, 1, &self->m_RendererID);
				int levels = CalculateMipMapCount(self->m_Width, self->m_Height);
				ALALBA_CORE_INFO("Creating srgb texture width {0} mips", levels);
				glTextureStorage2D(self->m_RendererID, levels, GL_SRGB8, self->m_Width, self->m_Height);
				glTextureParameteri(self->m_RendererID, GL_TEXTURE_MIN_FILTER, levels > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
				glTextureParameteri(self->m_RendererID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

				glTextureSubImage2D(self->m_RendererID, 0, 0, 0, self->m_Width, self->m_Height, GL_RGB, GL_UNSIGNED_BYTE, self->m_ImageData);
				glGenerateTextureMipmap(self->m_RendererID);
			}
			else
			{
				// comment out DSA version, since it doesn't work
				// 1
				glGenTextures(1, &self->m_RendererID);
				glBindTexture(GL_TEXTURE_2D, self->m_RendererID);
				//glCreateTextures(GL_TEXTURE_2D, 1, &self->m_RendererID);

				// 2
				glTexImage2D(GL_TEXTURE_2D, 0, AlalbaToOpenGLTextureFormat(self->m_Format), self->m_Width, self->m_Height, 0, srgb ? GL_SRGB8 : AlalbaToOpenGLTextureFormat(self->m_Format), GL_UNSIGNED_BYTE, self->m_ImageData);
				//glTextureStorage2D(self->m_RendererID, 0, AlalbaToOpenGLTextureFormat(self->m_Format), self->m_Width, self->m_Height);

				// 3
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				//glTextureParameteri(self->m_RendererID, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				//glTextureParameteri(self->m_RendererID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				//glTextureParameteri(self->m_RendererID, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				//glTextureParameteri(self->m_RendererID, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

				// 5
				glGenerateMipmap(GL_TEXTURE_2D);
				//glGenerateTextureMipmap(self->m_RendererID);
				// 6
				glBindTexture(GL_TEXTURE_2D, 0);
				//glBindTextureUnit(GL_TEXTURE_2D, 0);
				
			}
			stbi_image_free(self->m_ImageData);
			});
	}
	void OpenGLTexture2D::Bind(unsigned int slot) const
	{
		ALALBA_RENDER_S1(slot, {
			glActiveTexture(GL_TEXTURE0 + slot);
			glBindTexture(GL_TEXTURE_2D, self->m_RendererID);
			});
	}

	OpenGLTexture2D::~OpenGLTexture2D()
	{
		auto self = this;
		ALALBA_RENDER_1(self, {
			glDeleteTextures(1, &self->m_RendererID);
		});
	}

	void OpenGLTexture2D::ReloadFromMemory(unsigned char* data)
	{
		auto self = this;
		ALALBA_RENDER_2(self, data, {
			if (self->m_RendererID)
			{
				glDeleteTextures(1, &self->m_RendererID);
				self->m_RendererID = 0;
			}
			glGenTextures(1, &self->m_RendererID);
			glBindTexture(GL_TEXTURE_2D, self->m_RendererID);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			//glTextureParameterf(self->m_RendererID, GL_TEXTURE_MAX_ANISOTROPY, RendererAPI::GetCapabilities().MaxAnisotropy);

			//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self->m_Width, self->m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self->m_ImageData);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self->m_Width, self->m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
			//glGenerateMipmap(GL_TEXTURE_2D);
			//glBindTexture(GL_TEXTURE_2D, 0);
										});
	}

	//////////////////////////////////////////////////////////////////////////////////
	// TextureCube
	//////////////////////////////////////////////////////////////////////////////////

	OpenGLTextureCube::OpenGLTextureCube(const std::string& path)
		: m_FilePath(path)
	{
		int width, height, channels;
		stbi_set_flip_vertically_on_load(false);
		m_ImageData = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb);

		m_Width = width;
		m_Height = height;
		m_Format = TextureFormat::RGB;

		unsigned int faceWidth = m_Width / 4;
		unsigned int faceHeight = m_Height / 3;
		ALALBA_CORE_ASSERT(faceWidth == faceHeight, "Non-square faces!");

		std::array<unsigned char*, 6> faces;
		for (size_t i = 0; i < faces.size(); i++)
			faces[i] = new unsigned char[faceWidth * faceHeight * 3]; // 3 BPP

		int faceIndex = 0;

		for (size_t i = 0; i < 4; i++)
		{
			for (size_t y = 0; y < faceHeight; y++)
			{
				size_t yOffset = y + faceHeight;
				for (size_t x = 0; x < faceWidth; x++)
				{
					size_t xOffset = x + i * faceWidth;
					faces[faceIndex][(x + y * faceWidth) * 3 + 0] = m_ImageData[(xOffset + yOffset * m_Width) * 3 + 0];
					faces[faceIndex][(x + y * faceWidth) * 3 + 1] = m_ImageData[(xOffset + yOffset * m_Width) * 3 + 1];
					faces[faceIndex][(x + y * faceWidth) * 3 + 2] = m_ImageData[(xOffset + yOffset * m_Width) * 3 + 2];
				}
			}
			faceIndex++;
		}

		for (size_t i = 0; i < 3; i++)
		{
			// Skip the middle one
			if (i == 1)
				continue;

			for (size_t y = 0; y < faceHeight; y++)
			{
				size_t yOffset = y + i * faceHeight;
				for (size_t x = 0; x < faceWidth; x++)
				{
					size_t xOffset = x + faceWidth;
					faces[faceIndex][(x + y * faceWidth) * 3 + 0] = m_ImageData[(xOffset + yOffset * m_Width) * 3 + 0];
					faces[faceIndex][(x + y * faceWidth) * 3 + 1] = m_ImageData[(xOffset + yOffset * m_Width) * 3 + 1];
					faces[faceIndex][(x + y * faceWidth) * 3 + 2] = m_ImageData[(xOffset + yOffset * m_Width) * 3 + 2];
				}
			}
			faceIndex++;
		}

		ALALBA_RENDER_S3(faces, faceWidth, faceHeight, {
			glGenTextures(1, &self->m_RendererID);
			glBindTexture(GL_TEXTURE_CUBE_MAP, self->m_RendererID);

			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTextureParameterf(self->m_RendererID, GL_TEXTURE_MAX_ANISOTROPY, RendererAPI::GetCapabilities().MaxAnisotropy);

			auto format = AlalbaToOpenGLTextureFormat(self->m_Format);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, format, faceWidth, faceHeight, 0, format, GL_UNSIGNED_BYTE, faces[2]);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, format, faceWidth, faceHeight, 0, format, GL_UNSIGNED_BYTE, faces[0]);

			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, format, faceWidth, faceHeight, 0, format, GL_UNSIGNED_BYTE, faces[4]);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, format, faceWidth, faceHeight, 0, format, GL_UNSIGNED_BYTE, faces[5]);

			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, format, faceWidth, faceHeight, 0, format, GL_UNSIGNED_BYTE, faces[1]);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, format, faceWidth, faceHeight, 0, format, GL_UNSIGNED_BYTE, faces[3]);

			glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

			glBindTexture(GL_TEXTURE_2D, 0);

			for (size_t i = 0; i < faces.size(); i++)
				delete[] faces[i];

			stbi_image_free(self->m_ImageData);
			});
	}

	OpenGLTextureCube::~OpenGLTextureCube()
	{
		auto self = this;
		ALALBA_RENDER_1(self, {
			glDeleteTextures(1, &self->m_RendererID);
			});
	}

	void OpenGLTextureCube::Bind(unsigned int slot) const
	{
		ALALBA_RENDER_S1(slot, {
			glActiveTexture(GL_TEXTURE0 + slot);
			glBindTexture(GL_TEXTURE_CUBE_MAP, self->m_RendererID);
			});
	}

}
