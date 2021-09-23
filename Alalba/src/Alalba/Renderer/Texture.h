#pragma once

#include "Alalba/Core/Base.h"

namespace Alalba {

	enum class ALALBA_API TextureFormat
	{
		None = 0,
		RGB = 1,
		RGBA = 2,
	};

	class ALALBA_API Texture
	{
	public:
		virtual ~Texture() {}
	};

	class ALALBA_API Texture2D : public Texture
	{
	public:
		static Texture2D* Create(TextureFormat format, unsigned int width, unsigned int height);

		virtual TextureFormat GetFormat() const = 0;
		virtual unsigned int GetWidth() const = 0;
		virtual unsigned int GetHeight() const = 0;
	};

}
