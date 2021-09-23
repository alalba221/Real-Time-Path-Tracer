#pragma once
#include "Alalba/Core/Base.h"
#include "Alalba/Renderer/Renderer.h"

#include <string>

namespace Alalba
{
	struct ALALBA_API ShaderUniform
	{
		
	};

	struct ALALBA_API ShaderUniformCollection
	{

	};

	class ALALBA_API Shader
	{
	public:
		virtual void Bind() = 0;
		

		// Represents a complete shader program stored in a single file.
		// Note: currently for simplicity this is simply a string filepath, however
		//       in the future this will be an asset object + metadata
		static Shader* Create(const std::string& filepath);
	};

}
