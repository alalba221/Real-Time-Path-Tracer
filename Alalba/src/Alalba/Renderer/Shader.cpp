#include "alalbapch.h"
#include "Shader.h"

#include "Alalba/Platforms/OpenGL/OpenGLShader.h"

namespace Alalba {

	Shader* Shader::Create(const std::string& filepath)
	{
		switch (RendererAPI::Current())
		{
			case RendererAPIType::None: return nullptr;
			case RendererAPIType::OpenGL: return new OpenGLShader(filepath);
		}
		return nullptr;
	}

}
