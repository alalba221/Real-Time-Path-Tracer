#pragma once
#include "Alalba/Renderer/Shader.h"
#include <glad/glad.h>
namespace Alalba {

	class ALALBA_API OpenGLShader : public Shader
	{
	public:
		OpenGLShader(const std::string& filepath);

		virtual void Bind() override;
	private:
		void ReadShaderFromFile(const std::string& filepath);
		void CompileAndUploadShader();
		static GLenum ShaderTypeFromString(const std::string& type);

	private:
		RendererID m_RendererID;

		std::string m_ShaderSource;
	};

}
