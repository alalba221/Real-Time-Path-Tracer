#include "alalbapch.h"
#include "Alalba/Platforms/OpenGL/OpenGLVertexBuffer.h"
#include "VertexBuffer.h"
namespace Alalba {

	VertexBuffer* VertexBuffer::Create(unsigned int size)
	{
		switch (RendererAPI::Current())
		{
			case RendererAPIType::None:    return nullptr;
			case RendererAPIType::OpenGL:  return new OpenGLVertexBuffer(size);
		}
		return nullptr;

	}

}
