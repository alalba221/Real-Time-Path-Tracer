#include "alalbapch.h"

#include "Alalba/Platforms/OpenGL/OpenGLIndexBuffer.h"

namespace Alalba {

	IndexBuffer* IndexBuffer::Create(unsigned int size)
	{
		switch (RendererAPI::Current())
		{
			case RendererAPIType::None:    return nullptr;
			case RendererAPIType::OpenGL:  return new OpenGLIndexBuffer(size);
		}
		return nullptr;

	}

}
