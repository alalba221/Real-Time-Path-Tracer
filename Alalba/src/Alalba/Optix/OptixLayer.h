#pragma once
#include "Alalba/Core/Layer.h"

namespace Alalba {
	class ALALBA_API OptixLayer : public Layer
	{
	public:
		OptixLayer();
		OptixLayer(const std::string& name);
		virtual ~OptixLayer();

		//void Begin();
		//void End();

		virtual void OnAttach() override;
		virtual void OnDetach() override;
		virtual void OnImGuiRender() override;
	private:
		float m_Time = 0.0f;
	};
}