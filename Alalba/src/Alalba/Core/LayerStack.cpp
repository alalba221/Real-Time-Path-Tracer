#include "alalbapch.h"
#include "LayerStack.h"
namespace Alalba {

	LayerStack::LayerStack()
	{
		
	}

	LayerStack::~LayerStack()
	{
		for (Layer* layer : m_Layers)
		{
			delete layer;
		}
	}

	void LayerStack::PushLayer(Layer* layer)
	{
		// insert: copies or moves the elements into the container 
    // by calling copy constructor or move constructor.
		
    // emplace elements are constructed in-place, 
    // i.e. no copy or move operations are performed.
		m_Layers.emplace(m_Layers.begin()+m_LayerInsertIndex, layer);
		m_LayerInsertIndex++;
	}
	void LayerStack::PushOverlayer(Layer* overlay)
	{
		m_Layers.emplace_back(overlay);
	}

	void LayerStack::PopLayer(Layer* layer)
	{
		auto it = std::find(m_Layers.begin(), m_Layers.end(), layer);
		if (it != m_Layers.end())
		{
			m_Layers.erase(it);
			m_LayerInsertIndex--;
		}
	}
	void LayerStack::PopOverlayer(Layer* overlay)
	{
		auto it = std::find(m_Layers.begin(), m_Layers.end(), overlay);
		if (it != m_Layers.end())
		{
			m_Layers.erase(it);
		}
	}
}