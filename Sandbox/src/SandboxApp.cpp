#include "Alalba.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

static void ImGuiShowHelpMarker(const char* desc)
{
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}


class EditorLayer : public Alalba::Layer
{
public:
	EditorLayer()
		: Layer("Example"),
		m_ClearColor{ 0.2f, 0.3f, 0.8f, 1.0f }, 
		m_TriangleColor{ 0.8f, 0.2f, 0.3f, 1.0f },
		m_Camera(glm::perspectiveFov(glm::radians(45.0f), 1280.0f, 720.0f, 0.1f, 10000.0f))
	{
	}
	virtual void OnAttach () override
	{
	/*	static float vertices[] = {
			 -0.500000,  -0.500000,	 0.500000,0,0,
			  0.500000,  -0.500000,	 0.500000,0,0,
			 -0.500000,		0.500000,	 0.500000,0,0,
			  0.500000,   0.500000,	 0.500000,0,0,
			 -0.500000,   0.500000,	-0.500000,0,0,
			  0.500000,   0.500000, -0.500000,0,0,
			 -0.500000,  -0.500000, -0.500000,0,0,
			  0.500000,  -0.500000, -0.500000,0,0 
		};

		static unsigned int indices[] = {
			 0, 1, 2,
			 2, 1, 3,
			 2, 3, 4,
			 4, 3, 5,
			 4, 5, 6,
			 6, 5, 7,
			 6, 7, 0,
			 0, 7, 1,
			 1, 7, 3,
			 3, 7, 5,
			 6, 0, 4,
			 4, 0, 2
		};
		m_VB = std::unique_ptr<Alalba::VertexBuffer>(Alalba::VertexBuffer::Create());
		m_VB->SetData(vertices, sizeof(vertices));

		m_IB = std::unique_ptr<Alalba::IndexBuffer>(Alalba::IndexBuffer::Create());
		m_IB->SetData(indices, sizeof(indices));*/

		m_BRDFLUT.reset(Alalba::Texture2D::Create("assets/textures/cerberus/cerberus_A.png"));
		m_Mesh.reset(new Alalba::Mesh("assets/meshes/cerberus.fbx"));
		m_SimplePBRShader.reset(Alalba::Shader::Create("assets/shaders/shader.glsl"));
	}
	void OnUpdate() override
	{
		m_Camera.Update();
		auto viewProjection = m_Camera.GetProjectionMatrix() * m_Camera.GetViewMatrix();

		Alalba::Renderer::Clear(m_ClearColor[0], m_ClearColor[1], m_ClearColor[2], m_ClearColor[3]);
	
		Alalba::UniformBufferDeclaration<sizeof(glm::mat4) * 2+sizeof(glm::vec4) *1, 3> simplePbrShaderUB;
		simplePbrShaderUB.Push("u_Color", m_TriangleColor);
		simplePbrShaderUB.Push("u_ViewProjectionMatrix", viewProjection);
		simplePbrShaderUB.Push("u_ModelMatrix", glm::mat4(1.0f));

		m_SimplePBRShader->UploadUniformBuffer(simplePbrShaderUB);
		
		//m_VB->Bind();
		m_SimplePBRShader->Bind();
		m_BRDFLUT->Bind(0);
		/*m_IB->Bind();*/
		//Alalba::Renderer::DrawIndexed(36);
		m_Mesh->Render();
	}

	virtual void OnImGuiRender() override
	{
		static bool show_demo_window = true;
		if (show_demo_window)
			ImGui::ShowDemoWindow(&show_demo_window);

		ImGui::Begin("GameLayer");
		ImGui::ColorEdit4("Clear Color", m_ClearColor);
		ImGui::ColorEdit4("Triangle Color", glm::value_ptr(m_TriangleColor));
		ImGui::End();

	}

	void OnEvent(Alalba::Event& event) override
	{
		
		if (event.GetEventType() == Alalba::EventType::KeyPressed)
		{
			Alalba::KeyPressedEvent& e = (Alalba::KeyPressedEvent&)event;
			if (e.GetKeyCode() == ALALBA_TAB)
					ALALBA_APP_TRACE("Tab key is pressed (event)!");
			//	ALALBA_APP_TRACE("{0}", (char)e.GetKeyCode());
		}
	}
	private: 
		
		Alalba::Camera m_Camera;
		float m_ClearColor[4];
		std::unique_ptr<Alalba::VertexBuffer> m_VB;
		std::unique_ptr<Alalba::IndexBuffer> m_IB;
		std::unique_ptr<Alalba::Shader> m_SimplePBRShader;
		glm::vec4 m_TriangleColor;

		std::unique_ptr<Alalba::Texture2D> m_BRDFLUT;
		std::unique_ptr<Alalba::Mesh> m_Mesh;

};

class Sandbox : public Alalba::Application{
public:
  Sandbox()
  {
    PushLayer(new EditorLayer());
  };
  ~Sandbox(){};
};
Alalba::Application* Alalba::CreateApplication(){
  return new Sandbox();
}