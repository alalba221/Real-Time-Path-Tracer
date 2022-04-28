#include "Alalba.h"
#include "stb/include/stb_image_write.h"
#include "Alalba/Core/Application.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace Alalba;
using namespace gdt;
using namespace glm;

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

class OptixLayer : public Alalba::Layer 
{
public:
  OptixLayer(const Model* model)
    :Layer("OptixLayer"),
    m_Camera(glm::vec3(278, 278, -600), glm::vec3(278, 278, 0), glm::vec3(0.f, 1.f, 0.f)),
    sample(model)
  {}
  virtual ~OptixLayer() {}
  virtual void OnAttach() override
  {
    // optix
    fbSize = vec2i(1280.0f * 2, 720.0f * 2);
    sample.resize(fbSize);
    pixels.resize(1280 * 720 * 2 * 2);
    sample.render();
    sample.downloadPixels(pixels.data());
    // opengl
		//m_FrameBuffer.reset(Alalba::FrameBuffer::Create(1280 * 2, 720 * 2, Alalba::FrameBufferFormat::RGBA16F));
    m_FinalPresentBuffer.reset(Alalba::FrameBuffer::Create(1280 * 2, 720 * 2, Alalba::FrameBufferFormat::RGBA16F));
    m_Shader.reset(Alalba::Shader::Create("assets/shaders/shader.glsl"));
    // Create Quad
    float x = -1, y = -1;
    float width = 2, height = 2;
    struct QuadVertex
    {
      glm::vec3 Position;
      glm::vec2 TexCoord;
    };
    QuadVertex* data = new QuadVertex[4];
    data[0].Position = glm::vec3(x, y, 0);
    data[0].TexCoord = glm::vec2(0, 0);

    data[1].Position = glm::vec3(x + width, y, 0);
    data[1].TexCoord = glm::vec2(1, 0);

    data[2].Position = glm::vec3(x + width, y + height, 0);
    data[2].TexCoord = glm::vec2(1, 1);

    data[3].Position = glm::vec3(x, y + height, 0);
    data[3].TexCoord = glm::vec2(0, 1);

    m_VertexBuffer.reset(Alalba::VertexBuffer::Create());
    m_VertexBuffer->SetData(data, 4 * sizeof(QuadVertex));

    uint32_t* indices = new uint32_t[6]{ 0, 1, 2, 2, 3, 0, };
    m_IndexBuffer.reset(Alalba::IndexBuffer::Create());
    m_IndexBuffer->SetData(indices, 6 * sizeof(unsigned int));

    // create an empty texture
    m_OptixResult.reset(Alalba::Texture2D::Create(TextureFormat::RGBA, 1280 * 2, 720 * 2));
    

  }
  virtual void OnDetach() override
  {
    //dump
  }
  void OnUpdate() override
  {
    /// Cam test
    m_Camera.Update();
    if (m_Camera.m_Changed)
    {
      sample.setCamera(m_Camera);
      m_Camera.m_Changed = false;
    }
    
    // update render data
    sample.render();
    pixels.clear();
    
		pixels.resize(fbSize.x * fbSize.y);
    sample.downloadPixels(pixels.data());

   // 
		//m_FinalPresentBuffer->Bind();
    Alalba::Renderer::Clear(0.2f, 0.3f, 0.8f, 1.0f);
    m_OptixResult->ReloadFromMemory((unsigned char*)pixels.data(), 1280 * 2, 720 * 2);
    m_Shader->Bind();
    m_OptixResult->Bind(0);
    m_VertexBuffer->Bind();
    m_IndexBuffer->Bind();
    Alalba::Renderer::DrawIndexed(m_IndexBuffer->GetCount(), false);
		//m_FinalPresentBuffer->Unbind();

  }
  enum class PropertyFlag
  {
    None = 0, ColorProperty = 1
  };

  void Property(const std::string& name, bool& value)
  {
    ImGui::Text(name.c_str());
    ImGui::NextColumn();
    ImGui::PushItemWidth(-1);

    std::string id = "##" + name;
    ImGui::Checkbox(id.c_str(), &value);

    ImGui::PopItemWidth();
    ImGui::NextColumn();
  }

  void Property(const std::string& name, float& value, float min = -1.0f, float max = 1.0f, PropertyFlag flags = PropertyFlag::None)
  {
    ImGui::Text(name.c_str());
    ImGui::NextColumn();
    ImGui::PushItemWidth(-1);

    std::string id = "##" + name;
    ImGui::SliderFloat(id.c_str(), &value, min, max);

    ImGui::PopItemWidth();
    ImGui::NextColumn();
  }

  void Property(const std::string& name, glm::vec3& value, PropertyFlag flags)
  {
    Property(name, value, -1.0f, 1.0f, flags);
  }

  void Property(const std::string& name, glm::vec3& value, float min = -1.0f, float max = 1.0f, PropertyFlag flags = PropertyFlag::None)
  {
    ImGui::Text(name.c_str());
    ImGui::NextColumn();
    ImGui::PushItemWidth(-1);

    std::string id = "##" + name;
    if ((int)flags & (int)PropertyFlag::ColorProperty)
      ImGui::ColorEdit3(id.c_str(), glm::value_ptr(value), ImGuiColorEditFlags_NoInputs);
    else
      ImGui::SliderFloat3(id.c_str(), glm::value_ptr(value), min, max);

    ImGui::PopItemWidth();
    ImGui::NextColumn();
  }

  void Property(const std::string& name, glm::vec4& value, PropertyFlag flags)
  {
    Property(name, value, -1.0f, 1.0f, flags);
  }

  void Property(const std::string& name, glm::vec4& value, float min = -1.0f, float max = 1.0f, PropertyFlag flags = PropertyFlag::None)
  {
    ImGui::Text(name.c_str());
    ImGui::NextColumn();
    ImGui::PushItemWidth(-1);

    std::string id = "##" + name;
    if ((int)flags & (int)PropertyFlag::ColorProperty)
      ImGui::ColorEdit4(id.c_str(), glm::value_ptr(value), ImGuiColorEditFlags_NoInputs);
    else
      ImGui::SliderFloat4(id.c_str(), glm::value_ptr(value), min, max);

    ImGui::PopItemWidth();
    ImGui::NextColumn();
  }
  virtual void OnImGuiRender() override
  {		//ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
		//static bool p_open = true;

		//static bool opt_fullscreen_persistant = true;
		//static bool opt_padding = false;
		//static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
		//bool opt_fullscreen = opt_fullscreen_persistant;

		//// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
		//// because it would be confusing to have two docking targets within each others.
		//ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
		//if (opt_fullscreen)
		//{
		//	ImGuiViewport* viewport = ImGui::GetMainViewport();
		//	ImGui::SetNextWindowPos(viewport->Pos);
		//	ImGui::SetNextWindowSize(viewport->Size);
		//	ImGui::SetNextWindowViewport(viewport->ID);
		//	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		//	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		//	window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
		//	window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
		//}

		//// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
		//// and handle the pass-thru hole, so we ask Begin() to not render a background.
		//if (ImGuiDockNodeFlags_PassthruCentralNode)
		//	window_flags |= ImGuiWindowFlags_NoBackground;

		//ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		//ImGui::Begin("DockSpace Demo", &p_open, window_flags);
		//ImGui::PopStyleVar();

		//if (opt_fullscreen)
		//	ImGui::PopStyleVar(2);

		//// Dockspace
		//ImGuiIO& io = ImGui::GetIO();
		//if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
		//{
		//	ImGuiID dockspace_id = ImGui::GetID("MyDockspace");
		//	ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
		//}
	//	// Editor Panel ------------------------------------------------------------------------------

	//	ImGui::Begin("Model");
	//	ImGui::RadioButton("Spheres", (int*)&m_Scene, (int)Scene::Spheres);
	//	ImGui::SameLine();
	//	ImGui::RadioButton("Model", (int*)&m_Scene, (int)Scene::Model);
	//	ImGui::ColorEdit4("Clear Color", m_ClearColor);

	//	ImGui::Begin("Environment");
	//	ImGui::Columns(2);
	//	ImGui::AlignTextToFramePadding();

	//	Property("Light Direction", m_Light.Direction);
	//	Property("Light Radiance", m_Light.Radiance, PropertyFlag::ColorProperty);
	//	Property("Light Position", m_Light.Position,-100,100);
	//	Property("Light Color", m_Light.Color, PropertyFlag::ColorProperty);
	//	Property("Light Multiplier", m_LightMultiplier, 0.0f, 5.0f);
	//	Property("Exposure", m_Exposure, 0.0f, 5.0f);

	//	//Property("Radiance Prefiltering", m_RadiancePrefilter);
	//	//Property("Env Map Rotation", m_EnvMapRotation, -360.0f, 360.0f);

	//	ImGui::Columns(1);


	//	auto cameraForward = m_Camera.GetForwardDirection();
	//	ImGui::Text("Camera Forward: %.2f, %.2f, %.2f", cameraForward.x, cameraForward.y, cameraForward.z);
	//	auto cameraPosition = m_Camera.GetPosition();
	//	ImGui::Text("Camera Forward: %.2f, %.2f, %.2f", cameraPosition.x, cameraPosition.y, cameraPosition.z);
	//	
	//	ImGui::End();
	//	ImGui::Separator();
	//	{
	//		ImGui::Text("Mesh");
	//		std::string fullpath = m_Mesh ? m_Mesh->GetFilePath() : "None";
	//		size_t found = fullpath.find_last_of("/\\");
	//		std::string path = found != std::string::npos ? fullpath.substr(found + 1) : fullpath;
	//		ImGui::Text(path.c_str()); ImGui::SameLine();
	//		if (ImGui::Button("...##Mesh"))
	//		{
	//			std::string filename = Alalba::Application::Get().OpenFile("");
	//			if (filename != "")
	//				m_Mesh.reset(new Alalba::Mesh(filename));
	//		}
	//	}
	//	ImGui::Separator();

	//	ImGui::Text("Shader Parameters");
	//	ImGui::Checkbox("Radiance Prefiltering", &m_RadiancePrefilter);
	//	ImGui::SliderFloat("Env Map Rotation", &m_EnvMapRotation, -360.0f, 360.0f);

	//	ImGui::Separator();

	//	// Textures ------------------------------------------------------------------------------
	//	{
	//		// Albedo
	//		if (ImGui::CollapsingHeader("Albedo", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
	//		{
	//			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 10));
	//			ImGui::Image(m_AlbedoInput.TextureMap ? (void*)m_AlbedoInput.TextureMap->GetRendererID() : (void*)m_CheckerboardTex->GetRendererID(), ImVec2(64, 64));
	//			ImGui::PopStyleVar();
	//			if (ImGui::IsItemHovered())
	//			{
	//				if (m_AlbedoInput.TextureMap)
	//				{
	//					ImGui::BeginTooltip();
	//					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
	//					ImGui::TextUnformatted(m_AlbedoInput.TextureMap->GetPath().c_str());
	//					ImGui::PopTextWrapPos();
	//					ImGui::Image((void*)m_AlbedoInput.TextureMap->GetRendererID(), ImVec2(384, 384));
	//					ImGui::EndTooltip();
	//				}
	//				if (ImGui::IsItemClicked())
	//				{
	//					std::string filename = Alalba::Application::Get().OpenFile("");
	//					if (filename != "")
	//						m_AlbedoInput.TextureMap.reset(Alalba::Texture2D::Create(filename, m_AlbedoInput.SRGB));
	//				}
	//			}
	//			ImGui::SameLine();
	//			ImGui::BeginGroup();
	//			ImGui::Checkbox("Use##AlbedoMap", &m_AlbedoInput.UseTexture);
	//			if (ImGui::Checkbox("sRGB##AlbedoMap", &m_AlbedoInput.SRGB))
	//			{
	//				if (m_AlbedoInput.TextureMap)
	//					m_AlbedoInput.TextureMap.reset(Alalba::Texture2D::Create(m_AlbedoInput.TextureMap->GetPath(), m_AlbedoInput.SRGB));
	//			}
	//			ImGui::EndGroup();
	//			ImGui::SameLine();
	//			ImGui::ColorEdit3("Color##Albedo", glm::value_ptr(m_AlbedoInput.Color), ImGuiColorEditFlags_NoInputs);
	//		}
	//	}
	//	{
	//		// Normals
	//		if (ImGui::CollapsingHeader("Normals", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
	//		{
	//			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 10));
	//			ImGui::Image(m_NormalInput.TextureMap ? (void*)m_NormalInput.TextureMap->GetRendererID() : (void*)m_CheckerboardTex->GetRendererID(), ImVec2(64, 64));
	//			ImGui::PopStyleVar();
	//			if (ImGui::IsItemHovered())
	//			{
	//				if (m_NormalInput.TextureMap)
	//				{
	//					ImGui::BeginTooltip();
	//					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
	//					ImGui::TextUnformatted(m_NormalInput.TextureMap->GetPath().c_str());
	//					ImGui::PopTextWrapPos();
	//					ImGui::Image((void*)m_NormalInput.TextureMap->GetRendererID(), ImVec2(384, 384));
	//					ImGui::EndTooltip();
	//				}
	//				if (ImGui::IsItemClicked())
	//				{
	//					std::string filename = Alalba::Application::Get().OpenFile("");
	//					if (filename != "")
	//						m_NormalInput.TextureMap.reset(Alalba::Texture2D::Create(filename));
	//				}
	//			}
	//			ImGui::SameLine();
	//			ImGui::Checkbox("Use##NormalMap", &m_NormalInput.UseTexture);
	//		}
	//	}
	//	{
	//		// Metalness
	//		if (ImGui::CollapsingHeader("Metalness", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
	//		{
	//			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 10));
	//			ImGui::Image(m_MetalnessInput.TextureMap ? (void*)m_MetalnessInput.TextureMap->GetRendererID() : (void*)m_CheckerboardTex->GetRendererID(), ImVec2(64, 64));
	//			ImGui::PopStyleVar();
	//			if (ImGui::IsItemHovered())
	//			{
	//				if (m_MetalnessInput.TextureMap)
	//				{
	//					ImGui::BeginTooltip();
	//					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
	//					ImGui::TextUnformatted(m_MetalnessInput.TextureMap->GetPath().c_str());
	//					ImGui::PopTextWrapPos();
	//					ImGui::Image((void*)m_MetalnessInput.TextureMap->GetRendererID(), ImVec2(384, 384));
	//					ImGui::EndTooltip();
	//				}
	//				if (ImGui::IsItemClicked())
	//				{
	//					std::string filename = Alalba::Application::Get().OpenFile("");
	//					if (filename != "")
	//						m_MetalnessInput.TextureMap.reset(Alalba::Texture2D::Create(filename));
	//				}
	//			}
	//			ImGui::SameLine();
	//			ImGui::Checkbox("Use##MetalnessMap", &m_MetalnessInput.UseTexture);
	//			ImGui::SameLine();
	//			ImGui::SliderFloat("Value##MetalnessInput", &m_MetalnessInput.Value, 0.0f, 1.0f);
	//		}
	//	}
	//	{
	//		// Roughness
	//		if (ImGui::CollapsingHeader("Roughness", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
	//		{
	//			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 10));
	//			ImGui::Image(m_RoughnessInput.TextureMap ? (void*)m_RoughnessInput.TextureMap->GetRendererID() : (void*)m_CheckerboardTex->GetRendererID(), ImVec2(64, 64));
	//			ImGui::PopStyleVar();
	//			if (ImGui::IsItemHovered())
	//			{
	//				if (m_RoughnessInput.TextureMap)
	//				{
	//					ImGui::BeginTooltip();
	//					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
	//					ImGui::TextUnformatted(m_RoughnessInput.TextureMap->GetPath().c_str());
	//					ImGui::PopTextWrapPos();
	//					ImGui::Image((void*)m_RoughnessInput.TextureMap->GetRendererID(), ImVec2(384, 384));
	//					ImGui::EndTooltip();
	//				}
	//				if (ImGui::IsItemClicked())
	//				{
	//					std::string filename = Alalba::Application::Get().OpenFile("");
	//					if (filename != "")
	//						m_RoughnessInput.TextureMap.reset(Alalba::Texture2D::Create(filename));
	//				}
	//			}
	//			ImGui::SameLine();
	//			ImGui::Checkbox("Use##RoughnessMap", &m_RoughnessInput.UseTexture);
	//			ImGui::SameLine();
	//			ImGui::SliderFloat("Value##RoughnessInput", &m_RoughnessInput.Value, 0.0f, 1.0f);
	//		}
	//	}

	//	{
	//		// ao
	//		if (ImGui::CollapsingHeader("ao", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
	//		{
	//			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 10));
	//			ImGui::Image(m_AOInput.TextureMap ? (void*)m_AOInput.TextureMap->GetRendererID() : (void*)m_CheckerboardTex->GetRendererID(), ImVec2(64, 64));
	//			ImGui::PopStyleVar();
	//			if (ImGui::IsItemHovered())
	//			{
	//				if (m_AOInput.TextureMap)
	//				{
	//					ImGui::BeginTooltip();
	//					ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
	//					ImGui::TextUnformatted(m_AOInput.TextureMap->GetPath().c_str());
	//					ImGui::PopTextWrapPos();
	//					ImGui::Image((void*)m_AOInput.TextureMap->GetRendererID(), ImVec2(384, 384));
	//					ImGui::EndTooltip();
	//				}
	//				if (ImGui::IsItemClicked())
	//				{
	//					std::string filename = Alalba::Application::Get().OpenFile("");
	//					if (filename != "")
	//						m_AOInput.TextureMap.reset(Alalba::Texture2D::Create(filename));
	//				}
	//			}
	//			ImGui::SameLine();
	//			ImGui::Checkbox("Use##AOMap", &m_AOInput.UseTexture);
	//			ImGui::SameLine();
	//			ImGui::SliderFloat("Value##AOInput", &m_AOInput.Value, 0.0f, 1.0f);
	//		}
	//	}

	//	ImGui::Separator();
	//	if (ImGui::TreeNode("Shaders"))
	//	{
	//		auto& shaders = Alalba::Shader::s_AllShaders;
	//		for (auto& shader : shaders)
	//		{
	//			if (ImGui::TreeNode(shader->GetName().c_str()))
	//			{
	//				std::string buttonName = "Reload##" + shader->GetName();
	//				if (ImGui::Button(buttonName.c_str()))
	//					shader->Reload();
	//				ImGui::TreePop();
	//			}
	//		}
	//		ImGui::TreePop();
	//	}
	//	ImGui::End();

		//ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		//ImGui::Begin("Viewport");
		//auto viewportSize = ImGui::GetContentRegionAvail();
		//// m_FrameBuffer->Resize((uint32_t)viewportSize.x, (uint32_t)viewportSize.y);
		//m_FinalPresentBuffer->Resize((uint32_t)viewportSize.x, (uint32_t)viewportSize.y);
		////m_Camera.SetProjectionMatrix(glm::perspectiveFov(glm::radians(45.0f), viewportSize.x, viewportSize.y, 0.1f, 10000.0f));
		//ImGui::Image((void*)m_FinalPresentBuffer->GetColorAttachmentRendererID(), viewportSize, {0, 1}, {1, 0});
		////ImGui::Image((void*)m_FinalPresentBuffer->GetColorAttachmentRendererID(), ImVec2(1280*2,720*2), { 0, 1 }, { 1, 0 });
		//ImGui::End();
		//ImGui::PopStyleVar();
		

		//if (ImGui::BeginMenuBar())
		//{
		//	if (ImGui::BeginMenu("Docking"))
		//	{
		//		// Disabling fullscreen would allow the window to be moved to the front of other windows, 
		//		// which we can't undo at the moment without finer window depth/z control.
		//		//ImGui::MenuItem("Fullscreen", NULL, &opt_fullscreen_persistant);

		//		if (ImGui::MenuItem("Flag: NoSplit", "", (dockspace_flags & ImGuiDockNodeFlags_NoSplit) != 0))                 dockspace_flags ^= ImGuiDockNodeFlags_NoSplit;
		//		if (ImGui::MenuItem("Flag: NoDockingInCentralNode", "", (dockspace_flags & ImGuiDockNodeFlags_NoDockingInCentralNode) != 0))  dockspace_flags ^= ImGuiDockNodeFlags_NoDockingInCentralNode;
		//		if (ImGui::MenuItem("Flag: NoResize", "", (dockspace_flags & ImGuiDockNodeFlags_NoResize) != 0))                dockspace_flags ^= ImGuiDockNodeFlags_NoResize;
		//		//if (ImGui::MenuItem("Flag: PassthruDockspace", "", (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) != 0))       dockspace_flags ^= ImGuiDockNodeFlags_PassthruCentralNode;
		//		if (ImGui::MenuItem("Flag: AutoHideTabBar", "", (dockspace_flags & ImGuiDockNodeFlags_AutoHideTabBar) != 0))          dockspace_flags ^= ImGuiDockNodeFlags_AutoHideTabBar;
		//		ImGui::Separator();
		//		if (ImGui::MenuItem("Close DockSpace", NULL, false, p_open != NULL))
		//			p_open = false;
		//		ImGui::EndMenu();
		//	}
		//	ImGuiShowHelpMarker(
		//		"You can _always_ dock _any_ window into another by holding the SHIFT key while moving a window. Try it now!" "\n"
		//		"This demo app has nothing to do with it!" "\n\n"
		//		"This demo app only demonstrate the use of ImGui::DockSpace() which allows you to manually create a docking node _within_ another window. This is useful so you can decorate your main application window (e.g. with a menu bar)." "\n\n"
		//		"ImGui::DockSpace() comes with one hard constraint: it needs to be submitted _before_ any window which may be docked into it. Therefore, if you use a dock spot as the central point of your application, you'll probably want it to be part of the very first window you are submitting to imgui every frame." "\n\n"
		//		"(NB: because of this constraint, the implicit \"Debug\" window can not be docked into an explicit DockSpace() node, because that window is submitted as part of the NewFrame() call. An easy workaround is that you can create your own implicit \"Debug##2\" window after calling DockSpace() and leave it in the window stack for anyone to use.)"
		//	);

		//	ImGui::EndMenuBar();
		//}

		//ImGui::End();
  }
  void OnEvent(Alalba::Event& event) override
  {
    // dump
  }
private:
  Alalba::Camera m_Camera;
  std::unique_ptr<Alalba::Texture2D> m_OptixResult;
  //std::unique_ptr<Alalba::SampleRenderer> m_sample;

  std::unique_ptr<Alalba::Shader> m_Shader;
  std::unique_ptr<Alalba::FrameBuffer> m_FrameBuffer, m_FinalPresentBuffer;

  std::unique_ptr<Alalba::VertexBuffer> m_VertexBuffer;
  std::unique_ptr<Alalba::IndexBuffer> m_IndexBuffer;
  SampleRenderer sample;

  vec2i fbSize;
  std::vector<vec4f> pixels;
};


class Sandbox02 : public Alalba::Application {

public:

	Sandbox02()
	{

    Lambertian* white = new Lambertian(gdt::vec3f(.73f, .73f, .73f));
    Lambertian* red   = new Lambertian(gdt::vec3f(.65f, .05f, .05f));
    Lambertian* green = new Lambertian(gdt::vec3f(.12f, .45f, .15f));
    Diffuse_light* light = new Diffuse_light(gdt::vec3f(15.f, 15.f, 15.f));
   
    Microfacet* shortbox= new Microfacet(0.1, 0.9, vec3f(.8f, .7f, .8f), vec3f(0.5f, 0.5f, 0.5f));
    Microfacet* tallbox = new Microfacet(0.1, 0.9, vec3f(.8f, .7f, .1f), vec3f(0.7f, 0.7f, 0.7f));
		Microfacet* bunny		=	new Microfacet(0.1, 0.9, vec3f(.1f, .7f, .8f), vec3f(0.5f, 0.5f, 0.5f));
		

    Model* model = new Model;
    
    model->MergeModel(loadOBJ("assets/cornellbox/floor.obj",white));
    model->MergeModel(loadOBJ("assets/cornellbox/left.obj", red));
    model->MergeModel(loadOBJ("assets/cornellbox/light.obj", light));
    model->MergeModel(loadOBJ("assets/cornellbox/right.obj", green));
    
    
    model->MergeModel(loadOBJ("assets/cornellbox/shortbox.obj", shortbox));
    model->MergeModel(loadOBJ("assets/cornellbox/tallbox.obj", tallbox));
    
    model->MergeModel(loadOBJ("assets/bunny/untitled.obj", bunny));
		PushLayer(new OptixLayer(model));
	};
	~Sandbox02() {};
  void RenderImGui() override
  {
    
  }
};
Alalba::Application* Alalba::CreateApplication() {
	return new Sandbox02();
}