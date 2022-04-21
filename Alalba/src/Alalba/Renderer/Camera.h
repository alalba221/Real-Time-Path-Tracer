#pragma once

#include <glm/glm.hpp>
// for gdt camera
namespace Alalba {

	class ALALBA_API Camera
	{
	public:
		Camera(const glm::mat4& projectionMatrix);
		Camera(const glm::vec3& from, const glm::vec3& at, const glm::vec3& up);

		void Focus();
		void Update();

		inline float GetDistance() const { return m_Distance; }
		inline void SetDistance(float distance) { m_Distance = distance; }

		inline void SetProjectionMatrix(const glm::mat4& projectionMatrix) { m_ProjectionMatrix = projectionMatrix; }

		const glm::mat4& GetProjectionMatrix() const { return m_ProjectionMatrix; }
		const glm::mat4& GetViewMatrix() const { return m_ViewMatrix; }

		glm::vec3 GetUpDirection();
		glm::vec3 GetRightDirection();
		glm::vec3 GetForwardDirection();
		const glm::vec3& GetPosition() const { return m_Position; }

		bool m_Changed = true;
	private:
		void MousePan(const glm::vec2& delta);
		void MouseRotate(const glm::vec2& delta);
		void MouseZoom(float delta);

		glm::vec3 CalculatePosition();
		glm::quat GetOrientation();
	private:
		glm::mat4 m_ProjectionMatrix, m_ViewMatrix;
		glm::vec3 m_Position, m_Rotation, m_FocalPoint, m_Up;

		bool m_Panning, m_Rotating;
		glm::vec2 m_InitialMousePosition;
		glm::vec3 m_InitialFocalPoint, m_InitialRotation;

		float m_Distance;
		float m_PanSpeed, m_RotationSpeed, m_ZoomSpeed;

		float m_Pitch, m_Yaw;
	};

}


