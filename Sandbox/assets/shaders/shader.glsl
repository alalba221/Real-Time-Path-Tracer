#type vertex
#version 430

layout(location = 0) in vec3 a_Position;


uniform mat4 u_ViewProjectionMatrix;
uniform mat4 u_ModelMatrix;

void main()
{

	gl_Position = u_ViewProjectionMatrix * u_ModelMatrix * vec4(a_Position, 1.0);
}

#type fragment
#version 430
uniform vec4 u_Color;
layout(location = 0) out vec4 finalColor;

void main()
{
	finalColor = vec4(u_Color);
}
