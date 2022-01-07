#type vertex
#version 430

layout(location = 0) in vec3 a_Position;
layout(location = 4) in vec2 a_TexCoord;

uniform mat4 u_ViewProjectionMatrix;
uniform mat4 u_ModelMatrix;

out vec2 TexCoord;
void main()
{

	gl_Position = u_ViewProjectionMatrix * u_ModelMatrix * vec4(a_Position, 1.0);

	TexCoord = a_TexCoord;
}

#type fragment
#version 430
uniform vec4 u_Color;
out vec4 finalColor;
in vec2 TexCoord;
uniform sampler2D u_BRDFLUTTexture;
uniform sampler2D u_Texture;
void main()
{
	finalColor = texture(u_Texture, 1-TexCoord); 
}
