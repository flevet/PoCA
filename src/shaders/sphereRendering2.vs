#version 330 core
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in float vertexFeature;
layout(location = 3) in vec4 vertexColor;

uniform vec4 clipPlaneX;
uniform vec4 clipPlaneY;
uniform vec4 clipPlaneZ;
uniform vec4 clipPlaneW;
uniform vec4 clipPlaneH;
uniform vec4 clipPlaneT;

out vec3 v_normal;
out float v_feature;
out vec3 v_color;

void main()
{
	vec4 pos = vec4(vertexPosition, 1);
	gl_Position = pos;
	v_normal = vertexNormal;
	v_feature = vertexFeature;
	v_color = vertexColor.rgb;
	
	gl_ClipDistance[0] = dot(pos, clipPlaneX);
	gl_ClipDistance[1] = dot(pos, clipPlaneY);
	gl_ClipDistance[2] = dot(pos, clipPlaneZ);
	gl_ClipDistance[3] = dot(pos, clipPlaneW);
	gl_ClipDistance[4] = dot(pos, clipPlaneH);
	gl_ClipDistance[5] = dot(pos, clipPlaneT);
};