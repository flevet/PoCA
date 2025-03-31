#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertex_sigmas;
layout(location = 2) in float vertexFeature;
layout(location = 3) in vec4 vertexColor;
uniform mat4 MVP;

uniform vec4 clipPlaneX;
uniform vec4 clipPlaneY;
uniform vec4 clipPlaneZ;
uniform vec4 clipPlaneW;
uniform vec4 clipPlaneH;
uniform vec4 clipPlaneT;

out vec3 sigmas;
out float feature;
out vec4 colorIn;
out float gl_ClipDistance[6];

void main(){	
	vec4 pos = vec4(vertexPosition_modelspace, 1);
	gl_Position = pos;
	sigmas = vertex_sigmas;
	feature = vertexFeature;
	colorIn = vertexColor;

	gl_ClipDistance[0] = dot(pos, clipPlaneX);
	gl_ClipDistance[1] = dot(pos, clipPlaneY);
	gl_ClipDistance[2] = dot(pos, clipPlaneZ);
	gl_ClipDistance[3] = dot(pos, clipPlaneW);
	gl_ClipDistance[4] = dot(pos, clipPlaneH);
	gl_ClipDistance[5] = dot(pos, clipPlaneT);
}

