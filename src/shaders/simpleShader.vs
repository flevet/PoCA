#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 2) in float vertexFeature;
layout(location = 3) in vec4 vertexColor;
layout(location = 4) in vec3 vertexNormal;

uniform mat4 MVP;
uniform mat4 model;
const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;
out float feature;
out vec4 colorIn;
out vec3 normal;
out float vclipDistance;

void main() {
	vec4 pos = vec4(vertexPosition_modelspace, 1);
	gl_Position = MVP * pos;
	feature = vertexFeature;
	colorIn = vertexColor;
	normal = vertexNormal;
	vclipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(model * pos, clipPlanes[n]);
		vclipDistance = d < vclipDistance ? d : vclipDistance;
	}
}