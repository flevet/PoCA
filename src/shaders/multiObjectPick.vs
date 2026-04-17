#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in float vertexIndex;
layout(location = 2) in float vertexFeature;
layout(location = 3) in float objectIndex;

uniform mat4 MVP;
uniform mat4 model;
uniform bool hasFeature;
uniform samplerBuffer objectModels;

const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;

out float id;
out float feature;
out float vclipDistance;

mat4 fetchObjectModel(int idx)
{
	int base = idx * 4;
	return mat4(
		texelFetch(objectModels, base + 0),
		texelFetch(objectModels, base + 1),
		texelFetch(objectModels, base + 2),
		texelFetch(objectModels, base + 3));
}

void main() {
	vec4 pos = vec4(vertexPosition_modelspace, 1.0);
	mat4 objectModel = fetchObjectModel(int(objectIndex + 0.5));
	vec4 localPos = objectModel * pos;
	gl_Position = MVP * localPos;
	id = vertexIndex;
	feature = hasFeature ? vertexFeature : 3.402823466e+38;
	vclipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(model * localPos, clipPlanes[n]);
		vclipDistance = d < vclipDistance ? d : vclipDistance;
	}
}
