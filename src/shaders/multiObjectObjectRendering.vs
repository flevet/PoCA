#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float vertexFeature;
layout(location = 3) in float objectIndex;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform samplerBuffer objectModels;

const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;

out float feature;
out vec3 FragPos;
out vec3 Normal;
out vec3 NormalSSAO;
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
	vec4 pos = vec4(aPos, 1.0);
	mat4 objectModel = fetchObjectModel(int(objectIndex + 0.5));
	mat4 fullModel = model * objectModel;
	FragPos = vec3(fullModel * pos);
	Normal = mat3(transpose(inverse(fullModel))) * aNormal;
	NormalSSAO = Normal;
	gl_Position = projection * view * vec4(FragPos, 1.0);
	feature = vertexFeature;
	vclipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(fullModel * pos, clipPlanes[n]);
		vclipDistance = d < vclipDistance ? d : vclipDistance;
	}
}
