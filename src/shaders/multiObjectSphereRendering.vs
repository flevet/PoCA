#version 330 core
layout(location = 0) in vec3 vertexPosition;
layout(location = 2) in float vertexFeature;
layout(location = 4) in float objectIndex;

uniform samplerBuffer objectModels;
uniform mat4 model;

const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;

out vec3 v_normal;
out float v_feature;
out vec3 v_color;
out float v_clipDistance;

mat4 fetchObjectModel(int idx)
{
	int base = idx * 4;
	return mat4(
		texelFetch(objectModels, base + 0),
		texelFetch(objectModels, base + 1),
		texelFetch(objectModels, base + 2),
		texelFetch(objectModels, base + 3));
}

void main()
{
	vec4 pos = vec4(vertexPosition, 1.0);
	mat4 objectModel = fetchObjectModel(int(objectIndex + 0.5));
	vec4 localPos = objectModel * pos;
	gl_Position = localPos;
	v_normal = vec3(0.0, 0.0, 1.0);
	v_feature = vertexFeature;
	v_color = vec3(1.0);
	v_clipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(model * localPos, clipPlanes[n]);
		v_clipDistance = d < v_clipDistance ? d : v_clipDistance;
	}
}
