#version 330 core
layout(location = 0) in vec3 points;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in float vertexFeature;
layout(location = 3) in float objectIndex;

uniform mat4 MVP;
uniform mat4 model;
uniform vec4 viewport;
uniform samplerBuffer objectModels;

const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;

out vec3 v_normal;
out float v_feature;
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

vec2 wrldToScreen(vec3 coord, mat4 fullMVP) {
	vec4 clipSpacePos = fullMVP * vec4(coord, 1.0);
	vec3 ndcSpacePos = clipSpacePos.xyz / clipSpacePos.w;
	vec2 viewOffset = viewport.xy, viewSize = viewport.zw;
	return ((ndcSpacePos.xy + 1.0) / 2.0) * viewSize + viewOffset;
}

void main()
{
	mat4 objectModel = fetchObjectModel(int(objectIndex + 0.5));
	mat4 fullMVP = MVP * objectModel;
	vec4 localPos = objectModel * vec4(points, 1.0);
	gl_Position = vec4(wrldToScreen(points, fullMVP), 0.0, 1.0);
	v_normal = mat3(transpose(inverse(model * objectModel))) * vertexNormal;
	v_feature = vertexFeature;
	v_clipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(model * localPos, clipPlanes[n]);
		v_clipDistance = d < v_clipDistance ? d : v_clipDistance;
	}
}
