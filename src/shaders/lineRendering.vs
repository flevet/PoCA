#version 330 core
layout(location = 0) in vec3 points;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in float vertexFeature;
uniform mat4 MVP;
uniform uvec4 viewport;
const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;
out vec3 v_normal;
out float v_feature;
out float v_clipDistance;

vec2 wrldToScreen(vec3 coord) {
  vec4 clipSpacePos = MVP * vec4(coord, 1.f);
  vec3 ndcSpacePos = vec3(clipSpacePos.x, clipSpacePos.y, clipSpacePos.z) / clipSpacePos.w;
  vec2 viewOffset = vec2(viewport.x, viewport.y), viewSize = vec2(viewport[2], viewport[3]);
  vec2 windowSpacePos = ((vec2(ndcSpacePos.x, ndcSpacePos.y) + 1.f) / 2.f) * viewSize + viewOffset;
  return windowSpacePos;
}

void main()
{
	vec4 pos = vec4(wrldToScreen(vec3(points.xyz)), 0., 1.);
	gl_Position = pos;
	v_normal = vertexNormal;
	v_feature = vertexFeature;
	
	vec4 wrldpos = vec4(points, 1.);
	v_clipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(pos, clipPlanes[n]);
		v_clipDistance = d < v_clipDistance ? d : v_clipDistance;
	}
}