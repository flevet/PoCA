#version 330 core
layout(location = 0) in vec3 centers;
uniform mat4 MVP;
uniform uvec4 viewport;
uniform float radius;
uniform float thickness;
uniform float antialias;
out float radius_vs;
out float radius_vs2;

vec2 wrldToScreen(vec3 coord) {
  vec4 clipSpacePos = MVP * vec4(coord, 1.f);
  vec3 ndcSpacePos = vec3(clipSpacePos.x, clipSpacePos.y, clipSpacePos.z) / clipSpacePos.w;
  vec2 viewOffset = vec2(viewport.x, viewport.y), viewSize = vec2(viewport[2], viewport[3]);
  vec2 windowSpacePos = ((vec2(ndcSpacePos.x, ndcSpacePos.y) + 1.f) / 2.f) * viewSize + viewOffset;
  return windowSpacePos;
}

void main()
{
	vec2 center = wrldToScreen(vec3(centers.xyz));
	vec2 offset = wrldToScreen(vec3(centers.x + radius, centers.y, 0.));
	radius_vs = length(offset - center);
	radius_vs2 = radius_vs + 2 * thickness + antialias;
	gl_Position = vec4(centers, 1.0);
}