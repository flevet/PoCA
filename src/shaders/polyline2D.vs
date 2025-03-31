#version 330 core
layout(location = 0) in vec3 points;
uniform mat4 MVP;
uniform uvec4 viewport;
out vec4 v_points;

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
}