#version 330 core
layout(location = 0) in vec3 points;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in float vertexFeature;
uniform mat4 MVP;
uniform uvec4 viewport;
uniform vec4 clipPlaneX;
uniform vec4 clipPlaneY;
uniform vec4 clipPlaneZ;
uniform vec4 clipPlaneW;
uniform vec4 clipPlaneH;
uniform vec4 clipPlaneT;
out vec3 v_normal;
out float v_feature;

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
	gl_ClipDistance[0] = dot(wrldpos, clipPlaneX);
	gl_ClipDistance[1] = dot(wrldpos, clipPlaneY);
	gl_ClipDistance[2] = dot(wrldpos, clipPlaneZ);
	gl_ClipDistance[3] = dot(wrldpos, clipPlaneW);
	gl_ClipDistance[4] = dot(wrldpos, clipPlaneH);
	gl_ClipDistance[5] = dot(wrldpos, clipPlaneT);
}