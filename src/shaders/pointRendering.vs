#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in float vertexFeature;
layout(location = 3) in vec4 vertexColor;
uniform mat4 MVP;
const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;
uniform uvec4 viewport;
uniform float sizePoints;
uniform bool activatedAntialias;
uniform float antialias;
out float feature;
out vec4 colorIn;
out vec2 v_center;
out vec3 v_normal;
out float vclipDistance;
vec2 wrldToScreen(vec3 coord) {
  vec4 clipSpacePos = MVP * vec4(coord, 1.f);
  vec3 ndcSpacePos = vec3(clipSpacePos.x, clipSpacePos.y, clipSpacePos.z) / clipSpacePos.w;
  vec2 viewOffset = vec2(viewport.x, viewport.y), viewSize = vec2(viewport[2], viewport[3]);
  vec2 windowSpacePos = ((vec2(ndcSpacePos.x, ndcSpacePos.y) + 1.f) / 2.f) * viewSize + viewOffset;
  return windowSpacePos;
}
void main() {
	v_center = wrldToScreen(vertexPosition_modelspace.xyz);
	vec4 pos = vec4(vertexPosition_modelspace, 1);
	gl_Position = MVP * pos;
	gl_PointSize = sizePoints;
	if(activatedAntialias)
		gl_PointSize = gl_PointSize + 2.5 * antialias;
	feature = vertexFeature;
	colorIn = vertexColor;
	v_normal = vertexNormal;//(MVP * vec4(vertexNormal, 1)).xyz;
	vclipDistance= 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(pos, clipPlanes[n]);
		vclipDistance = d < vclipDistance ? d : vclipDistance;
	}
}