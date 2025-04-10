#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in float v_selection;
const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;
uniform float minX;
out float v_clipDistance;
void main() {
	vec4 pos = vec4(vertexPosition_modelspace, 1);
	if(v_selection == 0.f)
		pos = vec4(minX - 10.f, 0.f, 0.f, 1.f);
	gl_Position = pos;
	v_clipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(pos, clipPlanes[n]);
		v_clipDistance = d < v_clipDistance ? d : v_clipDistance;
	}
}