#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;
out vec3 Normal;
out float v_clipDistance;
void main() {
	vec4 pos = vec4(aPos, 1.0);
	Normal = /*mat3(transpose(inverse(model))) */ aNormal;
	gl_Position = pos;
	v_clipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(pos, clipPlanes[n]);
		v_clipDistance = d < v_clipDistance ? d : v_clipDistance;
	}
}