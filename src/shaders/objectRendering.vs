#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float vertexFeature;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;
out float feature;
out vec3 FragPos;
out vec3 Normal;
out vec3 NormalSSAO;
out float vclipDistance;
void main() {
	vec4 pos = vec4(aPos, 1.0);
	FragPos = vec3(model * pos);
	Normal = mat3(transpose(inverse(model))) * aNormal;
	NormalSSAO = mat3(transpose(inverse(model))) * aNormal;
	gl_Position = (projection * view) * vec4(FragPos, 1.0);
	feature = vertexFeature;
	vclipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(pos, clipPlanes[n]);
		vclipDistance = d < vclipDistance ? d : vclipDistance;
	}
}