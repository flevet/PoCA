#version 330 core
layout(location = 0) in vec3 unitCirclePosition;
layout(location = 9) in float feature;
layout(location = 5) in mat4 model_matrix;
layout(location = 10) in float objectIndex;

uniform mat4 MVP;
uniform mat4 model;
uniform samplerBuffer objectModels;

const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;

out float vfeature;
out float vclipDistance;

mat4 fetchObjectModel(int idx)
{
	int base = idx * 4;
	return mat4(
		texelFetch(objectModels, base + 0),
		texelFetch(objectModels, base + 1),
		texelFetch(objectModels, base + 2),
		texelFetch(objectModels, base + 3));
}

void main(){
	mat4 objectModel = fetchObjectModel(int(objectIndex + 0.5));
	vec4 pos = objectModel * model_matrix * vec4(unitCirclePosition, 1.0);
	gl_Position = MVP * pos;
	vfeature = feature;
	vclipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(model * pos, clipPlanes[n]);
		vclipDistance = d < vclipDistance ? d : vclipDistance;
	}
}
