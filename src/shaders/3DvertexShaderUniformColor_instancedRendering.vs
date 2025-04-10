#version 330 core

layout(location = 0) in vec3 unitCirclePosition;
layout(location = 9) in float feature;
layout(location = 5) in mat4 model_matrix;

uniform mat4 MVP;

const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;

out float vfeature;
out float vclipDistance;

void main(){
	vec4 pos = model_matrix * vec4(unitCirclePosition, 1);

    	gl_Position = MVP * pos;

	vfeature = feature;

	vclipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(pos, clipPlanes[n]);
		vclipDistance = d < vclipDistance ? d : vclipDistance;
	}
}

