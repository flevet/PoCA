#version 330 core

layout(location = 0) in vec3 unitCirclePosition;
layout(location = 9) in float feature;
layout(location = 5) in mat4 model_matrix;

uniform mat4 MVP;

uniform vec4 clipPlaneX;
uniform vec4 clipPlaneY;
uniform vec4 clipPlaneZ;
uniform vec4 clipPlaneW;
uniform vec4 clipPlaneH;
uniform vec4 clipPlaneT;

out float vfeature;

void main(){
	vec4 pos = model_matrix * vec4(unitCirclePosition, 1);

    gl_Position = MVP * pos;

	vfeature = feature;

	gl_ClipDistance[0] = dot(pos, clipPlaneX);
	gl_ClipDistance[1] = dot(pos, clipPlaneY);
	gl_ClipDistance[2] = dot(pos, clipPlaneZ);
	gl_ClipDistance[3] = dot(pos, clipPlaneW);
	gl_ClipDistance[4] = dot(pos, clipPlaneH);
	gl_ClipDistance[5] = dot(pos, clipPlaneT);
}

