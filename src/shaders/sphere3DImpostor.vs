#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal;
layout(location = 1) in float vRadius;

const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;

out vec3 v_normal;
out float v_radius;
out float v_clipDistance;

void main()
{
	vec4 pos = vec4(vertexPosition_modelspace, 1);
	gl_Position = pos;
	v_normal = vertexNormal;
	v_radius = vRadius;
	
	v_clipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(pos, clipPlanes[n]);
		v_clipDistance = d < v_clipDistance ? d : v_clipDistance;
	}
};