#version 330 core
layout(location = 0) in vec3 vertexPosition;
layout(location = 2) in float vertexFeature;
layout(location = 3) in vec4 vertexColor;

const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;

out float v_feature;
out vec3 v_color;
out float v_clipDistance;

void main()
{
	vec4 pos = vec4(vertexPosition, 1);
	gl_Position = pos;
	v_feature = vertexFeature;
	v_color = vertexColor.rgb;
	
	v_clipDistance = 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(pos, clipPlanes[n]);
		v_clipDistance = d < v_clipDistance ? d : v_clipDistance;
	}
}