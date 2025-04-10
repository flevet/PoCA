#version 330 core
layout(location = 0) in vec3 vertexPosition;
layout(location = 2) in float vertexFeature;
layout(location = 3) in vec4 vertexColor;

uniform mat4 MVP;
uniform mat4 view;
uniform mat4 projection;

const int MAX_CLIPPING_PLANES = 50;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform int nbClipPlanes;

uniform float radius;
uniform vec3 light_position;

out float feature;
out float v_size;
out vec3 v_color;
out vec4 v_eye_position;
out vec3 v_light_direction;
out float vclipDistance;

void main() {
	vec4 pos = vec4(vertexPosition, 1);
	feature = vertexFeature;
	v_color = vertexColor.rgb;
	
    v_eye_position = view * pos;
    v_light_direction = normalize(light_position);
    gl_Position = MVP * pos;
    // stackoverflow.com/questions/8608844/...
    //  ... resizing-point-sprites-based-on-distance-from-the-camera
    vec4 p = projection * vec4(radius, radius, v_eye_position.z, v_eye_position.w);
    v_size = radius;//512.0 * p.x / p.w;
    gl_PointSize = v_size + 5.0;
	
	vclipDistance= 3.402823466e+38;
	for(int n = 0; n < nbClipPlanes; n++){
		float d = dot(pos, clipPlanes[n]);
		vclipDistance = d < vclipDistance ? d : vclipDistance;
	}
}