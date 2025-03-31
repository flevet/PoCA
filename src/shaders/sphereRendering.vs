#version 330 core
layout(location = 0) in vec3 vertexPosition;
layout(location = 2) in float vertexFeature;
layout(location = 3) in vec4 vertexColor;

uniform mat4 MVP;
uniform mat4 view;
uniform mat4 projection;

uniform vec4 clipPlaneX;
uniform vec4 clipPlaneY;
uniform vec4 clipPlaneZ;
uniform vec4 clipPlaneW;
uniform vec4 clipPlaneH;
uniform vec4 clipPlaneT;

uniform float radius;
uniform vec3 light_position;

out float feature;
out float v_size;
out vec3 v_color;
out vec4 v_eye_position;
out vec3 v_light_direction;

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
	
	gl_ClipDistance[0] = dot(pos, clipPlaneX);
	gl_ClipDistance[1] = dot(pos, clipPlaneY);
	gl_ClipDistance[2] = dot(pos, clipPlaneZ);
	gl_ClipDistance[3] = dot(pos, clipPlaneW);
	gl_ClipDistance[4] = dot(pos, clipPlaneH);
	gl_ClipDistance[5] = dot(pos, clipPlaneT);
}