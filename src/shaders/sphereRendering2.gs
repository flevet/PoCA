#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
uniform mat4 MVP;
uniform mat4 projection;
uniform float radius;

in vec3 v_color[];
in vec3 v_normal[];
in float v_feature[];

out vec2 texC;
out vec3 center;
out vec3 vcolor;
out vec3 vnormal;
out float vfeature;

const vec2 uv[4] = vec2[4](vec2(1, 1),
	vec2(1, -1),
	vec2(-1, 1),
	vec2(-1, -1));
void main() {
	vec4 vc[4];
	vc[0] = vec4(radius, radius, 0.0, 0.0);
	vc[1] = vec4(radius, -radius, 0.0, 0.0);
	vc[2] = vec4(-radius, radius, 0.0, 0.0);
	vc[3] = vec4(-radius, -radius, 0.0, 0.0);
	for (int i = 0; i < 4; i++) {
		gl_Position = MVP * gl_in[0].gl_Position + projection * vc[i];
		texC = uv[i];
		center = gl_in[0].gl_Position.xyz;
		vfeature = v_feature[0];
		vnormal = v_normal[0];
		vcolor = v_color[0];
		gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];
		gl_ClipDistance[1] = gl_in[0].gl_ClipDistance[1];
		gl_ClipDistance[2] = gl_in[0].gl_ClipDistance[2];
		gl_ClipDistance[3] = gl_in[0].gl_ClipDistance[3];
		gl_ClipDistance[4] = gl_in[0].gl_ClipDistance[4];
		gl_ClipDistance[5] = gl_in[0].gl_ClipDistance[5];
		EmitVertex();
	}
	EndPrimitive();
};