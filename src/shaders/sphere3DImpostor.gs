#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
uniform mat4 MVP;
uniform mat4 projection;
in float v_radius[];
in vec3 v_normal[];
in float v_clipDistance[];
out vec2 texC;
out vec3 center;
out vec3 normal;
out float radius;
out float vclipDistance;

const vec2 uv[4] = vec2[4](vec2(1, 1),
	vec2(1, -1),
	vec2(-1, 1),
	vec2(-1, -1));
void main() {
	vec4 vc[4];
	vc[0] = vec4(v_radius[0], v_radius[0], 0.0, 0.0);
	vc[1] = vec4(v_radius[0], -v_radius[0], 0.0, 0.0);
	vc[2] = vec4(-v_radius[0], v_radius[0], 0.0, 0.0);
	vc[3] = vec4(-v_radius[0], -v_radius[0], 0.0, 0.0);
	for (int i = 0; i < 4; i++) {
		gl_Position = MVP * gl_in[0].gl_Position + projection * vc[i];
		texC = uv[i];
		center = gl_in[0].gl_Position.xyz;
		normal = v_normal[0];
		radius = v_radius[0];
		vclipDistance = v_clipDistance[0];
		EmitVertex();
	}
	EndPrimitive();
};