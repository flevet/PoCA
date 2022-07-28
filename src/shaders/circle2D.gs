#version 330 core
layout (points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 MVP;
uniform mat4 projection;
uniform float thickness;
uniform float radius;
uniform float antialias;

in float radius_vs[];
in float radius_vs2[];

out vec2 texC;
out vec3 center;
out float vradius;
out float vradius2;

const vec2 uv[4] = vec2[4](vec2(1, 1),
	vec2(1, -1),
	vec2(-1, 1),
	vec2(-1, -1));

void main() {
		vec4 vc[4];
		float rapport = abs(radius_vs2[0] / radius_vs[0]);
		vradius = radius * rapport;
		vc[0] = vec4(vradius, vradius, 0.0, 0.0);
		vc[1] = vec4(vradius, -vradius, 0.0, 0.0);
		vc[2] = vec4(-vradius, vradius, 0.0, 0.0);
		vc[3] = vec4(-vradius, -vradius, 0.0, 0.0);
		
		vradius = radius_vs[0];
		vradius2 = radius_vs2[0];

		for (int i = 0; i < 4; i++) {
			gl_Position = MVP * gl_in[0].gl_Position + projection * vc[i];
			texC = uv[i];
			center = gl_in[0].gl_Position.xyz;
			EmitVertex();
		}    
    EndPrimitive();
}  