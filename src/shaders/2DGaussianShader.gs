#version 330 core
layout (points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec3 v_sigma[];
in float v_feature[];
in vec3 v_color[];
in float v_clipDistance[];

uniform mat4 MVP;
uniform mat4 projection;
uniform float radius;
uniform bool fixedRadius;

out vec2 texC;
out vec3 center;
out vec3 vcolor;
out vec3 vsigma;
out float vfeature;
out float vradius;
out float vclipDistance;

const vec2 uv[4] = vec2[4](vec2(1, 1),
	vec2(1, -1),
	vec2(-1, 1),
	vec2(-1, -1));

void main() {
		vec4 vc[4];
		if(fixedRadius)
			vradius = radius;
		else
			vradius = v_sigma[0].x;
		vc[0] = vec4(vradius, vradius, 0.0, 0.0);
		vc[1] = vec4(vradius, -vradius, 0.0, 0.0);
		vc[2] = vec4(-vradius, vradius, 0.0, 0.0);
		vc[3] = vec4(-vradius, -vradius, 0.0, 0.0);

		for (int i = 0; i < 4; i++) {
			gl_Position = MVP * gl_in[0].gl_Position + projection * vc[i];
			texC = uv[i];
			center = gl_in[0].gl_Position.xyz;
			vfeature = v_feature[0];
			vsigma = v_sigma[0];
			vcolor = v_color[0];
			vclipDistance = v_clipDistance[0];
			EmitVertex();
		}    
    EndPrimitive();
}  