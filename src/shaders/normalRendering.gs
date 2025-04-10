#version 330 core
layout(triangles) in;
layout(line_strip, max_vertices = 2) out;
uniform mat4 MVP;

in vec3 Normal[];
in float v_clipDistance[];

out float vclipDistance;

void main() {
	vec3 pos = (gl_in[0].gl_Position.xyz + gl_in[1].gl_Position.xyz + gl_in[2].gl_Position.xyz) / 3.;
	gl_Position = MVP * vec4(pos, 1);
	vclipDistance = v_clipDistance[0];
	EmitVertex();
	
	vec3 pos2 = pos + 10. * Normal[0];
	gl_Position = MVP * vec4(pos2, 1);
	vclipDistance = v_clipDistance[0];
	EmitVertex();
	
	EndPrimitive();
};