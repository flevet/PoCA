#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;
uniform mat4 MVP;
uniform mat4 projection;
uniform float radius;
uniform vec2 pixelSize;
uniform bool screenRadius;
out vec2 TexCoords_GS;
out vec3 center;
const vec2 uv[4] = vec2[4](vec2(1,1), vec2(1,0), vec2(0,1),	vec2(0,0));
void main() {
	vec4 vc[4];
	vc[0] = vec4(radius, radius, 0.0, 0.0);
	vc[1] = vec4(radius, -radius, 0.0, 0.0);
	vc[2] = vec4(-radius, radius, 0.0, 0.0);
	vc[3] = vec4(-radius, -radius, 0.0, 0.0);
	for(int i = 0; i < 4; i++){
//		gl_Position = MVP * (gl_in[0].gl_Position + vc[i]);
		gl_Position = screenRadius ? (MVP * gl_in[0].gl_Position) + (vc[i] * vec4(pixelSize, 0.f, 0.f)) : (MVP * gl_in[0].gl_Position) + (projection * vc[i]);
		TexCoords_GS = uv[i];
		center = gl_in[0].gl_Position.xyz;
		gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];
		gl_ClipDistance[1] = gl_in[0].gl_ClipDistance[1];
		gl_ClipDistance[2] = gl_in[0].gl_ClipDistance[2];
		gl_ClipDistance[3] = gl_in[0].gl_ClipDistance[3];
		gl_ClipDistance[4] = gl_in[0].gl_ClipDistance[4];
		gl_ClipDistance[5] = gl_in[0].gl_ClipDistance[5];
		EmitVertex();
	}
	EndPrimitive();
}