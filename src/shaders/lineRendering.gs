#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform mat4 MVP;
uniform vec2 resolution;
uniform float thickness;
uniform float antialias;
in vec3 v_normal[];
in float v_feature[];
out vec2 uv;
out float linelength;
out float thick;
out vec3 normal;
out float feature;
void main() {
	vec2 p0 = vec2(gl_in[0].gl_Position.xy), p1 = vec2(gl_in[1].gl_Position.xy);
	float lengthL = length(p1-p0);
	vec2 T = normalize(p1-p0);
   vec2 O = vec2(-T.y, T.x);
   O = -O;
   vec2 axis = vec2(1, 0);
   thick = thickness;
   //if(dot(T, axis) > 0)
	//	O = -O;
	vec2 toto = p1-p0;
	//if(dot(T, axis) > 0.)
	//	lengthL = 0;
   float d = ceil(thick + 2.5 * antialias);
	vec2 offset = T * d;
	vec2 displacement = O * (d / 2.0);
	vec2 vc[4];
	vc[0] = vec2(p0 + displacement - offset);
	vc[1] = vec2(p1 + displacement + offset);
	vc[2] = vec2(p0 - displacement - offset);
	vc[3] = vec2(p1 - displacement + offset);
	vec3 norms[4];
	norms[0] = v_normal[0];
	norms[1] = v_normal[1];
	norms[2] = v_normal[0];
	norms[3] = v_normal[1];
	float feats[4];
	feats[0] = v_feature[0];
	feats[1] = v_feature[1];
	feats[2] = v_feature[0];
	feats[3] = v_feature[1];
   vec2 TexCoords_GS[4] = vec2[4](vec2(-d, d),
	    vec2(lengthL + d, d),
	    vec2(-d, -d),
	    vec2(lengthL + d, -d));
	for (int i = 0; i < 4; i++) {
		gl_Position = vec4(2.0*vc[i]/resolution-1.0, 0.0, 1.0);
		uv = TexCoords_GS[i];
		linelength = lengthL;
		normal = norms[i];
		feature = feats[i];
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