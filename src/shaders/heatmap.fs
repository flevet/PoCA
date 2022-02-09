#version 330 core
layout (location = 0) out float gColor;
in vec2 TexCoords_GS;
in vec3 center;
uniform highp float u_intensity;
#define GAUSS_COEF 0.3989422804014327
void main() {
	vec2 texC =  ( TexCoords_GS - 0.5 ) * 2;
	float d = dot(texC, texC);
	if (d > 1.0f) {
		/*gColor = 0; //*/discard;
	}
	else{
		float d2 = -0.5 * 3.0 * 3.0 * d;
		float val = u_intensity * GAUSS_COEF * exp(d2);
		gColor = val;
	}
}