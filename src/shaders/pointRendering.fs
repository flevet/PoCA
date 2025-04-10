#version 330 core
in float feature;
in vec4 colorIn;
in vec2 v_center;
in vec3 v_normal;
in float vclipDistance;
out vec4 color;
uniform sampler1D lutTexture;
uniform float minFeatureValue;
uniform float maxFeatureValue;
uniform bool useSpecialColors;
uniform float sizePoints;
uniform bool activatedAntialias;
uniform bool activatedCulling;
uniform float antialias;
uniform vec3 cameraForward;
uniform bool clip;
void main() {
	if(clip && vclipDistance < 0.f)
		discard;
		
	if(activatedCulling){
		float res = dot(cameraForward, v_normal);
		if(res < 0.f)
			discard;
	}

	vec2 p = gl_FragCoord.xy - v_center;
	float a = 1.0;
	if(!activatedAntialias){
		if(length(p) > sizePoints / 2.)
			discard;
	}
	else {
		float d = length(p) - (sizePoints / 2. - antialias);
		if(d > 0.0) a = exp(-d * d);
	}
	if (useSpecialColors) {
		color = vec4(colorIn.rgb, a);
	}
	else {
		if (feature < minFeatureValue)
			discard;
		float inter = maxFeatureValue - minFeatureValue;
		color = vec4(texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz, a);
	}
}