#version 330 core
out vec4 color;
uniform float thickness;
uniform float antialias;
uniform bool activatedCulling;
uniform bool useSingleColor;
uniform vec3 cameraForward;
uniform sampler1D lutTexture;
uniform float minFeatureValue;
uniform float maxFeatureValue;
uniform vec4 singleColor;
in float linelength;
in vec2 uv;
in float thick;
in vec3 normal;
in float feature;
void main() {
	if(activatedCulling){
		float res = dot(cameraForward, normal);
		if(res < 0.f)
			discard;
	}
	
	if (feature < minFeatureValue)
			discard;
	
	if(useSingleColor){
		color = singleColor;
	}
	else{
		float inter = maxFeatureValue - minFeatureValue;
		color = vec4(texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz, 1);
	}
	
	float d = 0;
	float w = thick / 2.0 - antialias; 
	// Cap at start
	if (uv.x < 0)
		discard;//d = 0.;// length(uv) - w; 
	// Cap at end
	else if (uv.x >= linelength)
		discard;//d = 0.;//  = length(uv - vec2(linelength, 0)) - w; 
	// Body
	else
		d = abs(uv.y) - w;
	
	if (d < 0) {
		color = vec4(color.rgb, 1.0); 
	}
	else {
		d /= antialias; 
		//color = vec4(singleColor.rgb, exp(-d * d));
		color = vec4(color.rgb, exp(-d * d)); 
	}
}