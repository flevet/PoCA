#version 330 core

in float vfeature;
in float vclipDistance;

out vec4 color;

uniform bool useSingleColor;
uniform vec4 singleColor;
uniform bool clip;

uniform sampler1D lutTexture;
uniform float minFeatureValue;
uniform float maxFeatureValue;

void main(){
	if(clip && vclipDistance < 0.f)
		discard;
		
	if(vfeature < minFeatureValue) 
		discard;
	
	if(useSingleColor)
		color = singleColor;
	else{
		float inter = maxFeatureValue - minFeatureValue;
		color = vec4(texture(lutTexture, ((vfeature - minFeatureValue) / inter)).xyz, 1);
	}
}