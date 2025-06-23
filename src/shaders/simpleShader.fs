#version 330 core
in float feature;
in vec4 colorIn;
in vec3 normal;
in float vclipDistance;
out vec4 color;

uniform sampler1D lutTexture;
uniform float minFeatureValue;
uniform float maxFeatureValue;
uniform float alpha;
uniform bool useSpecialColors;
uniform bool activatedCulling; 
uniform vec3 cameraForward; 
uniform bool clip;

uniform bool applyUniformColor;
uniform vec4 uniformColor;

void main() {
	if (clip && vclipDistance < 0.f)
		discard;
	if (activatedCulling) {
		float res = dot(cameraForward, normal); 
		if (res > 0.f)
			discard; 
	}
	if (useSpecialColors) {
		if (colorIn.a == 0.)
			discard;
		color = colorIn;
	}
	else if(applyUniformColor)
		color = vec4(uniformColor.rgb, alpha);
	else {
		if (feature < minFeatureValue)
			discard;
		float inter = maxFeatureValue - minFeatureValue;
		if (inter > -0.0001 && inter < 0.0001)
			color = vec4(texture(lutTexture, 0).xyz, alpha);
		else
			color = vec4(texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz, alpha);//0.01);
	}
}