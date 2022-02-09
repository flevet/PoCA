#version 330 core
uniform sampler1D lutTexture;
uniform float minFeatureValue;
uniform float maxFeatureValue;
uniform float alpha;

uniform bool useSpecialColors;

in vec2 texC;
in vec3 center;
in float vfeature;
in vec3 vcolor;
in vec3 vsigma;
in float vradius;

out vec4 color_out;

float gauss(float x, float x0, float sx){
    float arg = x-x0;
    arg = -1./2.*arg*arg/sx;
    
    float a = 1./(pow(2.*3.1415*sx, 0.5));
    
    return a*exp(arg);
}

float gauss(float arg, float sx){
    //arg = -1./2.*arg*arg/sx;
    //float a = 1./(pow(2.*3.1415*sx, 0.5));
	
	float a = 1./(sx * sqrt(2 * 3.1415));
	float b = -(arg*arg)/(2*sx*sx);
    
    return a*exp(b);
}

void main()
{
	float d = dot(texC, texC);
	//if (d > 1.0f)
	//	discard;
	//float normalizedMaxAmplitude = 1. / gauss(0., vradius / 3.);
	//float amplitude = gauss(d * vradius, vradius / 3.) * normalizedMaxAmplitude * 1;
	float normalizedMaxAmplitude = 1. / gauss(0., 1);
	float amplitude = gauss(d * 3, 1) * normalizedMaxAmplitude * alpha;
		
	vec3 colorTmp;
	if (useSpecialColors) {
		colorTmp = vec3(vcolor);
	}
	else {
		if (vfeature < minFeatureValue)
			discard;
		float inter = maxFeatureValue - minFeatureValue;
		colorTmp = vec3(texture(lutTexture, ((vfeature - minFeatureValue) / inter)).xyz);
	}
	
    color_out = vec4(colorTmp, amplitude);
};