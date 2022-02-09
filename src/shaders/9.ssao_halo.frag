#version 330 core

layout (location = 0) out vec3 gHalo;

uniform float nbPoints;
uniform float minFeatureValue;

in vec2 TexCoords_GS;
in vec3 center;
in vec3 spot_sigma;
in float spot_feature;

void main()
{
	if(spot_feature < minFeatureValue) 
		discard;
		
	//vec2 texC = TexCoords_GS;
	vec2 texC =  ( TexCoords_GS - 0.5 ) * 2;//vec2(1./0.5, 1./0.5);

    float d = dot(texC, texC);

    if (d > 1.0f)
    {
        discard;
    }
	
	float realD = d * (spot_sigma.x * 1.2);
	float haloD = realD < spot_sigma.x ? realD / spot_sigma.x : 1.f - ((realD - spot_sigma.x) / spot_sigma.x); 
	//gHalo = vec3(gl_PrimitiveID / nbPoints, haloD, gl_PrimitiveID / nbPoints);
	gHalo = vec3(haloD, haloD, haloD);
}