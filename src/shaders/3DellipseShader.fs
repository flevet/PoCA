#version 330 core
in vec3 TexCoords_GS;
in vec3 spot_sigma;
in float spot_feature;
in vec4 spot_position;

uniform sampler1D lutTexture;
uniform vec3 cameraPos;
uniform float minFeatureValue;
uniform float maxFeatureValue;
uniform float globalAlpha;
uniform float radius;
uniform bool fixedRadius;

out vec4 color;

void distanceToEllipsoidCenterFixed(in vec3 position, out float d){
	vec3 p = position - TexCoords_GS;
	d = (pow(p.x, 2.) / pow(radius, 2)) + (pow(p.y, 2.) / pow(radius, 2)) + (pow(p.z, 2) / pow(radius, 2));
}

void distanceToEllipsoidCenterSigma(in vec3 position, out float d){
	vec3 p = position - TexCoords_GS;
	d = (pow(p.x, 2.) / pow(spot_sigma.x, 2)) + (pow(p.y, 2.) / pow(spot_sigma.y, 2)) + (pow(p.z, 2) / pow(spot_sigma.z, 2));
}

void main()
{
	if(spot_feature < minFeatureValue) 
		discard;

	vec3 position = spot_position.xyz;

	//Ray marching test
	float nbSteps = 25;
	float distanceCube = 2 * radius;
	float stepSize = distanceCube / nbSteps;

	vec3 direction = -cameraPos;
	vec3 centre =  TexCoords_GS;
	vec3 norm;
	float minD = 10000;

	float found = 0., d;
	
	if(fixedRadius)
		for (int i = 0; i < nbSteps ; i++)
		{
			distanceToEllipsoidCenterFixed(position, d);// distance(position, centre);
			if ( d <= 1 ){
				found = found + 1;
				minD = min(minD, d);
				//norm = normalize(position - centre);
			}
	 
			position += (direction * stepSize);
		}
	else
		for (int i = 0; i < nbSteps ; i++)
		{
			distanceToEllipsoidCenterSigma(position, d);// distance(position, centre);
			if ( d <= 1 ){
				found = found + 1;
				minD = min(minD, d);
				//norm = normalize(position - centre);
			}
	 
			position += (direction * stepSize);
		}
	if(found < 0.5)
		discard;
	float inter = maxFeatureValue - minFeatureValue;
	color = vec4(texture(lutTexture, ((spot_feature - minFeatureValue) / inter)).xyz, globalAlpha * (1-minD));// * (1-minD);// vec4(1,0,0,1);// fragmentColor;
}  