#version 330 core
in vec3 TexCoords_GS;
in vec3 spot_axes;
in vec4 spot_position;
in vec3 spot_center;
in mat3 rotMat;

uniform mat4 MVP;
uniform vec4 singleColor;
uniform vec3 cameraForward;

out vec4 color;

void distanceToEllipsoidCenter(in vec3 position, out float d){
	vec3 p = transpose(rotMat) * (position - TexCoords_GS);
	d = (pow(p.x, 2.) / pow(spot_axes.x, 2)) + (pow(p.y, 2.) / pow(spot_axes.y, 2)) + (pow(p.z, 2) / pow(spot_axes.z, 2));
}

/*void distanceToEllipsoidCenter(in vec3 position, out float d){
	vec3 p = position - TexCoords_GS;
	d = (pow(p.x, 2.) / pow(spot_axes.x, 2)) + (pow(p.y, 2.) / pow(spot_axes.y, 2)) + (pow(p.z, 2) / pow(spot_axes.z, 2));
}*/

/*void distanceToEllipsoidCenter(in vec3 position, out float d){
	vec3 p = transpose(rotMat) * (position - TexCoords_GS);
	d = (pow(p.x, 2.) / pow(spot_axes.x, 2)) + (pow(p.y, 2.) / pow(spot_axes.y, 2)) + (pow(p.z, 2) / pow(spot_axes.z, 2));
}*/

//void distanceToEllipsoidCenter(in vec3 position, out float d){
//	vec3 p = /*transpose(rotMat) **/ (position - TexCoords_GS);
//	//p = p * 20.;
//	d = (pow(p.x, 2.) / pow(1, 2)) + (pow(p.y, 2.) / pow(1, 2)) + (pow(p.z, 2) / pow(1, 2));
//}

void main()
{
	//color = vec4(singleColor.rgb, 0.3);
	vec3 position = spot_position.xyz;

	//Ray marching test
	float nbSteps = 25;
	float distanceCube = 2 * spot_axes.z;
	float stepSize = distanceCube / nbSteps;

	vec3 direction = cameraForward;// position - cameraPos;//vec3(position.xy, cameraPos.z);
	vec3 centre =  TexCoords_GS;
	vec3 norm;
	float minD = 10000;

	bool found = false;
	float d;
	for (int i = 0; i < nbSteps && !found; i++)
	{
		distanceToEllipsoidCenter(position, d);// distance(position, centre);
		if (d <= 1)
			found = true;
		else
			position += (direction * stepSize);
	}
	if(!found)
		discard;
	vec4 clipPos = MVP * vec4(position, 1.0f);
	float ndcDepth = clipPos.z / clipPos.w;
	gl_FragDepth = ((gl_DepthRange.diff * ndcDepth) + gl_DepthRange.near + gl_DepthRange.far) / 2.f;
	color = vec4(singleColor.rgb, 0.3);
}  

/*float sdEllipsoid( in vec3 p, in vec3 r ) // approximated
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

void main()
{
	vec3 position = spot_position.xyz;
	vec3 p = position - TexCoords_GS;
	float d = sdEllipsoid(p, spot_axes);
	//distanceToEllipsoidCenter(position, d);
	if (d < 1)
		discard;
	color = vec4(singleColor.rgb, 0.3);
}*/