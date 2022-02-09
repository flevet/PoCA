#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D ssaoInput;
uniform vec2 screenDimension;
uniform vec4 backColor;

uniform float radius;
uniform vec2 directions[64];

const int dx[8] = int[](-1, 0, 1, 1, 1, 0, -1, -1);
const int dy[8] = int[](1, 1, 1, 0, -1, -1, -1, 0);

void main() 
{
	bool changed = false;
	vec4 col = texture2D(ssaoInput, TexCoords);
	vec4 tmp;
	float minD = radius;
	if(col.a == 0.f){//col.r == 0.f && col.g == 0.f && col.b == 0.f){
		for(int i = 0; i < 64; ++i)
		{
			vec2 coords = vec2(TexCoords.x + ((radius * directions[i].x) / screenDimension.x), TexCoords.y + ((radius * directions[i].y) / screenDimension.y));
			vec4 gInfo = texture2D(ssaoInput, coords);
			if(gInfo.a != 0)
			{
				changed = true;
				tmp = gInfo;
				bool found = false;
				for(int j = 1; j < radius && !found; j++){
					coords = vec2(TexCoords.x + ((j * directions[i].x) / screenDimension.x), TexCoords.y + ((j * directions[i].y) / screenDimension.y));
					gInfo = texture2D(ssaoInput, coords);
					if(gInfo.a != 0)
					{
						found = true;
						minD = min(minD, j);
					}
				}
			}
		}
		float r = minD / radius;
		vec4 color = changed ? vec4(r,r,r,1) : vec4(backColor.rgb, 0.);
		FragColor = color;
		return;
	}
	discard;
	/*bool border = false;
	float index = texture2D(ssaoInput, TexCoords).r;
	for(int i = 0; i < 8; i++){
		float index2 = texture2D(ssaoInput, vec2(TexCoords.x + (dx[i] / screenDimension.x), TexCoords.y + (dy[i] / screenDimension.y))).r;
		border = border || (index2 != index);
	}
	if(border)
		FragColor = vec4(0,0,0,1);
	else
		FragColor = vec4(1,1,1,0);*/
		
	//FragColor = vec4(col.g, col.g, col.g, col.g);
	//float d = texture2D(ssaoInput, TexCoords).g;
	//FragColor = vec4(d, d, d, 1);
	/*float index = texture2D(ssaoInput, TexCoords).r;
	bool border = false;
	for(int i = 0; i < 8; i++){
		float index2 = texture2D(ssaoInput, vec2(TexCoords.x + (dx[i] / screenDimension.x), TexCoords.y + (dy[i] / screenDimension.y))).r;
		border = border || (index2 != index);
	}
	if(border)
		FragColor = vec4(0,0,0,1);
	else
		FragColor = vec4(1,1,1,1);*/
}