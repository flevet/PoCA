#version 330 core
out vec4 FragColor;
//layout (location = 0) out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D ssao;
uniform sampler2D silhouette;

struct Light {
    vec3 Position;
    vec3 Color;
    
    float Linear;
    float Quadratic;
};
uniform Light light;

uniform mat4 projection;
uniform bool useSSAO;
uniform bool useSilhouette;
uniform bool debug;
uniform vec4 backColor;
uniform vec2 screenDimension;
uniform float radius;

void main()
{
	//FragColor = vec4(texture(gNormal, TexCoords).rgb, 1.f);
	vec3 FragPos = texture(gPosition, TexCoords).rgb;
	vec3 Diffuse = texture(gAlbedo, TexCoords).rgb;

	bool backFrag = Diffuse.r == 0.f && Diffuse.g == 0.f && Diffuse.b == 0.f;
	float backAlpha = backFrag ? 0.f : 1.f;
	
	if(debug){
		float z = texture(ssao, TexCoords).z;
		//FragColor = vec4(z, z, z, backAlpha);
		FragColor = vec4(texture(ssao, TexCoords).rgb, backAlpha);
		return;
	}
	
	vec4 FragSil = texture(silhouette, TexCoords);
	//FragColor = FragSil;
	if(useSilhouette && FragSil.a != 0.f){
		FragColor = FragSil;
		return;
	}
	
	/*if(backFrag){
		FragColor = vec4(backColor.rgb, backAlpha);
		return;
	}*/
	
	// retrieve data from gbuffer
	vec3 NormalTmp = texture(gNormal, TexCoords).rgb;
	vec3 Normal = normalize(texture(gNormal, TexCoords).rgb);
	float AmbientOcclusion = texture(ssao, TexCoords).r;
	
	Diffuse = backFrag ? FragPos : Diffuse;
	// then calculate lighting as usual
	vec3 ambient = vec3(1. * Diffuse * AmbientOcclusion);
	vec3 lighting  = ambient; 
	FragColor = vec4(lighting, 1.0);
}
/*void main()
{ 
	FragColor = vec4(1, 0, 0, 1.0);
}*/