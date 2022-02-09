#version 330 core
layout (location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec3 gAlbedo;
in float feature;
in vec3 Normal;
in vec3 NormalSSAO;
in vec3 FragPos;
uniform mat4 view;
uniform mat4 model;
uniform sampler1D lutTexture;
uniform float minFeatureValue;
uniform float maxFeatureValue;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform bool applyIllumination;
uniform vec3 light_position;

void main() {
	if (feature < minFeatureValue)
		discard;
	float inter = maxFeatureValue - minFeatureValue;
	vec3 objectColor = texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz;
	// ambient
	float ambientStrength = 0.1;
	vec3 ambient = ambientStrength * lightColor;
	// diffuse 
	vec3 norm = normalize(Normal);
	vec3 lightDir = normalize(light_position);//normalize(lightPos - FragPos);
	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuse = diff * lightColor;
	// specular
	float specularStrength = 0.5;
	vec3 viewDir = normalize(viewPos - FragPos);
	vec3 reflectDir = reflect(-lightDir, norm);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
	vec3 specular = specularStrength * spec * lightColor;
	vec3 result;
	if (applyIllumination)
		result = (ambient + diffuse + specular) * objectColor;
	else
		result = objectColor;
		
	 vec3 trueNormal = (vec4(NormalSSAO, 1)).xyz;
	vec3 v_light_direction = normalize(light_position);
	float diffuse2 = clamp(dot(trueNormal, v_light_direction), 0.0, 1.0);
	vec4 color = vec4((0.5 + 0.5*diffuse2)*objectColor, 1.0);
	
	gAlbedo = result;
	gPosition = (view * vec4(FragPos, 1)).xyz;
	gNormal = mat3(transpose(inverse(view))) * NormalSSAO;//(view * vec4(NormalSSAO, 1)).xyz;
}