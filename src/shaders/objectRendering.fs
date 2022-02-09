#version 330 core
in float feature;
in vec3 Normal;
in vec3 NormalSSAO;
in vec3 FragPos;
out vec4 color;
uniform mat4 view;
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
	vec3 lightDir = normalize(light_position);// normalize(lightPos - FragPos);
	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuse = diff * lightColor;
	// specular
	float specularStrength = 0.5;
	vec3 viewDir = normalize(light_position);//normalize(viewPos - FragPos);
	vec3 reflectDir = reflect(-lightDir, norm);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
	vec3 specular = specularStrength * spec * lightColor;
	vec3 result;
	if (applyIllumination)
		result = (ambient + diffuse + specular) * objectColor;
	else
		result = objectColor;
	color = vec4(result, 0.4);
}