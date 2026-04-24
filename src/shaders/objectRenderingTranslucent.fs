#version 330 core
in float feature;
in vec3 Normal;
in vec3 NormalSSAO;
in vec3 FragPos;
in float vclipDistance;
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
uniform bool clip;
uniform float alpha;

uniform bool applyUniformColor;
uniform vec4 uniformColor;

void main() {
	if (clip && vclipDistance < 0.f)
		discard;

	if (feature < minFeatureValue)
		discard;

	vec3 objectColor;
	float inter = maxFeatureValue - minFeatureValue;
	if (inter > -0.0001 && inter < 0.0001)
		objectColor = texture(lutTexture, 0).xyz;
	else
		objectColor = texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz;

	if (applyUniformColor)
		objectColor = uniformColor.rgb;

	vec3 norm = normalize(gl_FrontFacing ? Normal : -Normal);
	vec3 lightDir = normalize(light_position);
	vec3 viewDir = normalize(light_position);

	float frontDiffuse = max(dot(norm, lightDir), 0.0);
	float backScatter = pow(max(dot(-norm, lightDir), 0.0), 1.5);
	float rim = pow(1.0 - max(dot(norm, viewDir), 0.0), 2.0);

	float specularStrength = 0.15;
	vec3 reflectDir = reflect(-lightDir, norm);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 24.0);
	vec3 specular = specularStrength * spec * lightColor;

	vec3 result;
	if (applyIllumination) {
		float ambient = 0.18;
		float translucent = 0.55 * backScatter + 0.25 * rim;
		float diffuse = 0.55 * frontDiffuse;
		result = (ambient + diffuse + translucent) * objectColor + specular;
	}
	else {
		result = objectColor;
	}

	float finalAlpha = clamp(alpha * (0.35 + 0.45 * backScatter + 0.20 * rim), 0.02, 1.0);
	color = vec4(result, finalAlpha);
}
