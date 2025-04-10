#version 330 core
uniform mat4 view;
uniform mat4 MVP;
uniform vec4 singleColor;
uniform vec3 bboxPt1;
uniform vec3 bboxPt2;
struct Light {
	vec3 Position;
	vec3 Color;
	float Linear;
	float Quadratic;
};
uniform Light light;
uniform bool activatedCulling;
uniform vec3 cameraForward;
uniform bool clip;

in vec2 texC;
in vec3 center;
in float radius;
in vec3 normal;
in float vclipDistance;

out vec4 color;

void main()
{
	if(clip && vclipDistance < 0.f)
		discard;
		
	if(activatedCulling){
		float res = dot(cameraForward, normal);
		if(res < 0.f)
			discard;
	}
	
	float d = dot(texC, texC);
	if (d > 1.0f)
		discard;
	float z = sqrt(1.0f - d);
	vec3 Normal = normalize(vec3(texC, z));
	vec3 trueNormal = normalize(mat3(transpose(view)) * Normal);
	vec3 position = center + radius * trueNormal;
	vec3 FragPos = (view * vec4(position, 1.0f)).xyz;
	
	vec4 clipPos = MVP * vec4(position, 1.0f);
	float ndcDepth = clipPos.z / clipPos.w;
	gl_FragDepth = ((gl_DepthRange.diff * ndcDepth) + gl_DepthRange.near + gl_DepthRange.far) / 2.f;
	
    vec3 lighting = singleColor.rgb;
    float distance = d * 5;//radius*3;//length(light.Position - FragPos);
	float attenuation = 1.0 / (1.0 + light.Linear * distance + light.Quadratic * distance * distance);
	lighting *= attenuation;
	color = vec4(lighting, 0.6);//vec4(0,0,0, 0.6);
	//color = vec4((trueNormal + 1.0f) * 0.5f, 1.0f);//display normal
	/*vec3 Diffuse = singleColor.rgb;
	// then calculate lighting as usual
	vec3 lighting = vec3(0.6 * Diffuse);
	vec3 viewDir = normalize(vec3(0, 0, 1000.f) - FragPos); // viewpos is (0.0.0)
	// diffuse
	vec3 lightDir = normalize(vec3(0, 0, 1));// normalize(light.Position - FragPos);
	vec3 diffuse = max(dot(Normal, lightDir), 0.0) * Diffuse;// light.Color;
	// specular
	vec3 halfwayDir = normalize(lightDir + viewDir);
	float spec = pow(max(dot(Normal, halfwayDir), 0.0), 8.0);
	vec3 specular = light.Color * spec;
	// attenuation
	float distance = 5;//radius*3;//length(light.Position - FragPos);
	float attenuation = 1.0 / (1.0 + light.Linear * distance + light.Quadratic * distance * distance);
	diffuse *= attenuation;
	specular *= attenuation;
	lighting += diffuse + specular;
	color = vec4(lighting, 0.2);*/
};