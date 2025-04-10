#version 330 core
uniform mat4 view;
uniform mat4 MVP;
uniform float radius;

uniform sampler1D lutTexture;
uniform float minFeatureValue;
uniform float maxFeatureValue;
uniform float nbPoints;

uniform bool useSpecialColors;
uniform vec3 light_position;
uniform bool clip;

uniform bool activatedCulling;
uniform vec3 cameraForward;

in vec2 texC;
in vec3 center;
in float vfeature;
in vec3 vcolor;
in vec3 vnormal;
in float vclipDistance;

out vec4 color_out;

vec4 outline(float distance, float linewidth, float antialias, vec4 stroke, vec4 fill)
{
    vec4 frag_color;
    float t = linewidth/2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);
    if( border_distance < 0.0 )
        frag_color = stroke;
    else if( signed_distance < 0.0 )
        frag_color = mix(fill, stroke, sqrt(alpha));
    else
        frag_color = vec4(stroke.rgb, stroke.a * alpha);
    return frag_color;
}

void main()
{
	if(clip && vclipDistance < 0.f)
		discard;
		
	if(activatedCulling){
		float res = dot(cameraForward, vnormal);
		if(res < 0.f)
			discard;
	}
	
	vec2 P = gl_PointCoord.xy - vec2(0.5,0.5);
    float point_size = radius  + 5.0;
    float distance = length(P*point_size) - radius/2;
	
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
	
	vec3 v_light_direction = normalize(light_position);
	float diffuse = clamp(dot(trueNormal, v_light_direction), 0.0, 1.0);
	
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
	vec4 color = vec4((0.5 + 0.5*diffuse)*colorTmp, 1.0);
	colorTmp += 0.2;
    color_out = outline(distance, 1.0, 1.0, vec4(0,0,0,1), color);
    color_out = color;
	//color_out = vec4(colorTmp, 1.0);
};