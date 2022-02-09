#version 330 core

layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec3 gAlbedo;
layout (location = 3) out vec4 gInfo;
layout (location = 4) out float gPick;

uniform mat4 view;
uniform mat4 model;
uniform mat4 MVP;
uniform float radius;

uniform sampler1D lutTexture;
uniform float minFeatureValue;
uniform float maxFeatureValue;
uniform float nbPoints;

uniform bool useSpecialColors;
uniform vec3 light_position;

in vec2 TexCoords_GS;
in vec3 center;
in float vfeature;
in vec3 vcolor;

// 'colorImage' is a sampler2D with the depth image
// read from the current depth buffer bound to it.
//
float LinearizeDepth(in float depth)
{
    float zNear = gl_DepthRange.near;    // TODO: Replace by the zNear of your perspective projection
    float zFar  = gl_DepthRange.far; // TODO: Replace by the zFar  of your perspective projection
    return (2.0 * zNear) / (zFar + zNear - depth * (zFar - zNear));
}

void main()
{
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

	vec2 texC =  ( TexCoords_GS - 0.5 ) * 2;//vec2(1./0.5, 1./0.5);
    float d = dot(texC, texC);
    if (d > 1.0f)
    {
        discard;
    }

    float z = sqrt(1.0f - d);

    vec3 normal = vec3(texC, z);
    vec3 trueNormal = mat3(transpose(view)) * normal;
	
	vec3 v_light_direction = normalize(light_position);
	float diffuse = clamp(dot(trueNormal, v_light_direction), 0.0, 1.0);
	vec4 color = vec4((0.5 + 0.5*diffuse)*colorTmp, 1.0);

    vec3 position = center + radius * trueNormal;
	vec4 clipPos = MVP * vec4(position, 1.0f);
    float ndcDepth = clipPos.z / clipPos.w;
    gPosition = ((view * model) * vec4(position, 1.0f)).xyz;
    gl_FragDepth = ((gl_DepthRange.diff * ndcDepth) + gl_DepthRange.near + gl_DepthRange.far) * 0.5f;
	float depth = LinearizeDepth(gl_FragDepth);
    gNormal = normal;
	gAlbedo.rgb = color.rgb;
	
	float id = gl_PrimitiveID / nbPoints;
	gInfo = vec4(id, id, d, 1);
	gPick = gl_PrimitiveID;
}