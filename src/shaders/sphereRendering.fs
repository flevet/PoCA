#version 330 core
in float feature;
in vec3 v_color;
in float v_size;
in vec4 v_eye_position;
in vec3 v_light_direction;
out vec4 color_out;
uniform sampler1D lutTexture;
uniform float minFeatureValue;
uniform float maxFeatureValue;
uniform bool useSpecialColors;
uniform float radius;
uniform mat4 projection;

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

void main() {
	vec2 P = gl_PointCoord.xy - vec2(0.5,0.5);
    float point_size = v_size  + 5.0;
    float distance = length(P*point_size) - v_size/2;
    vec2 texcoord = gl_PointCoord* 2.0 - vec2(1.0);
    float x = texcoord.x;
    float y = texcoord.y;
    float d = 1.0 - x*x - y*y;
    //if (d <= 0.0) discard;
    float z = sqrt(d);
    vec4 pos = v_eye_position;
    pos.z += radius*z;
    vec3 pos2 = pos.xyz;
    pos = projection * pos;
    gl_FragDepth = 0.5*(pos.z / pos.w)+0.5;
   vec3 normal = vec3(x,y,z);
    float diffuse = clamp(dot(normal, v_light_direction), 0.0, 1.0);
	
	vec3 colorTmp;
	if (useSpecialColors) {
		colorTmp = vec3(v_color);
	}
	else {
		if (feature < minFeatureValue)
			discard;
		float inter = maxFeatureValue - minFeatureValue;
		colorTmp = vec3(texture(lutTexture, ((feature - minFeatureValue) / inter)).xyz);
	}
	
    vec4 color = vec4((0.5 + 0.5*diffuse)*colorTmp, 1.0);
	color_out = outline(distance, 1.0, 1.0, vec4(0,0,0,1), color);
    //color_out = color;
}