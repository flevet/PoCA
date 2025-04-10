/*
 * Copyright © 2018 Martino Pilia <martino.pilia@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#version 130

out vec4 a_colour;

uniform mat4 invMVP;

uniform vec4 viewport;
uniform vec3 ray_direction;
uniform vec3 top;
uniform vec3 bottom;

uniform bool cropped;
uniform vec3 top_crop;
uniform vec3 bottom_crop;

uniform vec3 background_colour;
uniform vec3 material_colour;
uniform vec3 light_position;

uniform int nb_steps;
uniform bool applyThreshold;
uniform bool is32bits;

uniform sampler3D volume;
uniform sampler2D jitter;
uniform sampler1D lutTexture;
uniform sampler1D featureTexture;

uniform float gamma;
uniform float histogram_min;
uniform float histogram_max;
uniform float current_min;
uniform float current_max;
uniform float maxValue;

// Ray
struct Ray {
    vec3 origin;
    vec3 direction;
};

// Axis-aligned bounding box
struct AABB {
    vec3 top;
    vec3 bottom;
};

// Slab method for ray-box intersection
void ray_box_intersection(Ray ray, AABB box, out float t_0, out float t_1)
{
    vec3 direction_inv = 1.0 / ray.direction;
    vec3 t_top = direction_inv * (box.top - ray.origin);
    vec3 t_bottom = direction_inv * (box.bottom - ray.origin);
    vec3 t_min = min(t_top, t_bottom);
    vec2 t = max(t_min.xx, t_min.yz);
    t_0 = max(0.0, max(t.x, t.y));
    vec3 t_max = max(t_top, t_bottom);
    t = min(t_max.xx, t_max.yz);
    t_1 = min(t.x, t.y);
}

void test_ray_box_intersection(Ray ray, AABB box, out bool intersected)
{
    vec3 direction_inv = 1.0 / ray.direction;
    vec3 t_top = direction_inv * (box.top - ray.origin);
    vec3 t_bottom = direction_inv * (box.bottom - ray.origin);
    vec3 t_min = min(t_top, t_bottom);
    vec2 t = max(t_min.xx, t_min.yz);
    float t_0 = max(0.0, max(t.x, t.y));
    vec3 t_max = max(t_top, t_bottom);
    t = min(t_max.xx, t_max.yz);
    float t_1 = min(t.x, t.y);
	intersected = t_1 >= t_0;
}

// A very simple colour transfer function
vec4 colour_transfer(float intensity)
{
    vec3 high = vec3(1.0, 1.0, 1.0);
    vec3 low = vec3(0.0, 0.0, 0.0);
    float alpha = (exp(intensity) - 1.0) / (exp(1.0) - 1.0);
    return vec4(intensity * high + (1.0 - intensity) * low, alpha);
}

// A very simple colour transfer function
vec4 my_colour_transfer(float intensity)
{
    /*if(intensity < histogram_min)
		return vec4(0, 0, 0, 1);
	if(intensity > histogram_max)
		return vec4(1, 1, 1, 1);*/
		
	float modified_intensity = intensity;//maxValue;
	modified_intensity = (modified_intensity - histogram_min) / (histogram_max - histogram_min);
	
	//float modified_intensity = (intensity - histogram_min) / ( histogram_max - histogram_min);
	return vec4(modified_intensity, modified_intensity, modified_intensity, (exp(modified_intensity) - 1.0) / (exp(1.0) - 1.0));
}

void main()
{
	vec4 ndcPos;
	ndcPos.xy = ((2.0 * gl_FragCoord.xy) - (2.0 * viewport.xy)) / (viewport.zw) - 1;
	ndcPos.z = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far) / (gl_DepthRange.far - gl_DepthRange.near);
	ndcPos.w = 1.0;
 
	vec4 clipPos = ndcPos;
	clipPos.z = -1.0;
	vec4 eyePos  = invMVP * clipPos;
	vec3 ray_origin = eyePos.xyz;
	
	/*vec4 tmp = vec4(gl_FragCoord.xy, 0, 1);
	tmp.x = (tmp.x - viewport[0]) / viewport[2];
	tmp.y = (tmp.y - viewport[1]) / viewport[3];
	tmp.x = tmp.x * 2 - 1;
	tmp.y = tmp.y * 2 - 1;

	vec4 obj = invMVP * tmp;
	obj /= obj.w;
	vec3 ray_origin = obj.xyz;*/
	
    float t_0, t_1, t_0_crop, t_1_crop;
    Ray casting_ray = Ray(ray_origin + ray_direction, ray_direction);
    AABB bounding_box = AABB(top, bottom);
    ray_box_intersection(casting_ray, bounding_box, t_0, t_1);
	
	if(cropped){
		AABB crop_bbox = AABB(top_crop, bottom_crop);
		ray_box_intersection(casting_ray, crop_bbox, t_0_crop, t_1_crop);
		if(t_0_crop > t_1_crop)
			discard;
		t_0 = t_0_crop;
		t_1 = t_1_crop;
	}
	
	//if(t_0 > t_1)
	//	discard;
	
    vec3 ray_start = (ray_origin + ray_direction * t_0 - bottom) / (top - bottom);
    vec3 ray_stop = (ray_origin + ray_direction * t_1 - bottom) / (top - bottom);
	
	//vec3 ray_stop = (ray_origin + ray_direction * t_0 - bottom) / (top - bottom);
    //vec3 ray_start = (ray_origin + ray_direction * t_1 - bottom) / (top - bottom);
	
    vec3 ray = ray_stop - ray_start;
    float ray_length = length(ray);
	vec3 ray_step = ray / float(nb_steps);
	
    /*vec3 ray_step = ray * step_length;
	int nbs = int(ray_length / step_length);*/

    // Random jitter
	vec2 viewport_size = viewport.zw;
    ray_start += ray_step * texture(jitter, gl_FragCoord.xy / viewport_size).r;

    vec3 position = ray_start;
	
	float maximum_intensity = 0.0;
	
	// Ray march until reaching the end of the volume
    for(int n = 0; n < nb_steps; n++){
		position = position + ray_step;
		float intensity = texture(volume, position).r;
		//texture is not normalized when in 32 bits, normalize it to check the feature texture as a lookup table 
		//if(is32bits)
		//	intensity = (intensity - histogram_min) / (histogram_max - histogram_min);
		float scale = (2 - 1.0) / 2;
		float offset = 1.0 / (2.0 * 2);
		//intensity = texture(featureTexture, scale * intensity + offset).r;
		if (intensity > maximum_intensity) {
            maximum_intensity = intensity;
        }
	}
	
	/*float nbSteps = 1000;
	float total_length = 64;
	vec3 steps = (ray_direction * total_length) / nbSteps;

	
	for(int n = 0; n < nbSteps; n++){
		//vec3 ray_pos = vec3(position.xy, position.z + n);
		position = position - steps;
		float intensity = texture(volume, position).r * 2;
		if (intensity > maximum_intensity) {
            maximum_intensity = intensity;
        }
	}*/
	
	//maximum_intensity = texture(volume, position).r;
	
	vec4 colour = my_colour_transfer(maximum_intensity);
		
		/*float intensity = maximum_intensity * maxValue;
		float inter = histogram_max - histogram_min;
		vec3 color = texture(lutTexture, ((intensity - histogram_min) / inter)).xyz;
		
		vec4(modified_intensity, modified_intensity, modified_intensity, (exp(modified_intensity) - 1.0) / (exp(1.0) - 1.0));*/

		// Blend background
		colour.rgb = colour.a * colour.rgb + (1 - colour.a) * pow(background_colour, vec3(gamma)).rgb;
		/*float intensity = colour.r * maxValue;
		intensity = (intensity - histogram_min) / (histogram_max - histogram_min);
		colour.rgb = texture(lutTexture, intensity).xyz;*/
		colour.rgb = texture(lutTexture, colour.r).xyz;
		colour.a = 1.0;

		// Gamma correction
		a_colour.rgb = pow(colour.rgb, vec3(1.0 / gamma));
		a_colour.a = colour.a;
		
		maximum_intensity = maximum_intensity / 4294967295;
		a_colour = vec4(maximum_intensity, maximum_intensity, maximum_intensity, 1);
}
