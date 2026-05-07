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

#version 440 core
#extension GL_ARB_bindless_texture : require

out vec4 a_colour;

uniform mat4 ModelViewProjectionMatrix;
uniform mat4 invMVP;

uniform vec4 viewport;
uniform vec3 ray_direction;
uniform vec3 camera_position;
uniform bool perspective_projection;
uniform vec3 top;
uniform vec3 bottom;

uniform bool cropped;
uniform vec3 top_crop;
uniform vec3 bottom_crop;

uniform vec3 background_colour;
uniform vec3 material_colour;
uniform vec3 light_position;

const int MAX_NB_IMAGES = 16;
uniform int nbImages;

uniform int nb_steps;
uniform bool applyThreshold[MAX_NB_IMAGES];
uniform bool isFloat[MAX_NB_IMAGES];
uniform bool isLabel[MAX_NB_IMAGES];
uniform bool scaleLUT[MAX_NB_IMAGES];

uniform float width_feature_texture[MAX_NB_IMAGES];
uniform float height_feature_texture[MAX_NB_IMAGES];

uniform sampler1D lutTexture[MAX_NB_IMAGES];
uniform sampler2D featureTexture[MAX_NB_IMAGES];

uniform sampler3D volume[MAX_NB_IMAGES];
uniform usampler3D uvolume[MAX_NB_IMAGES];

uniform float gamma;
uniform float pixel_min[MAX_NB_IMAGES];
uniform float pixel_max[MAX_NB_IMAGES];
uniform float feature_min[MAX_NB_IMAGES];
uniform float feature_max[MAX_NB_IMAGES];
uniform float current_min[MAX_NB_IMAGES];
uniform float current_max[MAX_NB_IMAGES];
uniform float labelBackground[MAX_NB_IMAGES];
uniform float featureTextureSize[MAX_NB_IMAGES];

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

void offset_feature_texture(float label_id, float w, float h, out float x, out float y){
	float id = label_id - 1;
	y = floor(id / w) / (h - 1);
	x = (id - (y * w)) / (w - 1);
}

float scaleOffsetVar(float texturesize, float pos){
	float scale = (texturesize - 1.0) / texturesize;
	float offset = 1.0 / (2.0 * texturesize);
	return scale * pos + offset;
}

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

void main()
{
	vec4 ndcPos;
	ndcPos.xy = ((2.0 * gl_FragCoord.xy) - (2.0 * viewport.xy)) / (viewport.zw) - 1;
	ndcPos.z = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far) / (gl_DepthRange.far - gl_DepthRange.near);
	ndcPos.w = 1.0;
 
	vec4 clipPos = ndcPos;
	clipPos.z = -1.0;
	vec4 eyePos  = invMVP * clipPos;
	eyePos /= eyePos.w;
	vec3 ray_origin = eyePos.xyz;
	vec3 current_ray_direction = perspective_projection ? normalize(ray_origin - camera_position) : ray_direction;
	vec3 current_ray_origin = perspective_projection ? camera_position : ray_origin + current_ray_direction;
	
    float t_0, t_1, t_0_crop, t_1_crop;
    Ray casting_ray = Ray(current_ray_origin, current_ray_direction);
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
	
    vec3 ray_start = (current_ray_origin + current_ray_direction * t_0 - bottom) / (top - bottom);
    vec3 ray_stop = (current_ray_origin + current_ray_direction * t_1 - bottom) / (top - bottom);
	
	//vec3 ray_stop = (ray_origin + ray_direction * t_0 - bottom) / (top - bottom);
    //vec3 ray_start = (ray_origin + ray_direction * t_1 - bottom) / (top - bottom);
	
    vec3 ray = ray_stop - ray_start;
    float ray_length = length(ray);
	vec3 ray_step = ray / float(nb_steps);
	
    vec3 position = ray_start;
	float intensity[MAX_NB_IMAGES];
	bool found = false;
	float maxDepth;
	for(int n = 0; n < nbImages; n++){
		intensity[n] = -3.402823466e+38;
		//found[n] = false;
	}

	for(int n = 0; n < nb_steps && !found; n++){
		position = position + ray_step;
		for(int curImage = 0; curImage < nbImages; curImage++){
			//if(!found[curImage]){
				float intensityTex;
				if(isFloat[curImage])
					intensityTex = texture(volume[curImage], position).r;
				else
					intensityTex = float(texture(uvolume[curImage], position).r);
				
				if(intensityTex >= current_min[curImage] && intensityTex <= current_max[curImage]){
					//we retrieve the true pixel value from the pixel
					//We need to normalize it in order to fetch the lookup table from featureTexture
					float x = intensityTex, y = 0;
					if(height_feature_texture[curImage] == 1){
						x = (intensityTex - pixel_min[curImage]) / (pixel_max[curImage] - pixel_min[curImage]);
					}
					else{
						offset_feature_texture(intensityTex, width_feature_texture[curImage], height_feature_texture[curImage], x, y);
						y = scaleOffsetVar(height_feature_texture[curImage], y);
					}
					x = scaleOffsetVar(width_feature_texture[curImage], x);
					intensity[curImage] = texture(featureTexture[curImage], vec2(x, y)).r;

					vec3 worldPos = position * (top - bottom) + bottom;
					// Compute depth at this sample position
					vec4 clipSpacePos = ModelViewProjectionMatrix * vec4(worldPos, 1.0);
					float ndcDepth = clipSpacePos.z / clipSpacePos.w;
					maxDepth = 0.5 * ndcDepth + 0.5; // Convert NDC [-1,1] to [0,1]

					found = true;
				}
			//}
		}
	}

	//bool atLeastOne = false;
	//for(int curImage = 0; curImage < nbImages; curImage++)
	//	atLeastOne = atLeastOne || found[curImage];
	if(!found)
		discard;

	a_colour = vec4(0.0);
	vec4 currentColor;
	for(int curImage = 0; curImage < nbImages; curImage++){
		if(scaleLUT[curImage]){
			if(intensity[curImage] < current_min[curImage]) intensity[curImage] = current_min[curImage];
			if(intensity[curImage] > current_max[curImage]) intensity[curImage] = current_max[curImage];
		}
		
		if(!applyThreshold[curImage] && !scaleLUT[curImage] && (intensity[curImage] < current_min[curImage] || intensity[curImage] > current_max[curImage]))
			continue;

		if(applyThreshold[curImage] && intensity[curImage] > current_min[curImage] && intensity[curImage] < current_max[curImage]){
			currentColor = vec4(1, 0, 0, 1);
		}
		else{
			//And we need to normalize a second time to fetch the correct color in lutTexture
			intensity[curImage] = (intensity[curImage] - feature_min[curImage]) / (feature_max[curImage] - feature_min[curImage]);
			
			if(isLabel[curImage]){
				float posLut = scaleOffsetVar(512, intensity[curImage]);
				currentColor.rgb = texture(lutTexture[curImage], posLut).xyz;
				currentColor.a = 1.0;
			}
			else{
				vec4 colour = vec4(intensity[curImage], intensity[curImage], intensity[curImage], (exp(intensity[curImage]) - 1.0) / (exp(1.0) - 1.0));

				// Blend background
				colour.rgb = colour.a * colour.rgb + (1 - colour.a) * pow(background_colour, vec3(gamma)).rgb;
				float posLut = scaleOffsetVar(512, colour.r);
				colour.rgb = texture(lutTexture[curImage], posLut).xyz;
				colour.a = 1.0;

				// Gamma correction
				currentColor.rgb = pow(colour.rgb, vec3(1.0 / gamma));
				currentColor.a = colour.a;
			}
		}
		a_colour = a_colour + currentColor;
		//a_colour = a_colour + vec4(intensity[curImage], intensity[curImage], intensity[curImage], 1);
	}
	gl_FragDepth = maxDepth;
	a_colour = clamp(a_colour, 0.0, 1.0);
}
