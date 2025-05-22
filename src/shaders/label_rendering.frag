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
uniform bool isFloat;
uniform bool isLabel;
uniform bool scaleLUT;
uniform bool isFrame;
uniform bool borderRendering;
uniform uint borderSize;

uniform float width_feature_texture;
uniform float height_feature_texture;

uniform sampler1D lutTexture;
uniform sampler2D featureTexture;

uniform sampler3D volume;
uniform usampler3D uvolume;

uniform float gamma;
uniform float pixel_min;
uniform float pixel_max;
uniform float feature_min;
uniform float feature_max;
uniform float current_min;
uniform float current_max;
uniform float labelBackground;
uniform float featureTextureSize;

const vec3 OFFSETS[4] = vec3[](
    vec3(1, 0, 0), vec3(-1, 0, 0),
    vec3(0, 1, 0), vec3(0, -1, 0)
);

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

bool isBorderVoxel(vec3 position, uint label, int radius) {
    vec3 volumeDims = vec3(textureSize(uvolume, 0)); // voxel grid dimensions
    vec3 texelSize = 1.0 / volumeDims;

    for (int x = -radius; x <= radius; ++x) {
        for (int y = -radius; y <= radius; ++y) {
            if (x == 0 && y == 0) continue; // skip center voxel

            vec3 offset = vec3(x, y, 0) * texelSize;
            vec3 neighborPos = position + offset;

            // Skip out-of-bounds neighbors
            if (any(lessThan(neighborPos, vec3(0.0))) || any(greaterThanEqual(neighborPos, vec3(1.0))))
                continue;

            uint neighborLabel = texture(uvolume, neighborPos).r;
            if (neighborLabel != label)
                return true;
        }
    }
    return false;
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
	
    float t_0, t_1, t_0_crop, t_1_crop;
    Ray casting_ray = Ray(ray_origin, ray_direction);
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
	
    vec3 ray_start = (ray_origin + ray_direction * t_0 - bottom) / (top - bottom);
    vec3 ray_stop = (ray_origin + ray_direction * t_1 - bottom) / (top - bottom);
	
	//vec3 ray_stop = (ray_origin + ray_direction * t_0 - bottom) / (top - bottom);
    //vec3 ray_start = (ray_origin + ray_direction * t_1 - bottom) / (top - bottom);
	
    vec3 ray = ray_stop - ray_start;
    float ray_length = length(ray);
	vec3 ray_step = ray / float(nb_steps);

    vec3 position = ray_start;
	float intensity = 0.f;
	
	bool found = false;
    for(int n = 0; n < nb_steps && !found; n++){
		position = position + ray_step;
		if(isFloat)
			intensity = texture(volume, position).r;
		else
			intensity = float(texture(uvolume, position).r);
		
		if(intensity >= current_min && intensity <= current_max){
		
			if(isLabel && borderRendering && isFrame){
				if (!isBorderVoxel(position, uint(intensity), int(borderSize)))
					continue;
			}
			//we retrieve the true pixel value from the pixel
			//We need to normalize it in order to fetch the lookup table from featureTexture
			float x = intensity, y = 0;
			if(height_feature_texture == 1){
				x = (intensity - pixel_min) / (pixel_max - pixel_min);
			}
			else{
				offset_feature_texture(intensity, width_feature_texture, height_feature_texture, x, y);
				y = scaleOffsetVar(height_feature_texture, y);
			}
			x = scaleOffsetVar(width_feature_texture, x);
			intensity = texture(featureTexture, vec2(x, y)).r;
			found = true;
		}
	}
	
	if(!found)
		discard;
	
	if(scaleLUT){
		if(intensity < current_min) intensity = current_min;
		if(intensity > current_max) intensity = current_max;
	}
	
	if(!applyThreshold && !scaleLUT && (intensity < current_min || intensity > current_max))
		discard;

	if(applyThreshold && intensity > current_min && intensity < current_max){
		a_colour = vec4(1, 0, 0, 1);
	}
	else{
		//And we need to normalize a second time to fetch the correct color in lutTexture
		intensity = (intensity - feature_min) / (feature_max - feature_min);
		
		if(isLabel){
			float posLut = scaleOffsetVar(512, intensity);
			a_colour.rgb = texture(lutTexture, posLut).xyz;
			a_colour.a = 1.0;
		}
		else{
			vec4 colour = vec4(intensity, intensity, intensity, (exp(intensity) - 1.0) / (exp(1.0) - 1.0));

			// Blend background
			colour.rgb = colour.a * colour.rgb + (1 - colour.a) * pow(background_colour, vec3(gamma)).rgb;
			float posLut = scaleOffsetVar(512, colour.r);
			colour.rgb = texture(lutTexture, posLut).xyz;
			colour.a = 1.0;

			// Gamma correction
			a_colour.rgb = pow(colour.rgb, vec3(1.0 / gamma));
			a_colour.a = colour.a;
		}
	}
}