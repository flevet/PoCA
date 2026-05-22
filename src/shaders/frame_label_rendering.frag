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

/*void main()
{
	a_colour = vec4(1,0, 0, 1);
}*/

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

const int MAX_CLIPPING_PLANES = 6;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform bool clip;

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
uniform int currentFrame;

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
uniform float featureTextureSize;

uniform vec3 scale;

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

bool isBorderVoxel(vec3 position, uint label, int radius) {
    if(!borderRendering)
	return true;
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

bool sampleFeatureValue(vec3 position, out float featureValue)
{
    float intensity;
    if (isFloat)
        intensity = texture(volume, position).r;
    else
        intensity = float(texture(uvolume, position).r);

    if (intensity <= 0)
        return false;

    float x = intensity, y = 0.0;
    if (height_feature_texture == 1.0) {
        x = (intensity - pixel_min) / max(pixel_max - pixel_min, 1e-6);
    }
    else {
        offset_feature_texture(intensity, width_feature_texture, height_feature_texture, x, y);
        y = scaleOffsetVar(height_feature_texture, y);
    }
    x = scaleOffsetVar(width_feature_texture, x);
    featureValue = texture(featureTexture, vec2(x, y)).r;
    if(featureValue < current_min || featureValue > current_max)
	return false;
    if (!isBorderVoxel(position, uint(intensity), int(borderSize)))
	return false;
    return true;
}

vec4 computeColor(float value)
{
    if (scaleLUT)
        value = clamp(value, current_min, current_max);

    if (!applyThreshold && !scaleLUT && (value < current_min || value > current_max))
        return vec4(0.0);

    if (applyThreshold && value > current_min && value < current_max)
        return vec4(1.0, 0.0, 0.0, 1.0);

    value = (value - feature_min) / max(feature_max - feature_min, 1e-6);
    float posLut = scaleOffsetVar(512.0, value);

    return vec4(texture(lutTexture, posLut).xyz, 1.0);
}

bool clippedByPlane(vec3 worldPos)
{
    if (!clip)
        return false;
    vec4 pos = vec4(worldPos, 1.0);
    for (int n = 0; n < MAX_CLIPPING_PLANES; ++n)
        if (dot(pos, clipPlanes[n]) < 0.0)
            return true;
    return false;
}

void main()
{
    vec4 ndcPos;
    ndcPos.xy = ((2.0 * gl_FragCoord.xy) - (2.0 * viewport.xy)) / viewport.zw - 1.0;
    ndcPos.z = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far) / (gl_DepthRange.far - gl_DepthRange.near);
    ndcPos.w = 1.0;

    vec4 clipPos = ndcPos;
    clipPos.z = -1.0;
    vec4 eyePos = invMVP * clipPos;
    eyePos /= eyePos.w;
    vec3 ray_origin = eyePos.xyz;
    vec3 current_ray_direction = perspective_projection ? normalize(ray_origin - camera_position) : ray_direction;
    vec3 current_ray_origin = perspective_projection ? camera_position : ray_origin + current_ray_direction;
    Ray casting_ray = Ray(current_ray_origin, current_ray_direction);

    float planeZ = float(currentFrame) + 0.5;
    if (abs(casting_ray.direction.z) < 1e-6)
        discard;

    float t = (planeZ - casting_ray.origin.z) / casting_ray.direction.z;
    if (t < 0.0)
        discard;

    vec3 worldPos = casting_ray.origin + t * casting_ray.direction;
    if (any(lessThan(worldPos, bottom)) || any(greaterThan(worldPos, top)))
        discard;

    if (clippedByPlane(worldPos))
        discard;

    if (cropped && (any(lessThan(worldPos, bottom_crop)) || any(greaterThan(worldPos, top_crop))))
        discard;

    if (clippedByPlane(worldPos))
        discard;

    vec3 position = (worldPos - bottom) / max(top - bottom, vec3(1e-6));
    position.z = (float(currentFrame) + 0.5 - bottom.z) / max(top.z - bottom.z, 1e-6);
    position = clamp(position, vec3(0.0), vec3(1.0));

    float value;
    if (!sampleFeatureValue(position, value))
        discard;

    if(!applyThreshold && !scaleLUT && (value < current_min || value > current_max))
		discard;
    
    worldPos = position * (top - bottom) + bottom;
    // Compute depth at this sample position
    vec4 clipSpacePos = ModelViewProjectionMatrix * vec4(worldPos, 1.0);
    float ndcDepth = clipSpacePos.z / clipSpacePos.w;
    gl_FragDepth = 0.5 * ndcDepth + 0.5; // Convert NDC [-1,1] to [0,1]

    a_colour = computeColor(value);
}
