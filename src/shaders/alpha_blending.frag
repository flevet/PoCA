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

const int MAX_CLIPPING_PLANES = 6;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform bool clip;

uniform vec3 background_colour;

uniform int nb_steps;
uniform bool applyThreshold;
uniform bool isFloat;
uniform bool isLabel;
uniform bool scaleLUT;

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

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct AABB {
    vec3 top;
    vec3 bottom;
};

void offset_feature_texture(float label_id, float w, float h, out float x, out float y)
{
    float id = label_id - 1.0;
    y = floor(id / w) / (h - 1.0);
    x = (id - (y * w)) / (w - 1.0);
}

float scaleOffsetVar(float texturesize, float pos)
{
    float scale = (texturesize - 1.0) / texturesize;
    float offset = 1.0 / (2.0 * texturesize);
    return scale * pos + offset;
}

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

bool sampleFeatureValue(vec3 position, out float featureValue)
{
    float intensity;
    if (isFloat)
        intensity = texture(volume, position).r;
    else
        intensity = float(texture(uvolume, position).r);

    if (intensity < pixel_min)
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
    featureValue = texture2D(featureTexture, vec2(x, y)).r;
    return true;
}

vec4 computeSampleColor(float featureValue)
{
    if (scaleLUT)
        featureValue = clamp(featureValue, current_min, current_max);

    if (!applyThreshold && !scaleLUT && (featureValue < current_min || featureValue > current_max))
        return vec4(0.0);

    if (applyThreshold) {
        if (featureValue > current_min && featureValue < current_max)
            return vec4(1.0, 0.0, 0.0, 1.0);
        return vec4(0.0);
    }

    float normalizedValue = (featureValue - feature_min) / max(feature_max - feature_min, 1e-6);
    normalizedValue = clamp(normalizedValue, 0.0, 1.0);

    float lutPos = scaleOffsetVar(512.0, normalizedValue);
    if (isLabel)
        return vec4(texture1D(lutTexture, lutPos).xyz, 1.0);

    float alpha = (exp(normalizedValue) - 1.0) / (exp(1.0) - 1.0);
    vec3 rgb = texture1D(lutTexture, lutPos).xyz;
    rgb = pow(rgb, vec3(1.0 / gamma));
    return vec4(rgb, alpha);
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
    vec3 ray_origin = eyePos.xyz;

    float t_0, t_1, t_0_crop, t_1_crop;
    Ray casting_ray = Ray(ray_origin + ray_direction, ray_direction);
    AABB bounding_box = AABB(top, bottom);
    ray_box_intersection(casting_ray, bounding_box, t_0, t_1);

    if (cropped) {
        AABB crop_bbox = AABB(top_crop, bottom_crop);
        ray_box_intersection(casting_ray, crop_bbox, t_0_crop, t_1_crop);
        if (t_0_crop > t_1_crop)
            discard;
        t_0 = t_0_crop;
        t_1 = t_1_crop;
    }

    vec3 ray_start = (ray_origin + ray_direction * t_0 - bottom) / (top - bottom);
    vec3 ray_stop = (ray_origin + ray_direction * t_1 - bottom) / (top - bottom);
    vec3 ray = ray_stop - ray_start;
    vec3 ray_step = ray / float(nb_steps);

    vec3 position = ray_start;
    vec4 accum = vec4(0.0);

    for (int step = 0; step < nb_steps && accum.a < 0.999; ++step) {
        position += ray_step;
        vec3 worldPos = position * (top - bottom) + bottom;
        if (clippedByPlane(worldPos))
            continue;
        float featureValue;
        if (!sampleFeatureValue(position, featureValue))
            continue;

        vec4 sampleColor = computeSampleColor(featureValue);
        if (sampleColor.a <= 0.0)
            continue;

        accum.rgb += (1.0 - accum.a) * sampleColor.a * sampleColor.rgb;
        accum.a += (1.0 - accum.a) * sampleColor.a;
    }

    if (accum.a <= 0.0)
        discard;

    vec3 background = pow(background_colour, vec3(1.0 / gamma));
    a_colour.rgb = accum.rgb + (1.0 - accum.a) * background;
    a_colour.a = accum.a;
}
