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
uniform bool isFrame[MAX_NB_IMAGES];

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

uniform vec3 scale;

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
    float scaleValue = (texturesize - 1.0) / texturesize;
    float offset = 1.0 / (2.0 * texturesize);
    return scaleValue * pos + offset;
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

bool sampleFeatureValue(int imageIndex, vec3 position, out float featureValue)
{
    float intensity;
    if (isFloat[imageIndex]) {
        intensity = texture(volume[imageIndex], position).r;
    }
    else {
        usampler3D tex = uvolume[imageIndex];
        intensity = float(texture(tex, position).r);
    }

    if (intensity < pixel_min[imageIndex])
        return false;

    float x = intensity, y = 0.0;
    if (height_feature_texture[imageIndex] == 1.0) {
        x = (intensity - pixel_min[imageIndex]) / max(pixel_max[imageIndex] - pixel_min[imageIndex], 1e-6);
    }
    else {
        offset_feature_texture(intensity, width_feature_texture[imageIndex], height_feature_texture[imageIndex], x, y);
        y = scaleOffsetVar(height_feature_texture[imageIndex], y);
    }
    x = scaleOffsetVar(width_feature_texture[imageIndex], x);
    featureValue = texture(featureTexture[imageIndex], vec2(x, y)).r;
    return true;
}

vec4 computeSampleColor(int imageIndex, float featureValue)
{
    if (scaleLUT[imageIndex])
        featureValue = clamp(featureValue, current_min[imageIndex], current_max[imageIndex]);

    if (!applyThreshold[imageIndex] && !scaleLUT[imageIndex] &&
        (featureValue < current_min[imageIndex] || featureValue > current_max[imageIndex]))
        return vec4(0.0);

    if (applyThreshold[imageIndex]) {
        if (featureValue > current_min[imageIndex] && featureValue < current_max[imageIndex])
            return vec4(1.0, 0.0, 0.0, 1.0);
        return vec4(0.0);
    }

    float normalizedValue = (featureValue - feature_min[imageIndex]) / max(feature_max[imageIndex] - feature_min[imageIndex], 1e-6);
    normalizedValue = clamp(normalizedValue, 0.0, 1.0);

    float lutPos = scaleOffsetVar(512.0, normalizedValue);
    if (isLabel[imageIndex])
        return vec4(texture(lutTexture[imageIndex], lutPos).xyz, 1.0);

    float alpha = (exp(normalizedValue) - 1.0) / (exp(1.0) - 1.0);
    vec3 rgb = texture(lutTexture[imageIndex], lutPos).xyz;
    rgb = pow(rgb, vec3(1.0 / gamma));
    return vec4(rgb, alpha);
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

    float t_0, t_1, t_0_crop, t_1_crop;
    Ray casting_ray = Ray(current_ray_origin, current_ray_direction);
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

    vec3 ray_start = (current_ray_origin + current_ray_direction * t_0 - bottom) / (top - bottom);
    vec3 ray_stop = (current_ray_origin + current_ray_direction * t_1 - bottom) / (top - bottom);
    vec3 ray_step = (ray_stop - ray_start) / float(nb_steps);
    vec3 position = ray_start;

    vec4 accum = vec4(0.0);
    for (int step = 0; step < nb_steps && accum.a < 0.999; ++step) {
        position += ray_step;
        for (int curImage = 0; curImage < nbImages && accum.a < 0.999; ++curImage) {
            float featureValue;
            if (!sampleFeatureValue(curImage, position, featureValue))
                continue;

            vec4 sampleColor = computeSampleColor(curImage, featureValue);
            if (sampleColor.a <= 0.0)
                continue;

            accum.rgb += (1.0 - accum.a) * sampleColor.a * sampleColor.rgb;
            accum.a += (1.0 - accum.a) * sampleColor.a;
        }
    }

    if (accum.a <= 0.0)
        discard;

    vec3 background = pow(background_colour, vec3(1.0 / gamma));
    a_colour.rgb = accum.rgb + (1.0 - accum.a) * background;
    a_colour.a = accum.a;
}
