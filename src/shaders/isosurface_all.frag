#version 440 core
#extension GL_ARB_bindless_texture : require

out vec4 a_colour;

uniform mat4 ModelViewProjectionMatrix;
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

float sampleRawIntensity(int imageIndex, vec3 position)
{
    position = clamp(position, vec3(0.0), vec3(1.0));
    if (isFloat[imageIndex])
        return texture(volume[imageIndex], position).r;

    usampler3D tex = uvolume[imageIndex];
    ivec3 tsize = textureSize(tex, 0);
    if (tsize.z == 1)
        tsize.z = 0;
    ivec3 texPos = ivec3(position * vec3(tsize));
    return float(texelFetch(tex, texPos, 0).r);
}

bool sampleFeatureValue(int imageIndex, vec3 position, out float featureValue)
{
    float intensity = sampleRawIntensity(imageIndex, position);
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

bool isSurfaceHit(int imageIndex, float featureValue)
{
    return featureValue > current_min[imageIndex] && featureValue < current_max[imageIndex];
}

vec3 gradientAt(int imageIndex, vec3 position)
{
    vec3 texel = vec3(
        1.0 / max(pixel_max[imageIndex] - pixel_min[imageIndex], 1.0),
        1.0 / max(pixel_max[imageIndex] - pixel_min[imageIndex], 1.0),
        1.0 / max(pixel_max[imageIndex] - pixel_min[imageIndex], 1.0));
    vec3 e = max(vec3(1.0 / 256.0), texel);

    float center = sampleRawIntensity(imageIndex, position);
    float dx = sampleRawIntensity(imageIndex, position + vec3(e.x, 0.0, 0.0)) - center;
    float dy = sampleRawIntensity(imageIndex, position + vec3(0.0, e.y, 0.0)) - center;
    float dz = sampleRawIntensity(imageIndex, position + vec3(0.0, 0.0, e.z)) - center;
    return normalize(-vec3(dx, dy, dz));
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
    float step_length = length(ray) / float(nb_steps);
    vec3 position = ray_start;

    bool found = false;
    int hitImage = -1;
    vec3 hitPosition = vec3(0.0);
    float hitDepth = 1.0;

    for (int step = 0; step < nb_steps && !found; ++step) {
        position += ray_step;
        for (int curImage = 0; curImage < nbImages && !found; ++curImage) {
            float featureValue;
            if (!sampleFeatureValue(curImage, position, featureValue))
                continue;
            if (!isSurfaceHit(curImage, featureValue))
                continue;

            hitImage = curImage;
            hitPosition = position;
            vec4 clipSpacePos = ModelViewProjectionMatrix * vec4(mix(bottom, top, position), 1.0);
            float ndcDepth = clipSpacePos.z / clipSpacePos.w;
            hitDepth = 0.5 * ndcDepth + 0.5;
            found = true;
        }
    }

    if (!found || hitImage < 0)
        discard;

    vec3 N = gradientAt(hitImage, hitPosition);
    vec3 worldPos = mix(bottom, top, hitPosition);
    vec3 L = normalize(light_position - worldPos);
    vec3 V = -normalize(ray_direction);
    vec3 H = normalize(L + V);

    float Ia = 0.1;
    float Id = max(0.0, dot(N, L));
    float Is = 8.0 * pow(max(0.0, dot(N, H)), 120.0);

    vec3 colour = (Ia + Id) * material_colour + Is * vec3(1.0);
    a_colour = vec4(pow(colour, vec3(1.0 / gamma)), 1.0);
    gl_FragDepth = clamp(hitDepth, 0.0, 1.0);
}
