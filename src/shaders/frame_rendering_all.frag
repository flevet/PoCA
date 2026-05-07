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

const int MAX_NB_IMAGES = 16;
uniform int nbImages;
uniform int currentFrame;
uniform float gamma;

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

uniform float pixel_min[MAX_NB_IMAGES];
uniform float pixel_max[MAX_NB_IMAGES];
uniform float feature_min[MAX_NB_IMAGES];
uniform float feature_max[MAX_NB_IMAGES];
uniform float current_min[MAX_NB_IMAGES];
uniform float current_max[MAX_NB_IMAGES];

struct Ray {
    vec3 origin;
    vec3 direction;
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

bool sampleFeatureValue(int imageIndex, vec3 position, out float featureValue)
{
    float intensity;
    if (isFloat[imageIndex])
        intensity = texture(volume[imageIndex], position).r;
    else
        intensity = float(texture(uvolume[imageIndex], position).r);

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

vec4 computeColor(int imageIndex, float value)
{
    if (scaleLUT[imageIndex])
        value = clamp(value, current_min[imageIndex], current_max[imageIndex]);

    if (!applyThreshold[imageIndex] && !scaleLUT[imageIndex] && (value < current_min[imageIndex] || value > current_max[imageIndex]))
        return vec4(0.0);

    if (applyThreshold[imageIndex] && value > current_min[imageIndex] && value < current_max[imageIndex])
        return vec4(1.0, 0.0, 0.0, 1.0);

    value = (value - feature_min[imageIndex]) / max(feature_max[imageIndex] - feature_min[imageIndex], 1e-6);
    float posLut = scaleOffsetVar(512.0, value);

    if (isLabel[imageIndex])
        return vec4(texture(lutTexture[imageIndex], posLut).xyz, 1.0);

    vec4 colour = vec4(value, value, value, (exp(value) - 1.0) / (exp(1.0) - 1.0));
    colour.rgb = colour.a * colour.rgb + (1.0 - colour.a) * pow(background_colour, vec3(gamma)).rgb;
    posLut = scaleOffsetVar(512.0, colour.r);
    colour.rgb = texture(lutTexture[imageIndex], posLut).xyz;
    colour.a = 1.0;
    colour.rgb = pow(colour.rgb, vec3(1.0 / gamma));
    return colour;
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

    if (cropped && (any(lessThan(worldPos, bottom_crop)) || any(greaterThan(worldPos, top_crop))))
        discard;

    vec3 position = (worldPos - bottom) / max(top - bottom, vec3(1e-6));
    position.z = (float(currentFrame) + 0.5 - bottom.z) / max(top.z - bottom.z, 1e-6);
    position = clamp(position, vec3(0.0), vec3(1.0));

    a_colour = vec4(0.0);
    bool hasContribution = false;
    for (int curImage = 0; curImage < nbImages; curImage++) {
        float value;
        if (!sampleFeatureValue(curImage, position, value))
            continue;

        vec4 currentColor = computeColor(curImage, value);
        if (currentColor.a > 0.0)
            hasContribution = true;
        a_colour += currentColor;
    }

    if (!hasContribution)
        discard;
    a_colour = clamp(a_colour, 0.0, 1.0);
}
