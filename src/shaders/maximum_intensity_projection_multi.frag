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
uniform int nbImages;
uniform int nb_steps;
uniform float gamma;

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct AABB {
    vec3 top;
    vec3 bottom;
};

struct ImageDescriptor {
    vec4 bottom;
    vec4 top;
    mat4 invModel;
    vec4 pixelParams;
    vec4 featureParams;
    vec4 featureDims;
    uvec4 flags;
    uvec4 lutHandle;
    uvec4 featureHandle;
    uvec4 volumeHandle;
};

layout(std430, binding = 0) readonly buffer ImageDescriptorBuffer {
    ImageDescriptor images[];
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

bool sampleFeatureValue(int imageIndex, vec3 parentPosition, out float featureValue)
{
    ImageDescriptor desc = images[imageIndex];
    vec3 imagePosition = (desc.invModel * vec4(parentPosition, 1.0)).xyz;
    if (any(lessThan(imagePosition, desc.bottom.xyz)) || any(greaterThan(imagePosition, desc.top.xyz)))
        return false;

    vec3 sizeImage = desc.top.xyz - desc.bottom.xyz;
    if (sizeImage.x <= 0.0 || sizeImage.y <= 0.0 || sizeImage.z < 0.0)
        return false;

    vec3 position = (imagePosition - desc.bottom.xyz) / max(sizeImage, vec3(1e-6));
    position = clamp(position, vec3(0.0), vec3(1.0));

    float intensity;
    if (desc.flags.y != 0u) {
        sampler3D volume = sampler3D(desc.volumeHandle.xy);
        intensity = texture(volume, position).r;
    }
    else {
        usampler3D uvolume = usampler3D(desc.volumeHandle.xy);
        ivec3 tsize = textureSize(uvolume, 0);
        if (tsize.z == 1)
            tsize.z = 0;
        ivec3 texPos = ivec3(position * vec3(tsize));
        intensity = float(texelFetch(uvolume, texPos, 0).r);
    }

    if (intensity < desc.pixelParams.x)
        return false;

    float x = intensity, y = 0.0;
    sampler2D featureTexture = sampler2D(desc.featureHandle.xy);
    if (desc.featureDims.y == 1.0) {
        x = (intensity - desc.pixelParams.x) / max(desc.pixelParams.y - desc.pixelParams.x, 1e-6);
    }
    else {
        offset_feature_texture(intensity, desc.featureDims.x, desc.featureDims.y, x, y);
        y = scaleOffsetVar(desc.featureDims.y, y);
    }
    x = scaleOffsetVar(desc.featureDims.x, x);
    featureValue = texture(featureTexture, vec2(x, y)).r;
    return true;
}

vec4 computeImageColor(ImageDescriptor desc, float value)
{
    if (desc.flags.w != 0u) {
        value = clamp(value, desc.featureParams.x, desc.featureParams.y);
    }
    if (desc.flags.x == 0u && desc.flags.w == 0u && (value < desc.featureParams.x || value > desc.featureParams.y))
        return vec4(0.0);

    if (desc.flags.x != 0u && value > desc.featureParams.x && value < desc.featureParams.y)
        return vec4(1.0, 0.0, 0.0, 1.0);

    value = (value - desc.pixelParams.z) / max(desc.pixelParams.w - desc.pixelParams.z, 1e-6);
    sampler1D lutTexture = sampler1D(desc.lutHandle.xy);

    if (desc.flags.z != 0u) {
        float posLut = scaleOffsetVar(512.0, value);
        return vec4(texture(lutTexture, posLut).xyz, 1.0);
    }

    vec4 colour = vec4(value, value, value, (exp(value) - 1.0) / (exp(1.0) - 1.0));
    colour.rgb = colour.a * colour.rgb + (1.0 - colour.a) * pow(background_colour, vec3(gamma)).rgb;
    float posLut = scaleOffsetVar(512.0, colour.r);
    colour.rgb = texture(lutTexture, posLut).xyz;
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

    vec3 ray_start = ray_origin + ray_direction * t_0;
    vec3 ray_stop = ray_origin + ray_direction * t_1;
    vec3 ray_step = (ray_stop - ray_start) / float(nb_steps);
    vec3 parentPosition = ray_start;

    float maximumIntensity[1024];
    for (int n = 0; n < nbImages; ++n)
        maximumIntensity[n] = -3.402823466e+38;

    for (int step = 0; step < nb_steps; ++step) {
        parentPosition += ray_step;
        for (int curImage = 0; curImage < nbImages; ++curImage) {
            float featureValue;
            if (!sampleFeatureValue(curImage, parentPosition, featureValue))
                continue;

            ImageDescriptor desc = images[curImage];
            if (desc.flags.w != 0u) {
                if (featureValue >= maximumIntensity[curImage])
                    maximumIntensity[curImage] = featureValue;
            }
            else if (featureValue >= maximumIntensity[curImage] && featureValue <= desc.featureParams.y) {
                maximumIntensity[curImage] = featureValue;
            }
        }
    }

    a_colour = vec4(0.0);
    for (int curImage = 0; curImage < nbImages; ++curImage) {
        vec4 currentColor = computeImageColor(images[curImage], maximumIntensity[curImage]);
        a_colour += currentColor;
    }
    a_colour = clamp(a_colour, 0.0, 1.0);
}
