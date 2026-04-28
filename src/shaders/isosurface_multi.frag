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
uniform int nbImages;
uniform int nb_steps;
uniform float gamma;

struct Ray { vec3 origin; vec3 direction; };
struct AABB { vec3 top; vec3 bottom; };

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

bool parentToImagePosition(int imageIndex, vec3 parentPosition, out vec3 position, out ImageDescriptor desc)
{
    desc = images[imageIndex];
    vec3 imagePosition = (desc.invModel * vec4(parentPosition, 1.0)).xyz;
    if (any(lessThan(imagePosition, desc.bottom.xyz)) || any(greaterThan(imagePosition, desc.top.xyz)))
        return false;

    vec3 sizeImage = desc.top.xyz - desc.bottom.xyz;
    if (sizeImage.x <= 0.0 || sizeImage.y <= 0.0 || sizeImage.z < 0.0)
        return false;

    position = (imagePosition - desc.bottom.xyz) / max(sizeImage, vec3(1e-6));
    position = clamp(position, vec3(0.0), vec3(1.0));
    return true;
}

float sampleRawIntensity(ImageDescriptor desc, vec3 position)
{
    position = clamp(position, vec3(0.0), vec3(1.0));
    if (desc.flags.y != 0u) {
        sampler3D volume = sampler3D(desc.volumeHandle.xy);
        return texture(volume, position).r;
    }

    usampler3D uvolume = usampler3D(desc.volumeHandle.xy);
    return float(texture(uvolume, position).r);
}

bool sampleFeatureValue(int imageIndex, vec3 parentPosition, out float featureValue, out vec3 imagePosition, out ImageDescriptor desc)
{
    if (!parentToImagePosition(imageIndex, parentPosition, imagePosition, desc))
        return false;

    float intensity = sampleRawIntensity(desc, imagePosition);
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

vec3 gradientAt(ImageDescriptor desc, vec3 imagePosition)
{
    vec3 e = vec3(
        1.0 / max(desc.featureDims.x, 256.0),
        1.0 / max(desc.featureDims.x, 256.0),
        1.0 / max(max(desc.featureDims.y, 1.0), 256.0));

    float center = sampleRawIntensity(desc, imagePosition);
    float dx = sampleRawIntensity(desc, imagePosition + vec3(e.x, 0.0, 0.0)) - center;
    float dy = sampleRawIntensity(desc, imagePosition + vec3(0.0, e.y, 0.0)) - center;
    float dz = sampleRawIntensity(desc, imagePosition + vec3(0.0, 0.0, e.z)) - center;
    vec3 localNormal = normalize(-vec3(dx, dy, dz));
    mat3 normalMatrix = transpose(mat3(desc.invModel));
    return normalize(normalMatrix * localNormal);
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

    bool found = false;
    int hitImage = -1;
    vec3 hitParentPosition = vec3(0.0);
    vec3 hitImagePosition = vec3(0.0);
    ImageDescriptor hitDesc;
    float hitDepth = 1.0;

    for (int step = 0; step < nb_steps && !found; ++step) {
        parentPosition += ray_step;
        for (int curImage = 0; curImage < nbImages && !found; ++curImage) {
            float featureValue;
            vec3 imagePosition;
            ImageDescriptor desc;
            if (!sampleFeatureValue(curImage, parentPosition, featureValue, imagePosition, desc))
                continue;
            if (!(featureValue > desc.featureParams.x && featureValue < desc.featureParams.y))
                continue;

            hitImage = curImage;
            hitParentPosition = parentPosition;
            hitImagePosition = imagePosition;
            hitDesc = desc;
            vec4 clipSpacePos = ModelViewProjectionMatrix * vec4(parentPosition, 1.0);
            float ndcDepth = clipSpacePos.z / clipSpacePos.w;
            hitDepth = 0.5 * ndcDepth + 0.5;
            found = true;
        }
    }

    if (!found || hitImage < 0)
        discard;

    vec3 N = gradientAt(hitDesc, hitImagePosition);
    vec3 L = normalize(light_position - hitParentPosition);
    vec3 V = -normalize(ray_direction);
    vec3 H = normalize(L + V);

    float Ia = 0.1;
    float Id = max(0.0, dot(N, L));
    float Is = 8.0 * pow(max(0.0, dot(N, H)), 120.0);

    vec3 colour = (Ia + Id) * material_colour + Is * vec3(1.0);
    a_colour = vec4(pow(colour, vec3(1.0 / gamma)), 1.0);
    gl_FragDepth = clamp(hitDepth, 0.0, 1.0);
}
