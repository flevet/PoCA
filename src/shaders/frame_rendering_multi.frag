#version 440 core
#extension GL_ARB_bindless_texture : require

out vec4 a_colour;

uniform mat4 invMVP;
uniform vec4 viewport;
uniform vec3 ray_direction;
uniform vec3 camera_position;
uniform bool perspective_projection;
uniform vec3 background_colour;
uniform int nbImages;
uniform int currentFrame;
uniform float gamma;

uniform bool cropped;
uniform vec3 top_crop;
uniform vec3 bottom_crop;

const int MAX_CLIPPING_PLANES = 6;
uniform vec4 clipPlanes[MAX_CLIPPING_PLANES];
uniform bool clip;

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

bool sampleFeatureValue(int imageIndex, vec3 parentRayOrigin, vec3 parentRayDirection, out float featureValue)
{
    ImageDescriptor desc = images[imageIndex];
    vec3 localOrigin = (desc.invModel * vec4(parentRayOrigin, 1.0)).xyz;
    vec3 localTarget = (desc.invModel * vec4(parentRayOrigin + parentRayDirection, 1.0)).xyz;
    vec3 localDirection = normalize(localTarget - localOrigin);

    if (abs(localDirection.z) < 1e-6)
        return false;

    float planeZ = float(currentFrame) + 0.5;
    float t = (planeZ - localOrigin.z) / localDirection.z;
    if (t < 0.0)
        return false;

    vec3 imagePosition = localOrigin + t * localDirection;
    if (any(lessThan(imagePosition, desc.bottom.xyz)) || any(greaterThan(imagePosition, desc.top.xyz)))
        return false;

    vec3 parentPosition = parentRayOrigin + t * parentRayDirection;
    if (cropped && (any(lessThan(parentPosition, bottom_crop)) || any(greaterThan(parentPosition, top_crop))))
        return false;

    vec3 sizeImage = desc.top.xyz - desc.bottom.xyz;
    vec3 position = (imagePosition - desc.bottom.xyz) / max(sizeImage, vec3(1e-6));
    position.z = (float(currentFrame) + 0.5 - desc.bottom.z) / max(sizeImage.z, 1e-6);
    position = clamp(position, vec3(0.0), vec3(1.0));

    float intensity;
    if (desc.flags.y != 0u)
        intensity = texture(sampler3D(desc.volumeHandle.xy), position).r;
    else
        intensity = float(texture(usampler3D(desc.volumeHandle.xy), position).r);

    if (intensity < desc.pixelParams.x)
        return false;

    float x = intensity, y = 0.0;
    sampler2D featureTexture = sampler2D(desc.featureHandle.xy);
    if (desc.featureDims.y == 1.0)
        x = (intensity - desc.pixelParams.x) / max(desc.pixelParams.y - desc.pixelParams.x, 1e-6);
    else {
        offset_feature_texture(intensity, desc.featureDims.x, desc.featureDims.y, x, y);
        y = scaleOffsetVar(desc.featureDims.y, y);
    }
    x = scaleOffsetVar(desc.featureDims.x, x);
    featureValue = texture(featureTexture, vec2(x, y)).r;
    return true;
}

vec4 computeColor(ImageDescriptor desc, float value)
{
    if (desc.flags.w != 0u)
        value = clamp(value, desc.featureParams.x, desc.featureParams.y);

    if (desc.flags.x == 0u && desc.flags.w == 0u && (value < desc.featureParams.x || value > desc.featureParams.y))
        return vec4(0.0);

    if (desc.flags.x != 0u && value > desc.featureParams.x && value < desc.featureParams.y)
        return vec4(1.0, 0.0, 0.0, 1.0);

    value = (value - desc.pixelParams.z) / max(desc.pixelParams.w - desc.pixelParams.z, 1e-6);
    float posLut = scaleOffsetVar(512.0, value);
    sampler1D lutTexture = sampler1D(desc.lutHandle.xy);

    if (desc.flags.z != 0u)
        return vec4(texture(lutTexture, posLut).xyz, 1.0);

    vec4 colour = vec4(value, value, value, (exp(value) - 1.0) / (exp(1.0) - 1.0));
    colour.rgb = colour.a * colour.rgb + (1.0 - colour.a) * pow(background_colour, vec3(gamma)).rgb;
    posLut = scaleOffsetVar(512.0, colour.r);
    colour.rgb = texture(lutTexture, posLut).xyz;
    colour.a = 1.0;
    colour.rgb = pow(colour.rgb, vec3(1.0 / gamma));
    return colour;
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
    vec3 parentRayDirection = perspective_projection ? normalize(ray_origin - camera_position) : ray_direction;
    vec3 parentRayOrigin = perspective_projection ? camera_position : ray_origin + parentRayDirection;

    a_colour = vec4(0.0);
    bool hasContribution = false;
    for (int curImage = 0; curImage < nbImages; curImage++) {
        float value;
        if (!sampleFeatureValue(curImage, parentRayOrigin, parentRayDirection, value))
            continue;
        vec4 currentColor = computeColor(images[curImage], value);
        if (currentColor.a > 0.0)
            hasContribution = true;
        a_colour += currentColor;
    }

    if (!hasContribution)
        discard;
    a_colour = clamp(a_colour, 0.0, 1.0);
}
