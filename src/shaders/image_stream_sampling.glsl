// Shared display-only preview/detail mapping. Scientific normalized positions stay unchanged.
// hasDetail / streamFlags.x means resident AND spatially/semantically renderable.
// CPU quality mismatch requests refinement; it never disables covered detail.
// Bounds alone must never reactivate a source-incompatible or uncovered texture.
float streamUnsigned(usampler3D tex, vec3 coord, bool linear)
{
    if (!linear) return float(texture(tex, coord).r);
    ivec3 dims = textureSize(tex, 0);
    vec3 voxel = coord * vec3(dims) - 0.5;
    ivec3 first = ivec3(floor(voxel));
    vec3 weight = fract(voxel);
    float value = 0.0;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) {
                ivec3 offset = ivec3(x, y, z);
                vec3 weights = mix(vec3(1.0) - weight, weight, vec3(offset));
                value += float(texelFetch(tex, clamp(first + offset, ivec3(0), dims - 1), 0).r) * weights.x * weights.y * weights.z;
            }
    return value;
}
float streamFloat(sampler3D tex, vec3 coord, bool linear)
{
    if (!linear) return texture(tex, coord).r;
    ivec3 dims = textureSize(tex, 0);
    vec3 voxel = coord * vec3(dims) - 0.5;
    ivec3 first = ivec3(floor(voxel));
    vec3 weight = fract(voxel);
    float value = 0.0;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) {
                ivec3 offset = ivec3(x, y, z);
                vec3 weights = mix(vec3(1.0) - weight, weight, vec3(offset));
                value += texelFetch(tex, clamp(first + offset, ivec3(0), dims - 1), 0).r * weights.x * weights.y * weights.z;
            }
    return value;
}
bool streamContains(vec3 point, vec3 low, vec3 high)
{
    return all(greaterThanEqual(point, low)) && all(lessThanEqual(point, high));
}
#ifdef STREAM_ARRAY
uniform vec3 residentBottom[MAX_NB_IMAGES], residentTop[MAX_NB_IMAGES];
uniform vec3 previewBottom[MAX_NB_IMAGES], previewTop[MAX_NB_IMAGES];
uniform bool hasDetail[MAX_NB_IMAGES], streamLinear[MAX_NB_IMAGES];
uniform sampler3D previewVolume[MAX_NB_IMAGES];
uniform usampler3D previewUvolume[MAX_NB_IMAGES];
float streamRaw(int imageIndex, vec3 scientificPosition)
{
    vec3 local = bottom + scientificPosition * (top - bottom);
    bool detail = hasDetail[imageIndex] && streamContains(local, residentBottom[imageIndex], residentTop[imageIndex]);
    vec3 low = detail ? residentBottom[imageIndex] : previewBottom[imageIndex];
    vec3 high = detail ? residentTop[imageIndex] : previewTop[imageIndex];
    if (!streamContains(local, low, high)) return -3.402823466e+38;
    vec3 coord = (local - low) / max(high - low, vec3(1e-6));
    if (isFloat[imageIndex])
        return detail ? streamFloat(volume[imageIndex], coord, streamLinear[imageIndex]) : streamFloat(previewVolume[imageIndex], coord, streamLinear[imageIndex]);
    return detail ? streamUnsigned(uvolume[imageIndex], coord, streamLinear[imageIndex]) : streamUnsigned(previewUvolume[imageIndex], coord, streamLinear[imageIndex]);
}
#endif
#ifdef STREAM_SINGLE
uniform vec3 residentBottom, residentTop, previewBottom, previewTop;
uniform bool hasDetail, streamLinear;
uniform sampler3D previewVolume;
uniform usampler3D previewUvolume;
float streamRaw(vec3 scientificPosition)
{
    vec3 local = bottom + scientificPosition * (top - bottom);
    bool detail = hasDetail && streamContains(local, residentBottom, residentTop);
    vec3 low = detail ? residentBottom : previewBottom;
    vec3 high = detail ? residentTop : previewTop;
    if (!streamContains(local, low, high)) return -3.402823466e+38;
    vec3 coord = (local - low) / max(high - low, vec3(1e-6));
    if (isFloat) return detail ? streamFloat(volume, coord, streamLinear) : streamFloat(previewVolume, coord, streamLinear);
    return detail ? streamUnsigned(uvolume, coord, streamLinear) : streamUnsigned(previewUvolume, coord, streamLinear);
}
uint streamLabel(vec3 scientificPosition)
{
    vec3 local = bottom + scientificPosition * (top - bottom);
    bool detail = hasDetail && streamContains(local, residentBottom, residentTop);
    vec3 low = detail ? residentBottom : previewBottom;
    vec3 high = detail ? residentTop : previewTop;
    if (!streamContains(local, low, high)) return 0u;
    vec3 coord = (local - low) / max(high - low, vec3(1e-6));
    return detail ? texture(uvolume, coord).r : texture(previewUvolume, coord).r;
}
#endif
#ifdef STREAM_MULTI
float streamRaw(ImageDescriptor desc, vec3 local)
{
    bool detail = desc.streamFlags.x != 0u && streamContains(local, desc.residentBottom.xyz, desc.residentTop.xyz);
    vec3 low = detail ? desc.residentBottom.xyz : desc.previewBottom.xyz;
    vec3 high = detail ? desc.residentTop.xyz : desc.previewTop.xyz;
    if (!streamContains(local, low, high)) return -3.402823466e+38;
    vec3 coord = (local - low) / max(high - low, vec3(1e-6));
    uvec2 handle = detail ? desc.volumeHandle.xy : desc.previewHandle.xy;
    if (desc.flags.y != 0u) return streamFloat(sampler3D(handle), coord, desc.streamFlags.y != 0u);
    return streamUnsigned(usampler3D(handle), coord, desc.streamFlags.y != 0u);
}
#endif
