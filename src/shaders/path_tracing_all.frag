#version 440 core
#extension GL_ARB_bindless_texture : require

out vec4 a_colour;

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

uniform vec3 background_colour;
uniform vec3 material_colour;
uniform vec3 light_position;

const int MAX_NB_IMAGES = 16;
const int IMAGE_INDEX = 0;

uniform int nbImages;
uniform int nb_steps;
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

uniform sampler2D previousAccum;
uniform uint sampleCounter;
uniform vec2 resolution;
uniform float pathTraceDensity;
uniform float pathTraceBrightness;
uniform float pathTraceOpacity;
uniform float pathTraceOpacityMin;
uniform float pathTraceOpacityMax;
uniform float pathTraceOpacityGamma;
uniform int pathTraceShadingMode;
uniform float pathTraceGradientFactor;
uniform float pathTraceAmbient;
uniform float pathTraceDiffuse;
uniform float pathTraceSpecular;
uniform float pathTraceRoughness;
uniform float pathTraceEmission;
uniform float pathTraceRim;
uniform float pathTraceShadowDensity;
uniform float pathTraceSkyIntensity;
uniform float pathTraceAreaLightSize;
uniform uint pathTraceShadowSteps;

const float EPSILON = 0.0005;
const int PATH_TRACE_SHADING_BRDF = 0;
const int PATH_TRACE_SHADING_PHASE = 1;
const int PATH_TRACE_SHADING_MIXED = 2;

struct Ray {
	vec3 origin;
	vec3 direction;
};

struct AABB {
	vec3 top;
	vec3 bottom;
};

uint hashUint(uint x)
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

float rand(inout uint state)
{
	state = hashUint(state);
	return float(state) * (1.0 / 4294967296.0);
}

float scaleOffsetVar(float textureSize, float pos)
{
	float safeSize = max(textureSize, 1.0);
	float scale = (safeSize - 1.0) / safeSize;
	float offset = 1.0 / (2.0 * safeSize);
	return scale * pos + offset;
}

void offsetFeatureTexture(float labelId, float w, float h, out float x, out float y)
{
	float safeW = max(w, 1.0);
	float safeH = max(h, 1.0);
	float id = max(labelId - 1.0, 0.0);
	float row = floor(id / safeW);
	y = row / max(safeH - 1.0, 1.0);
	x = (id - (row * safeW)) / max(safeW - 1.0, 1.0);
}

void rayBoxIntersection(Ray ray, AABB box, out float t0, out float t1)
{
	vec3 directionInv = 1.0 / ray.direction;
	vec3 tTop = directionInv * (box.top - ray.origin);
	vec3 tBottom = directionInv * (box.bottom - ray.origin);
	vec3 tMin = min(tTop, tBottom);
	vec2 tNear = max(tMin.xx, tMin.yz);
	t0 = max(0.0, max(tNear.x, tNear.y));
	vec3 tMax = max(tTop, tBottom);
	vec2 tFar = min(tMax.xx, tMax.yz);
	t1 = min(tFar.x, tFar.y);
}

float readRawVolume(vec3 position)
{
	if (any(lessThan(position, vec3(0.0))) || any(greaterThan(position, vec3(1.0))))
		return pixel_min[IMAGE_INDEX] - 1.0;
	if (isFloat[IMAGE_INDEX])
		return texture(volume[IMAGE_INDEX], position).r;
	return float(texture(uvolume[IMAGE_INDEX], position).r);
}

float featureFromRaw(float rawIntensity)
{
	float x = rawIntensity;
	float y = 0.0;
	if (height_feature_texture[IMAGE_INDEX] <= 1.0) {
		x = (rawIntensity - pixel_min[IMAGE_INDEX]) / max(pixel_max[IMAGE_INDEX] - pixel_min[IMAGE_INDEX], 0.000001);
	}
	else {
		offsetFeatureTexture(rawIntensity, width_feature_texture[IMAGE_INDEX], height_feature_texture[IMAGE_INDEX], x, y);
		y = scaleOffsetVar(height_feature_texture[IMAGE_INDEX], y);
	}
	x = scaleOffsetVar(width_feature_texture[IMAGE_INDEX], x);
	return texture(featureTexture[IMAGE_INDEX], vec2(clamp(x, 0.0, 1.0), clamp(y, 0.0, 1.0))).r;
}

bool featureIsActive(float featureValue)
{
	if (applyThreshold[IMAGE_INDEX])
		return featureValue > current_min[IMAGE_INDEX] && featureValue < current_max[IMAGE_INDEX];
	if (scaleLUT[IMAGE_INDEX])
		return true;
	return featureValue >= current_min[IMAGE_INDEX] && featureValue <= current_max[IMAGE_INDEX];
}

float normalizeFeature(float featureValue)
{
	float value = scaleLUT[IMAGE_INDEX] ? clamp(featureValue, current_min[IMAGE_INDEX], current_max[IMAGE_INDEX]) : featureValue;
	return clamp((value - feature_min[IMAGE_INDEX]) / max(feature_max[IMAGE_INDEX] - feature_min[IMAGE_INDEX], 0.000001), 0.0, 1.0);
}

float opacityTransfer(float normalizedFeature)
{
	float low = clamp(min(pathTraceOpacityMin, pathTraceOpacityMax), 0.0, 1.0);
	float high = min(max(max(pathTraceOpacityMin, pathTraceOpacityMax), low + 0.0001), 1.0);
	float opacity = clamp((normalizedFeature - low) / max(high - low, 0.0001), 0.0, 1.0);
	return max(pathTraceOpacity, 0.0) * pow(opacity, max(pathTraceOpacityGamma, 0.0001));
}

float densityAt(vec3 position, out float featureValue)
{
	float raw = readRawVolume(position);
	if (raw < pixel_min[IMAGE_INDEX])
		return 0.0;
	featureValue = featureFromRaw(raw);
	if (!featureIsActive(featureValue))
		return 0.0;
	return opacityTransfer(normalizeFeature(featureValue));
}

float densityOnly(vec3 position)
{
	float featureValue = 0.0;
	return densityAt(position, featureValue);
}

vec3 transferColor(float featureValue)
{
	if (applyThreshold[IMAGE_INDEX] && featureValue > current_min[IMAGE_INDEX] && featureValue < current_max[IMAGE_INDEX])
		return vec3(1.0, 0.0, 0.0);

	float lutPos = scaleOffsetVar(512.0, normalizeFeature(featureValue));
	vec3 color = texture(lutTexture[IMAGE_INDEX], lutPos).rgb * material_colour;
	return pow(clamp(color, 0.0, 1.0), vec3(1.0 / max(gamma, 0.0001)));
}

vec3 gradientNormal(vec3 position, vec3 viewDirection, out float gradientMagnitude)
{
	ivec3 textureDimensions = isFloat[IMAGE_INDEX] ? textureSize(volume[IMAGE_INDEX], 0) : textureSize(uvolume[IMAGE_INDEX], 0);
	vec3 stepSize = 1.0 / max(vec3(textureDimensions), vec3(1.0));
	float dx = densityOnly(position + vec3(stepSize.x, 0.0, 0.0)) - densityOnly(position - vec3(stepSize.x, 0.0, 0.0));
	float dy = densityOnly(position + vec3(0.0, stepSize.y, 0.0)) - densityOnly(position - vec3(0.0, stepSize.y, 0.0));
	float dz = densityOnly(position + vec3(0.0, 0.0, stepSize.z)) - densityOnly(position - vec3(0.0, 0.0, stepSize.z));
	gradientMagnitude = length(vec3(dx, dy, dz));
	vec3 worldScale = max(abs(top - bottom), vec3(0.000001));
	vec3 normal = vec3(dx, dy, dz) / worldScale;
	if (dot(normal, normal) < 0.00000001)
		normal = -viewDirection;
	normal = normalize(normal);
	if (dot(normal, -viewDirection) < 0.0)
		normal = -normal;
	return normal;
}

vec3 orthogonalVector(vec3 v)
{
	return normalize(abs(v.z) < 0.999 ? cross(v, vec3(0.0, 0.0, 1.0)) : cross(v, vec3(0.0, 1.0, 0.0)));
}

vec3 sampleAreaLightDirection(vec3 lightDirection, inout uint seed)
{
	float radius = max(pathTraceAreaLightSize, 0.0);
	if (radius <= 0.0)
		return lightDirection;
	float r = sqrt(rand(seed)) * radius;
	float a = 6.28318530718 * rand(seed);
	vec3 tangent = orthogonalVector(lightDirection);
	vec3 bitangent = normalize(cross(lightDirection, tangent));
	return normalize(lightDirection + (cos(a) * tangent + sin(a) * bitangent) * r);
}

float transmittanceToLight(vec3 position, vec3 lightDirectionWorld)
{
	vec3 worldScale = max(abs(top - bottom), vec3(0.000001));
	vec3 lightDirectionTexture = normalize(lightDirectionWorld / worldScale);
	Ray shadowRay = Ray(position + lightDirectionTexture * EPSILON, lightDirectionTexture);
	AABB textureBox = AABB(vec3(1.0), vec3(0.0));
	float t0 = 0.0;
	float t1 = 0.0;
	rayBoxIntersection(shadowRay, textureBox, t0, t1);
	if (t1 <= t0)
		return 1.0;

	int shadowSteps = clamp(int(pathTraceShadowSteps), 1, 512);
	float stepLength = (t1 - t0) / float(shadowSteps);
	float t = t0 + stepLength;
	float opticalDepth = 0.0;
	for (int step = 0; step < 512; ++step) {
		if (step >= shadowSteps)
			break;
		vec3 p = shadowRay.origin + shadowRay.direction * t;
		opticalDepth += densityOnly(p) * pathTraceDensity * pathTraceShadowDensity * stepLength;
		t += stepLength;
	}
	return exp(-opticalDepth);
}

vec3 brdfLighting(vec3 baseColor, vec3 normal, vec3 viewDirection, vec3 lightDirection, float shadow)
{
	vec3 toCamera = normalize(-viewDirection);
	vec3 halfVector = normalize(lightDirection + toCamera);
	float diffuse = max(dot(normal, lightDirection), 0.0);
	float ndotv = max(dot(normal, toCamera), 0.0);
	float rim = pow(clamp(1.0 - ndotv, 0.0, 1.0), 2.0);
	float roughness = clamp(pathTraceRoughness, 0.02, 1.0);
	float shininess = mix(256.0, 8.0, roughness);
	float specular = pow(max(dot(normal, halfVector), 0.0), shininess);
	vec3 color = baseColor * (pathTraceAmbient + pathTraceSkyIntensity);
	color += baseColor * pathTraceDiffuse * diffuse * shadow;
	color += baseColor * pathTraceRim * rim;
	color += vec3(pathTraceSpecular * specular * shadow);
	return color;
}

vec3 phaseLighting(vec3 baseColor, vec3 viewDirection, vec3 lightDirection, float shadow)
{
	float forward = pow(max(dot(lightDirection, -viewDirection), 0.0), 2.0);
	float isotropic = 0.5;
	float phase = mix(isotropic, forward, 0.35);
	return baseColor * (pathTraceSkyIntensity + pathTraceDiffuse * phase * shadow);
}

bool sampleFreePath(vec3 rayStart, vec3 rayDirection, float rayLength, inout uint seed, out vec3 hitPosition, out float hitFeature, out float hitDensity)
{
	float stepLength = rayLength / max(float(nb_steps), 1.0);
	float t = rand(seed) * stepLength;
	float target = -log(max(rand(seed), 0.000001));
	float opticalDepth = 0.0;

	for (int step = 0; step < 10000; ++step) {
		if (step >= nb_steps || t > rayLength)
			break;
		vec3 position = rayStart + rayDirection * t;
		float featureValue = 0.0;
		float density = densityAt(position, featureValue);
		opticalDepth += density * pathTraceDensity * stepLength;
		if (density > 0.0 && opticalDepth >= target) {
			hitPosition = position;
			hitFeature = featureValue;
			hitDensity = density;
			return true;
		}
		t += stepLength;
	}
	return false;
}

vec4 shadeSample(vec3 rayStart, vec3 rayStop, inout uint seed)
{
	vec3 ray = rayStop - rayStart;
	float rayLength = length(ray);
	if (rayLength <= 0.0)
		return vec4(background_colour, 0.0);

	vec3 rayDirection = ray / rayLength;
	vec3 hitPosition = vec3(0.0);
	float hitFeature = 0.0;
	float hitDensity = 0.0;
	if (!sampleFreePath(rayStart, rayDirection, rayLength, seed, hitPosition, hitFeature, hitDensity))
		return vec4(background_colour, 0.0);

	vec3 worldPosition = hitPosition * (top - bottom) + bottom;
	vec3 viewDirection = normalize(rayStop - rayStart);
	vec3 lightVector = light_position - worldPosition;
	vec3 lightDirection = normalize(lightVector);
	lightDirection = sampleAreaLightDirection(lightDirection, seed);
	float gradientMagnitude = 0.0;
	vec3 normal = gradientNormal(hitPosition, viewDirection, gradientMagnitude);
	float shadow = transmittanceToLight(hitPosition, lightDirection);
	vec3 baseColor = transferColor(hitFeature);
	vec3 brdf = brdfLighting(baseColor, normal, viewDirection, lightDirection, shadow);
	vec3 phase = phaseLighting(baseColor, viewDirection, lightDirection, shadow);
	float brdfWeight = 1.0;
	if (pathTraceShadingMode == PATH_TRACE_SHADING_PHASE)
		brdfWeight = 0.0;
	else if (pathTraceShadingMode == PATH_TRACE_SHADING_MIXED)
		brdfWeight = 1.0 - exp(-max(pathTraceGradientFactor, 0.0) * gradientMagnitude);
	vec3 color = mix(phase, brdf, clamp(brdfWeight, 0.0, 1.0));
	color += baseColor * hitDensity * pathTraceEmission;
	return vec4(max(color, vec3(0.0)), 1.0);
}

void writeAccumulated(vec4 sampleColor)
{
	vec2 uv = gl_FragCoord.xy / max(resolution, vec2(1.0));
	float exposure = max(pathTraceBrightness, 0.0001);
	vec4 previous = texture(previousAccum, uv);
	float previousSamples = float(sampleCounter);
	vec3 previousLinear = -log(max(vec3(0.000001), 1.0 - clamp(previous.rgb, 0.0, 0.999999))) / exposure;
	vec3 averagedLinear = (previousLinear * previousSamples + max(sampleColor.rgb, vec3(0.0))) / (previousSamples + 1.0);
	float averagedAlpha = (previous.a * previousSamples + sampleColor.a) / (previousSamples + 1.0);
	a_colour = vec4(1.0 - exp(-averagedLinear * exposure), averagedAlpha);
}

void main()
{
	uint seed = uint(gl_FragCoord.x) * 1973u
		+ uint(gl_FragCoord.y) * 9277u
		+ sampleCounter * 26699u
		+ 911u;
	vec2 jitter = vec2(rand(seed), rand(seed)) - vec2(0.5);

	vec4 ndcPos;
	ndcPos.xy = ((2.0 * (gl_FragCoord.xy + jitter)) - (2.0 * viewport.xy)) / viewport.zw - 1.0;
	ndcPos.z = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far) / (gl_DepthRange.far - gl_DepthRange.near);
	ndcPos.w = 1.0;

	vec4 clipPos = ndcPos;
	clipPos.z = -1.0;
	vec4 eyePos = invMVP * clipPos;
	eyePos /= eyePos.w;
	vec3 rayOrigin = eyePos.xyz;
	vec3 currentRayDirection = perspective_projection ? normalize(rayOrigin - camera_position) : ray_direction;
	vec3 currentRayOrigin = perspective_projection ? camera_position : rayOrigin + currentRayDirection;

	float t0 = 0.0;
	float t1 = 0.0;
	float t0Crop = 0.0;
	float t1Crop = 0.0;
	Ray castingRay = Ray(currentRayOrigin, currentRayDirection);
	AABB boundingBox = AABB(top, bottom);
	rayBoxIntersection(castingRay, boundingBox, t0, t1);
	if (t1 <= t0) {
		writeAccumulated(vec4(background_colour, 0.0));
		return;
	}

	if (cropped) {
		AABB cropBox = AABB(top_crop, bottom_crop);
		rayBoxIntersection(castingRay, cropBox, t0Crop, t1Crop);
		if (t0Crop > t1Crop) {
			writeAccumulated(vec4(background_colour, 0.0));
			return;
		}
		t0 = t0Crop;
		t1 = t1Crop;
	}

	vec3 rayStart = (currentRayOrigin + currentRayDirection * t0 - bottom) / (top - bottom);
	vec3 rayStop = (currentRayOrigin + currentRayDirection * t1 - bottom) / (top - bottom);
	writeAccumulated(shadeSample(rayStart, rayStop, seed));
}
