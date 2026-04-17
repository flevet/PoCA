#version 330 core
layout (location = 0) out float gIndex;

uniform float minFeatureValue;
uniform bool clip;

in float id;
in float feature;
in float vclipDistance;

void main() {
	if (clip && vclipDistance < 0.0)
		discard;
	if (feature < minFeatureValue)
		discard;
	gIndex = id;
}
