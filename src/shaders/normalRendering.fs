#version 330 core
out vec4 color;

uniform bool clip;
in float vclipDistance;

void main() {
	if(clip && vclipDistance < 0.f)
		discard;
		
	color = vec4(0,0,0,1);
}