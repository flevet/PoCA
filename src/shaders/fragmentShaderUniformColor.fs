#version 330 core

in float selection;
in float vclipDistance;

out vec4 color;

uniform vec4 singleColor;
uniform bool clip;

void main(){
	if(clip && vclipDistance < 0.f)
		discard;
		
	if(selection < 0.5) 
		discard;

	color = singleColor;
}