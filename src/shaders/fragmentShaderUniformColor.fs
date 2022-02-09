#version 330 core

in float selection;

out vec4 color;

uniform vec4 singleColor;

void main(){
	if(selection < 0.5) 
		discard;

	color = singleColor;
}