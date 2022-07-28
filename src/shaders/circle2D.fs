#version 330 core
uniform vec4 singleColor;
uniform float thickness;
uniform float antialias;
uniform bool activatedAntialias;
uniform float radius;

in vec2 texC;
in vec3 center;
in float vradius;
in float vradius2;

out vec4 color;

void main()
{
	float p = dot(texC, texC);
	
	float w = thickness / 2.0; 
	float a = 1.0;
	float d = 0.;
	d = abs(vradius - (p * vradius2));
	
	if (activatedAntialias) {
		if (d <= w) {
			color = vec4(singleColor.rgb, 1.0); 
		}
		else {
			d = d - w - antialias;
			d /= antialias; 
			color = vec4(singleColor.rgb, exp(-d * d));
		}
	}
	else {
		d = d - w;
		if (d > w)
			discard; 
		color = vec4(singleColor.rgb, 1.0); 
	}
};