#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 MVP;
uniform mat4 projection;
uniform float radius;

in vec3 v_color[];
in float v_feature[];
in float v_clipDistance[];

out vec2 TexCoords_GS;
out vec3 center;
out vec3 vcolor;
out float vfeature;
out float vclipDistance;

const vec2 uv[4] = vec2[4](vec2(1,1),
					vec2(1,0),
					vec2(0,1),
					vec2(0,0));

void main() {
		vec4 vc[4];
		vc[0] = vec4(radius, radius, 0.0, 0.0);
		vc[1] = vec4(radius, -radius, 0.0, 0.0);
		vc[2] = vec4(-radius, radius, 0.0, 0.0);
		vc[3] = vec4(-radius, -radius, 0.0, 0.0);
		
		for(int i = 0; i < 4; i++){
			//gl_Position = MVP * (gl_in[0].gl_Position + vc[i]); 
			gl_Position = MVP * gl_in[0].gl_Position + projection * vc[i]; 
			TexCoords_GS = uv[i];
			center = gl_in[0].gl_Position.xyz;
			vfeature = v_feature[0];
			vcolor = v_color[0];
			gl_PrimitiveID = gl_PrimitiveIDIn + 1;
			vclipDistance = v_clipDistance[0];
			EmitVertex();
		}
    EndPrimitive();
}