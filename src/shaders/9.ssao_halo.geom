#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 MVP;
uniform mat4 projection;

in vec3 sigmas[];
in float feature[];

out vec2 TexCoords_GS;
out vec3 center;
out vec3 spot_sigma;
out float spot_feature;

const vec2 uv[4] = vec2[4](vec2(1,1),
					vec2(1,0),
					vec2(0,1),
					vec2(0,0));

void main() {
		vec4 vc[4];

		float augmentedRadius = sigmas[0].x * 1.2f;

		vc[0] = vec4(augmentedRadius, augmentedRadius, 0.0, 0.0);
		vc[1] = vec4(augmentedRadius, -augmentedRadius, 0.0, 0.0);
		vc[2] = vec4(-augmentedRadius, augmentedRadius, 0.0, 0.0);
		vc[3] = vec4(-augmentedRadius, -augmentedRadius, 0.0, 0.0);
		
		for(int i = 0; i < 4; i++){
			//gl_Position = MVP * (gl_in[0].gl_Position + vc[i]); 
			gl_Position = MVP * gl_in[0].gl_Position + projection * vc[i]; 
			spot_sigma = sigmas[0];
			spot_feature = feature[0];
			TexCoords_GS = uv[i];
			center = gl_in[0].gl_Position.xyz;
			gl_PrimitiveID = gl_PrimitiveIDIn;
			gl_ClipDistance[0] = gl_in[0].gl_ClipDistance[0];
			gl_ClipDistance[1] = gl_in[0].gl_ClipDistance[1];
			gl_ClipDistance[2] = gl_in[0].gl_ClipDistance[2];
			gl_ClipDistance[3] = gl_in[0].gl_ClipDistance[3];
			gl_ClipDistance[4] = gl_in[0].gl_ClipDistance[4];
			gl_ClipDistance[5] = gl_in[0].gl_ClipDistance[5];
			EmitVertex();
		}
    EndPrimitive();
}