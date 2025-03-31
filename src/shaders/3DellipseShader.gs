#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 14) out;

in vec3 sigmas[];
in float feature[];
//in float selection[];

uniform mat4 MVP;
uniform float radius;
uniform bool fixedRadius;

out vec3 TexCoords_GS;
out vec3 spot_sigma;
out float spot_feature;
out vec4 spot_position;

const int indexes[14] = int[14](3, 2, 6, 7, 4, 2, 0, 3, 1, 6, 5, 4, 1, 0);

const vec3 uv[8] = vec3[8](vec3(1,1,1),
					vec3(0,1,1),
					vec3(1,1,0),
					vec3(0,1,0),
					vec3(1,0,1),
					vec3(0,0,1),
					vec3(0,0,0),
					vec3(1,0,0));

void main() {
		vec4 vc[8];
		if(fixedRadius){
			float r3 = radius;
			vc[0] = vec4(r3, r3, r3, 0.0);
			vc[1] = vec4(-r3, r3, r3, 0.0);
			vc[2] = vec4(r3, r3, -r3, 0.0);
			vc[3] = vec4(-r3, r3, -r3, 0.0);
			vc[4] = vec4(r3, -r3, r3, 0.0);
			vc[5] = vec4(-r3, -r3, r3, 0.0);
			vc[6] = vec4(-r3, -r3, -r3, 0.0);
			vc[7] = vec4(r3, -r3, -r3, 0.0);
		}
		else{
			vec3 sigma3 = sigmas[0];
			vc[0] = vec4(sigma3.x, sigma3.y, sigma3.z, 0.0);
			vc[1] = vec4(-sigma3.x, sigma3.y, sigma3.z, 0.0);
			vc[2] = vec4(sigma3.x, sigma3.y, -sigma3.z, 0.0);
			vc[3] = vec4(-sigma3.x, sigma3.y, -sigma3.z, 0.0);
			vc[4] = vec4(sigma3.x, -sigma3.y, sigma3.z, 0.0);
			vc[5] = vec4(-sigma3.x, -sigma3.y, sigma3.z, 0.0);
			vc[6] = vec4(-sigma3.x, -sigma3.y, -sigma3.z, 0.0);
			vc[7] = vec4(sigma3.x, -sigma3.y, -sigma3.z, 0.0);
		}

		for(int i = 0; i < 14; i++){
			spot_feature = feature[0];
			spot_sigma = sigmas[0];
			gl_Position = MVP * (gl_in[0].gl_Position + vc[indexes[i]]); 
			spot_position = gl_in[0].gl_Position + vc[indexes[i]];
			TexCoords_GS = gl_in[0].gl_Position.xyz;
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