#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 14) out;
in vec3 v_axes[];
in vec3 v_pos[];
in mat3 v_rotMat[];
uniform mat4 MVP;
out vec3 TexCoords_GS;
out vec3 spot_axes;
out vec3 spot_center;
out vec4 spot_position;
out mat3 rotMat;
const int indexes[14] = int[14](3, 2, 6, 7, 4, 2, 0, 3, 1, 6, 5, 4, 1, 0);
/*const vec3 uv[8] = vec3[8](vec3(1, 1, 1),
	vec3(0, 1, 1),
	vec3(1, 1, 0),
	vec3(0, 1, 0),
	vec3(1, 0, 1),
	vec3(0, 0, 1),
	vec3(0, 0, 0),
	vec3(1, 0, 0));*/
const vec3 uv[8] = vec3[8](vec3(1, 1, 1),
	vec3(-1, 1, 1),
	vec3(1, 1,-1),
	vec3(-1, 1, -1),
	vec3(1, -1, 1),
	vec3(-1, -1, 1),
	vec3(-1, -1, -1),
	vec3(1, -1, -1));
void main() {
	rotMat = v_rotMat[0];
	mat3 rotMatScaled;
	rotMatScaled[0] = rotMat[0] * v_axes[0].x;
	rotMatScaled[1] = rotMat[1] * v_axes[0].y;
	rotMatScaled[2] = rotMat[2] * v_axes[0].z;
	vec3 vc[8];
	for (int i = 0; i < 8; i++)
		vc[i] = rotMatScaled * uv[i];
	for (int i = 0; i < 14; i++) {
		spot_axes = v_axes[0];
		gl_Position = MVP * (gl_in[0].gl_Position + vec4(vc[indexes[i]], 0.0));
		spot_position = gl_in[0].gl_Position + vec4(vc[indexes[i]], 0.0);
		spot_center = v_pos[0];//gl_in[0].gl_Position + vc[indexes[i]];
		TexCoords_GS = gl_in[0].gl_Position.xyz;
		EmitVertex();
	}
	EndPrimitive();
};
/*void main() {
	vec3 axeX = vec3(ellipsoidRotation[0][0], ellipsoidRotation[1][0], ellipsoidRotation[2][0]);
	axeX = normalize(axeX);
	axeX = axeX * v_axes[0].x;
	vec3 axeY = vec3(ellipsoidRotation[0][1], ellipsoidRotation[1][1], ellipsoidRotation[2][1]);
	axeY = normalize(axeY);
	axeY = axeY * v_axes[0].y;
	vec3 axeZ = vec3(ellipsoidRotation[0][2], ellipsoidRotation[1][2], ellipsoidRotation[2][2]);
	axeZ = normalize(axeZ);
	axeZ = axeZ * v_axes[0].z;
	vec3 vc[8];
	vc[0] = vec3(axeX + axeY + axeZ);
	vc[1] = vec3(-axeX + axeY + axeZ);
	vc[2] = vec3(axeX + axeY - axeZ);
	vc[3] = vec3(-axeX + axeY - axeZ);
	vc[4] = vec3(axeX - axeY + axeZ);
	vc[5] = vec3(-axeX - axeY + axeZ);
	vc[6] = vec3(-axeX - axeY - axeZ);
	vc[7] = vec3(axeX - axeY - axeZ);
	
	//for (int i = 0; i < 8; i++)
	//	vc[i] = ellipsoidRotation * vc[i];
	for (int i = 0; i < 14; i++) {
		spot_axes = v_axes[0];
		gl_Position = MVP * (gl_in[0].gl_Position + vec4(vc[indexes[i]], 0.0));
		spot_position = gl_in[0].gl_Position + vec4(vc[indexes[i]], 0.0);
		spot_center = v_pos[0];//gl_in[0].gl_Position + vc[indexes[i]];
		TexCoords_GS = gl_in[0].gl_Position.xyz;
		EmitVertex();
	}
	EndPrimitive();
};*/

/*#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 14) out;
in vec3 v_axes[];
in vec3 v_pos[];
uniform mat4 MVP;
uniform mat3 ellipsoidRotation;
out vec3 TexCoords_GS;
out vec3 spot_axes;
out vec3 spot_center;
out vec4 spot_position;
const int indexes[14] = int[14](3, 2, 6, 7, 4, 2, 0, 3, 1, 6, 5, 4, 1, 0);
const vec3 uv[8] = vec3[8](vec3(1, 1, 1),
	vec3(0, 1, 1),
	vec3(1, 1, 0),
	vec3(0, 1, 0),
	vec3(1, 0, 1),
	vec3(0, 0, 1),
	vec3(0, 0, 0),
	vec3(1, 0, 0));
void main() {
	vec3 vc[8];
	vc[0] = vec3(v_axes[0].x, v_axes[0].y, v_axes[0].z);
	vc[1] = vec3(-v_axes[0].x, v_axes[0].y, v_axes[0].z);
	vc[2] = vec3(v_axes[0].x, v_axes[0].y, -v_axes[0].z);
	vc[3] = vec3(-v_axes[0].x, v_axes[0].y, -v_axes[0].z);
	vc[4] = vec3(v_axes[0].x, -v_axes[0].y, v_axes[0].z);
	vc[5] = vec3(-v_axes[0].x, -v_axes[0].y, v_axes[0].z);
	vc[6] = vec3(-v_axes[0].x, -v_axes[0].y, -v_axes[0].z);
	vc[7] = vec3(v_axes[0].x, -v_axes[0].y, -v_axes[0].z);
	//for (int i = 0; i < 8; i++)
	//	vc[i] = ellipsoidRotation * vc[i];
	for (int i = 0; i < 14; i++) {
		spot_axes = v_axes[0];
		gl_Position = MVP * (gl_in[0].gl_Position + vec4(vc[indexes[i]], 0.0));
		spot_position = gl_in[0].gl_Position + vec4(vc[indexes[i]], 0.0);
		spot_center = v_pos[0];//gl_in[0].gl_Position + vc[indexes[i]];
		TexCoords_GS = gl_in[0].gl_Position.xyz;
		EmitVertex();
	}
	EndPrimitive();
};*/