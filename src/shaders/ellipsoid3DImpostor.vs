#version 330 core
layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vAxes;
layout(location = 2) in vec3 vAxeX;
layout(location = 3) in vec3 vAxeY;
layout(location = 4) in vec3 vAxeZ;
out vec3 v_axes;
out vec3 v_pos;
out mat3 v_rotMat;
void main()
{
	vec4 pos = vec4(vPos, 1);
	gl_Position = pos;
	v_axes = vAxes;
	v_pos = vPos;
	v_rotMat[0] = vAxeX;
	v_rotMat[1] = vAxeY;
	v_rotMat[2] = vAxeZ;
};