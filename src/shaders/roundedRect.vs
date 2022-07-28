#version 330 core
precision highp float;
uniform vec4 box;
uniform vec2 window;
uniform float sigma;
attribute vec2 coord;
out vec2 vertex;
void main() {
  float padding = 3.0 * sigma;
  vertex = mix(box.xy - padding, box.zw + padding, coord);
  gl_Position = vec4(vertex / window * 2.0 - 1.0, 0.0, 1.0);
}