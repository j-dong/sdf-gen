#version 450
layout(location = 0) out float color;
layout(location = 0) in vec4 frag_pos_params;
void main() {
    float u = frag_pos_params.z;
    float v = frag_pos_params.w;
    color = float(u * u - v < 0.0);
}
