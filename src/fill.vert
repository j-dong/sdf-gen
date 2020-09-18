#version 450
layout(location = 0) in vec4 pos_params;
layout(location = 0) out vec4 frag_pos_params;
layout(push_constant) uniform PushConstants {
    vec2 size;
} U;
void main() {
    gl_Position = vec4(2.0 * pos_params.xy / U.size - 1.0, 0.0, 1.0);
    frag_pos_params = pos_params;
}
