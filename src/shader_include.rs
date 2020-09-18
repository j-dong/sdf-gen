#[repr(C)]
pub struct AlignedCode<T: ?Sized> {
    pub _align: [u32; 0],
    pub data: T,
}

macro_rules! include_shader {
    ($path: literal) => {
        {
            use crate::shader_include::AlignedCode;
            static ALIGNED_CODE: &AlignedCode<[u8]> = &AlignedCode {
                _align: [],
                data: *include_bytes!(concat!(env!("OUT_DIR"), "/", $path, ".spv")),
            };
            &ALIGNED_CODE.data
        }
    }
}
