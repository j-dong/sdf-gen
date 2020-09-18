use std::process::Command;
use std::path::{Path, PathBuf};

static SHADERS: &[&str] = &[
    "src/fill.vert",
    "src/fill.frag",
    "src/sdf.comp",
];

fn main() {
    // detect glslangValidator
    let mut compiler_paths = vec![
        "glslangValidator".to_owned(),
    ];
    if let Ok(path) = std::env::var("VULKAN_SDK") {
        compiler_paths.push(format!("{}/bin/glslangValidator", path));
        compiler_paths.push(format!("{}/Bin/glslangValidator", path));
    }
    let mut compiler = None;
    for path in compiler_paths {
        let res = Command::new(&path)
            .arg("--version")
            .output();
        match res {
            Ok(out) => {
                if !out.status.success() {
                    panic!("{} --version exited with failure status", &path);
                }
                let ver = std::str::from_utf8(&out.stdout).expect("non-utf8 output from glslangValidator --version");
                match ver.lines().next() {
                    Some(ver) => {
                        println!("found: {} at {}", ver, &path);
                        compiler = Some(path);
                        break;
                    }
                    None => {
                        println!("compiler at {} --version produced no output; skipping", &path);
                        continue;
                    }
                }
            }
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    continue
                } else {
                    panic!("error running {}: {}", &path, err);
                }
            }
        }
    }
    let compiler = match compiler {
        None => panic!("unable to locate glslangValidator"),
        Some(x) => x,
    };
    let out_dir = PathBuf::from(
        &std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    for shader in SHADERS {
        let mut filename = Path::new(shader)
            .file_name()
            .expect("shader does not have a filename")
            .to_os_string();
        filename.push(".spv");
        let out_filename = out_dir.join(filename);
        println!("cargo:rerun-if-changed={}", shader);
        println!("compiling: {} => {}", shader, out_filename.to_string_lossy());
        let status = Command::new(&compiler)
            .arg("-V")
            .arg("-g")
            .arg("-o")
            .arg(out_filename)
            .arg(shader)
            .status()
            .expect("error running glslangValidator to compile");
        if !status.success() {
            panic!("GLSL compilation failed");
        }
    }
}
