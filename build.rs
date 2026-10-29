//! With `--features vulkan`: compiles the GLSL compute shaders in
//! `shaders/` to SPIR-V in OUT_DIR (`gpu::spirv!` includes them). The
//! compiler is `glslc`: $GLSLC, else the one in PATH, else the Android NDK's
//! (`shader-tools/`, under $ANDROID_NDK_HOME or ~/Android/Sdk/ndk/*).

use std::path::PathBuf;
use std::process::Command;

fn glslc() -> PathBuf {
    if let Ok(p) = std::env::var("GLSLC") {
        return p.into();
    }
    if Command::new("glslc").arg("--version").output().is_ok() {
        return "glslc".into();
    }
    let mut ndks: Vec<PathBuf> = std::env::var("ANDROID_NDK_HOME").map(|p| vec![PathBuf::from(p)]).unwrap_or_default();
    if let Ok(home) = std::env::var("HOME")
        && let Ok(dir) = std::fs::read_dir(PathBuf::from(home).join("Android/Sdk/ndk"))
    {
        let mut v: Vec<PathBuf> = dir.flatten().map(|e| e.path()).collect();
        v.sort();
        ndks.extend(v.into_iter().rev());
    }
    for ndk in ndks {
        let p = ndk.join("shader-tools/linux-x86_64/glslc");
        if p.exists() {
            return p;
        }
    }
    panic!("glslc not found: set GLSLC or ANDROID_NDK_HOME");
}

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let glslc = glslc();
    for e in std::fs::read_dir("shaders").unwrap().flatten() {
        let src = e.path();
        if src.extension().is_none_or(|x| x != "comp") {
            continue;
        }
        let spv = out.join(src.file_stem().unwrap()).with_extension("spv");
        let st = Command::new(&glslc).args(["--target-env=vulkan1.1", "-O", "-o"]).arg(&spv).arg(&src).status().expect("run glslc");
        assert!(st.success(), "glslc failed on {}", src.display());
    }
}
