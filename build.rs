//! With `--features vulkan`: compiles the GLSL compute shaders in
//! `shaders/` to SPIR-V in OUT_DIR (`gpu::spirv!` includes them). The
//! compiler is `glslc`: $GLSLC, else the one in PATH, else the Android NDK's
//! (`shader-tools/`, under $ANDROID_NDK_HOME or ~/Android/Sdk/ndk/*).
//!
//! With `--features hexagon`: builds our DSP library (`dsp/`) for the
//! Hexagon v79 cDSP into OUT_DIR/libaether_dsp.so (`hexagon::DSP_LIB`
//! includes it): `qaic` turns `aether_dsp.idl` into the skel glue, and the
//! SDK's `hexagon-clang` compiles and links. The SDK is $HEXAGON_SDK_ROOT,
//! else ~/.hexagon-sdk/6.4.0.2.

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
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        shaders(&out);
    }
    if std::env::var_os("CARGO_FEATURE_HEXAGON").is_some() {
        dsp(&out);
    }
}

fn run(c: &mut Command) {
    let st = c.status().unwrap_or_else(|e| panic!("run {c:?}: {e}"));
    assert!(st.success(), "failed: {c:?}");
}

fn dsp(out: &std::path::Path) {
    println!("cargo:rerun-if-changed=dsp");
    println!("cargo:rerun-if-env-changed=HEXAGON_SDK_ROOT");
    let sdk = std::env::var("HEXAGON_SDK_ROOT").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".hexagon-sdk/6.4.0.2"));
    assert!(sdk.join("incs/remote.h").exists(), "Hexagon SDK not found at {}: set HEXAGON_SDK_ROOT", sdk.display());
    let tools = sdk.join("tools/HEXAGON_Tools/19.0.04/Tools");
    let (clang, qurt) = (tools.join("bin/hexagon-clang"), sdk.join("rtos/qurt/computev79/include"));
    let inc = |p: &std::path::Path| format!("-I{}", p.display());
    run(Command::new(sdk.join("ipc/fastrpc/qaic/bin/qaic")).arg("-mdll").arg("-o").arg(out).arg(inc(&sdk.join("incs"))).arg(inc(&sdk.join("incs/stddef"))).arg("dsp/aether_dsp.idl"));
    let flags = ["-mcpu=hexagonv79", "-mv79", "-mhvx=v79", "-mhvx-length=128b", "-O2", "-G0", "-fpic", "-Wall", "-Werror"];
    let incs = [out.to_path_buf(), qurt.clone(), qurt.join("qurt"), qurt.join("posix"), sdk.join("incs"), sdk.join("incs/stddef"), sdk.join("ipc/fastrpc/incs")];
    let mut objs = Vec::new();
    for src in [PathBuf::from("dsp/aether_dsp.c"), out.join("aether_dsp_skel.c")] {
        let obj = out.join(src.file_stem().unwrap()).with_extension("o");
        run(Command::new(&clang).args(flags).args(incs.iter().map(|p| inc(p))).arg("-c").arg(&src).arg("-o").arg(&obj));
        objs.push(obj);
    }
    let lib = tools.join("target/hexagon/lib");
    run(Command::new(&clang)
        .args(["-mcpu=hexagonv79", "-mv79", "-G0", "-fpic", "-shared", "-Wl,-Bsymbolic", "-Wl,-soname,libaether_dsp.so"])
        .arg(format!("-Wl,-L{}", lib.join("v79/G0/pic").display()))
        .arg(format!("-Wl,-L{}", lib.display()))
        .arg("-o")
        .arg(out.join("libaether_dsp.so"))
        .args(&objs)
        .arg("-lc"));
}

fn shaders(out: &std::path::Path) {
    println!("cargo:rerun-if-changed=shaders");
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
