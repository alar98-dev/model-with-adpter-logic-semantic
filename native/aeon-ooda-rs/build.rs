fn main() {
    println!("cargo:rustc-link-search=native=/home/alifer/swa-aeon/native/aeon-ooda-rs/c_src");
    println!("cargo:rustc-link-lib=static=sumerian_tensor");
    println!("cargo:rerun-if-changed=c_src/sumerian_tensor.c");
    println!("cargo:rerun-if-changed=c_src/sumerian_tensor.h");
}
