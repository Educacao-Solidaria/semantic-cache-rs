use semantic_cache_server::{alloc, banner, build_info};

fn main() {
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        println!("{}", build_info::version_line());
        return;
    }
    println!("{} (alocador: {})", banner(), alloc::ALLOCATOR);
}
