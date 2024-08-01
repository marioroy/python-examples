
This repo provides complementary demonstrations for a web article found on the web, by Christos Argyropoulos.

[The Quest for Performance Part II : Perl vs Python](https://chrisarg.github.io/Killing-It-with-PERL/2024/07/07/The-Quest-For-Performance-Part-II-PerlVsPython.md.html)

The inspiration came from reading [Comparison of various GPU acceleration frameworks using matrix-vector multiplication](https://github.com/99991/matvec-gpu), by Thomas Germer. If you're trying Mr. Germer's repo, comment out the line `ti.loop_config(block_dim=N)` for better performance on desktop GPUs. More over, set `block_size = 32`.

