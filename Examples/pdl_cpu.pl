
use v5.30;
use PDL;
use Getopt::Long;
use Time::HiRes qw(time);

sub compute_inplace {
    my ($array) = @_;
    $array->inplace->sqrt;
    $array->inplace->sin;
    $array->inplace->cos;
}

sub main {
    my $arraysize = 100_000_000;
    GetOptions(
        "arraysize=i" => \$arraysize,
    ) or die "usage: $0 [ --arraysize=N ]\n";

    # Generate the data structures for the benchmark
    my $array0 = random(float, $arraysize);
    my $array_copy = $array0->copy;

    for (1 .. 10) {
        my $start_time = time();
        compute_inplace($array_copy);
        my $elapsed_time = time() - $start_time;
        printf("%12.3f µs\n", $elapsed_time * 1e6);
    }
}

say "=" x 50;
say "In place in (Perl PDL - many CPU threads)";
say "=" x 50;

main();

