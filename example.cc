#include <cstdio>

#include "sparkyrng.hpp"

// create a PRNG object with default seed
sparkyrng::Random sparky;

int main() {
    // create a 64-bit pre seed
    uint64_t seed64 = sparkyrng::create_uint64_seed(sparkyrng::create_seed_seq());
    // reduce seed to 31 bits (not needed, but easier for users to work with)
    int seed = seed64 >> 33;
    // seed sparky
    sparky.seed(seed);
    printf("Seed is %d\n",seed);

    // generate random values
    for(int i=0;i<1000;++i) {
        uint64_t u = sparky.u64();
        double f = sparky.f52();
        printf("%d %lu %f\n", i, u, f);
    }
    return 0;
}
