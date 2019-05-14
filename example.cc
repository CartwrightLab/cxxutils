#include <cstdio>

#include "minion.hpp"

// create a PRNG object with default seed
minion::Random mrand;

int main() {
    // create a 64-bit pre seed
    uint64_t seed64 = minion::create_uint64_seed(minion::create_seed_seq());
    // reduce seed to 31 bits (not needed, but easier for users to work with)
    int seed = seed64 >> 33;
    // seed mrand
    mrand.seed(seed);
    printf("Seed is %d\n",seed);

    // generate random values
    for(int i=0;i<1000;++i) {
        uint64_t u = mrand.u64();
        double f = mrand.f52();
        printf("%d %lu %f\n", i, u, f);
    }
    return 0;
}
