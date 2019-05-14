# MinionRNG

MinionRNG is a high-performance pseudo-random-number library suitable for scientific simulation.
It is based on the [xoshiro256\*\*](http://xoshiro.di.unimi.it/) algorithm developed by David Blackman and Sebastiano Vigna.

Copyright &copy; 2019 Reed A. Cartwright, PhD \<reed@cartwrig.ht\>

See below for example usage.

## Example

```CXX
#include <cstdio>

#include "minion.hpp"

// create a PRNG object with default seed
minion::Random mrand;

int main() {
    // create a 64-bit pre-seed using (1) well-mixed bits, (2) pid, (3) time, and (4) std::random_device
    uint64_t seed64 = minion::create_uint64_seed(minion::create_seed_seq());
    // reduce seed to 31 bits (not needed, but easier for users to work with)
    int seed = seed64 >> 33;
    // reseed mrand
    mrand.seed(seed);
    printf("Seed is %d\n",seed);

    // generate random values
    for(int i=0;i<1000;++i) {
        // uniformly distributed integers between [0,2^64)
        uint64_t u = mrand.u64();
        // uniformly distributed floating-point numbers between (0,1)
        double f = mrand.f52();

        printf("%d %lu %f\n", i, u, f);
    }
    return 0;
}
```
