#include <cstdio>

#include "random.hpp"

// create a PRNG object with default seed
fragmites::random::Random mrand;

int main() {
    // create a 32-bit seed
    auto seed_seq = fragmites::random::auto_seed_seq();
    // seed mrand
    mrand.Seed(seed_seq);
    auto seed = mrand.GetSeed();
    std::string seed_str = fragmites::random::encode_seed(seed);

    printf("Seed is %s\n",seed_str.c_str());

    // generate random values
    for(int i=0;i<1000;++i) {
        uint64_t u = mrand.u64();
        double f = mrand.f52();
        printf("%d %lu %f\n", i, u, f);
    }
    return 0;
}
