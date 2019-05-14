#include "../minion.hpp"

extern "C" {
#include <unif01.h>
#include <bbattery.h>
};

minion::Random mrand;

unsigned int engine() {
    auto u = mrand.bits();
    return (u>>32);
}

unsigned int engine_bswap() {
    auto u = mrand.bits();
    u = __builtin_bswap64(u);
    return (u>>32);
}

double random_f52() {
    return mrand.f52();
}

double random_f53() {
    return mrand.f53();
}

int main() {
    unif01_Gen *gen;

    gen = unif01_CreateExternGenBits ("Random-bits", engine);
    bbattery_SmallCrush (gen);
    unif01_DeleteExternGenBits (gen);

    gen = unif01_CreateExternGenBits ("Random-bits-bswap", engine_bswap);
    bbattery_SmallCrush (gen);
    unif01_DeleteExternGenBits (gen);

    gen = unif01_CreateExternGen01 ("Random-f52", random_f52);
    bbattery_SmallCrush (gen);
    unif01_DeleteExternGen01 (gen);

    gen = unif01_CreateExternGen01 ("Random-f53", random_f53);
    bbattery_SmallCrush (gen);
    unif01_DeleteExternGen01 (gen);

    return 0;
}