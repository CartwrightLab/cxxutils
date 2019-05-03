#include "../sparkyrng.hpp"

extern "C" {
#include <unif01.h>
#include <bbattery.h>
};

sparkyrng::Random myrand;

unsigned int engine() {
    auto u = myrand.bits();
    return (u>>32);
}

unsigned int engine_bswap() {
    auto u = myrand.bits();
    u = __builtin_bswap64(u);
    return (u>>32);
}

double random_f52() {
    return myrand.f52();
}

double random_exp() {
    return exp(-myrand.exp());
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

    gen = unif01_CreateExternGen01 ("Random-exp", random_exp);
    bbattery_SmallCrush (gen);
    unif01_DeleteExternGen01 (gen);

    return 0;
}