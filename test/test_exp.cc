#include "../minion.hpp"

extern "C" {
#include <unif01.h>
#include <bbattery.h>
};

minion::Random mrand;

double random_exp() {
    return exp(-mrand.exp());
}

int main() {
    unif01_Gen *gen;

    gen = unif01_CreateExternGen01 ("Random-exp", random_exp);
    bbattery_SmallCrush (gen);
    unif01_DeleteExternGen01 (gen);

    return 0;
}