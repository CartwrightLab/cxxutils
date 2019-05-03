#include "../sparkyrng.hpp"

extern "C" {
#include <unif01.h>
#include <bbattery.h>
};

sparkyrng::Random myrand;

double random_umax() {
    int64_t i = INT64_C(132799643625263);
    int64_t u = myrand.u64(i);
    
    return (1.0*u)/i;
}

int main() {
    unif01_Gen *gen;

    gen = unif01_CreateExternGen01 ("Random-umax", random_umax);
    bbattery_SmallCrush (gen);
    unif01_DeleteExternGen01 (gen);

    return 0;
}