#include <cstdint>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

// number of bins
int n = 256;
// magic number
double r = 7.69711747013104972;

int main() {
    std::vector<double> x, y;
    x.resize(n+1);
    y.resize(n+1);

    double x1 = r;
    double y1 = exp(-x1);
    double t = exp(-x1);
    double A = x1*y1+t;


    x[0] = A/y1;
    y[0] = 0;
    x[1] = x1;
    y[1] = y1;
    for(int i=2;i<n;++i) {
        y[i] = y[i-1] + A/x[i-1];
        x[i] = -log(y[i]);
    }
    y[n] = 1.0;
    x[n] = 0.0;

    std::vector<int64_t> ek;
    std::vector<double>  ew;
    std::vector<double>  ef;
    ek.resize(n);
    ew.resize(n);
    ef.resize(n);
    for(int i=0;i<n;++i) {
        long double u = 9223372036854775808.0L;
        long double a = x[i+1];
        long double b = x[i];
        ek[i] = u*(a/b);
        ew[i] = x[i]/u;
        ef[i] = y[i+1];
        //printf("%d %ld %#0.17g %#0.17g\n",i, ek[i], ew[i], ef[i]);
    }

    printf("const double minion::detail::ew[256] = {\n");
    for(int i=0;i<n;++i) {
        if(i%4 == 0) {
            printf("\t");
        } else {
            printf(" ");
        }
        printf("%#0.17g,", ew[i]);
        if(i%4 == 3) {
            printf("\n");
        }
    }
    printf("};\n\n");

    printf("const double minion::detail::ef[256] = {\n");
    for(int i=0;i<n;++i) {
        if(i%4 == 0) {
            printf("\t");
        } else {
            printf(" ");
        }
        printf("%#0.17g,", ef[i]);
        if(i%4 == 3) {
            printf("\n");
        }
    }
    printf("};\n\n");

    printf("#define U UINT64_C\n");
    printf("const int64_t minion::detail::ek[256] = {\n");
    for(int i=0;i<n;++i) {
        if(i%4 == 0) {
            printf("\t");
        } else {
            printf(" ");
        }
        printf("U(%ld),", ek[i]);
        if(i%4 == 3) {
            printf("\n");
        }
    }
    printf("};\n");
    printf("#undef U\n");

    return 0;
}
