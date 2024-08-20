#include <iostream>
#include <vector>
#include <print>
#include <cstdint>

int main(void) {
    long  l1 = -785697499L;
    long long  l2 = 89565656974990LL;
    long l3 = 232'697'499L;

    int a = 23;
    int b {34};
    auto x = a <=> b;
    if(x < 0) {
    std::cout << "Less" << '\n';
    }

    double   d {1.23456};  // OK
    float    f {2.53f};    // OK
    unsigned u {120u};     // OK
    double e {f};  // OK float → double

    bool m = true;
    bool n = false;
    bool z = m and n;

    std::vector<int> v1 {5, 2};
    std::vector<int> v2 (5, 2);

    v1.push_back(4);
    v1.resize(6, 0);

    enum class day { mon, tue, wed, thu, fri, sat, sun };
    day d1 = day::mon;

    int i = 2;
    if(int x = 2 * i; x < 10) {
    std::cout << x << '\n';
    }

    using real = double;
    real pi = 3.14;
    std::cout << pi << '\n';

    std::int16_t i16 = 1234;
    std::cout << i16 << '\n';

    double *p1 = nullptr;
    p1 = &pi;
    std::cout << *p1 << '\n';

    std::print("{0} {2}{1}!\n", "Hello", 23, "C++"); // overload (2)

    return 0;
}