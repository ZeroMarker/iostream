#include <iostream>
#include <vector>

int main(void) {
  long  l1 = -7856974990L;
  long long  l2 = 89565656974990LL;
  long l3 = 512'232'697'499;
  
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
  return 0;
}