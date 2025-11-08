//
// Created by ttft3 on 2023/4/12.
//
#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <memory>
#include <algorithm>
#include <stack>

using namespace std;

// member variable initialize
struct Data {
    int i = 1;
    float f = 0;
    bool b = true;
};

// enum
enum class Number {
    One, Two, Three
};

// Template C++ 14 auto
template<typename T, typename U>
auto add(T x, U y) { return x + y; }

// constexpr
constexpr int pow(int x, int y) {
    int result = x;
    while (--y) {
        result *= x;
    }
    return result;
}

// inline namespace
namespace Parent {
    void foo() { cout << "Parent foo()" << endl; };

    namespace Child1 {
        void foo() { cout << "V 1.0" << endl; }
    }
    inline namespace [[deprecated]] Child2 {
        void foo2() { cout << "V 2.0" << endl; }
    }
}
namespace Parent::Child1 {

}


int main() {
    // Null pointer
    char *s = nullptr;

    Number number = Number::One;

    vector<pair<int, int>> list = {{1, 1},
                                   {2, 2},
                                   {3, 3}};
    for(auto it = list.cbegin(); it != list.cend(); ++it) {
        cout << it->first << ", " << it->second << endl;
    }

    cout << add(1, 3.0) << endl;

    int a[pow(2, 4)];

    // stdlib initialize
    vector<int> v = {1, 2, 3};
    ::list<int> l = {1, 2, 3};
    set<int> ss = {1, 2, 3};

    // range
    map<int, string> numMap = {{1, "one"},
                               {2, "two"},
                               {3, "three"}};
    for(auto [key, value]: numMap) {
        cout << key << " -> " << value << endl;
    }

    // smart pointer
    unique_ptr<Data> data(new Data);
    auto some = make_unique<Data>();

    // lambda
    auto it = find_if(v.cbegin(), v.cend(), [](int x) { return x % 2 == 0; });

    // inline namespace
    Parent::foo();
    Parent::Child1::foo();
    Parent::foo2();

    //Parent::Child2::foo2(); // deprecated

    vector<int> input;
    vector<int> output;
    stack<int> stack;
    for(int i = 0; i < 10; i++) {
        input.push_back(i);
    }
    for(int i = 0; i < input.size(); i++) {
        stack.push(input[i]);
    }
    int size = stack.size();
    for(int i = 0; i < size; i++) {
        output.push_back(stack.top());
        stack.pop();
    }
    
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
