import helloworld;

int main() {
    hello();
    return 0;
}

// g++ -std=c++20 -fmodules-ts -xc++-system-header iostream
// compile iostream manually
// g++ -std=c++20 -fmodules-ts helloworld.cc main.cc -o hello