#include <format>
#include <iostream>

int main() {
    int age = 30;
    std::string name = "Alice";

    std::string msg = std::format("Hello, {}! You are {} years old.", name, age);
    std::cout << msg << '\n';

    // 支持格式说明符
    double pi = 3.1415926;
    std::cout << std::format("{:.2f}\n", pi);  // 保留两位小数：3.14
    return 0;
}
