#include <iostream>
#include <nlohmann/json.hpp>

// 使用简写命名空间
using json = nlohmann::json;

int main() {
    // 创建一个 JSON 对象
    json j = {
        {"pi", 3.141},
        {"happy", true},
        {"name", "Niels"},
        {"nothing", nullptr},
        {"answer", {
            {"everything", 42}
        }},
        {"list", {1, 0, 2}}
    };

    // 将 JSON 对象序列化为字符串并打印
    std::cout << j.dump(4) << std::endl;

    return 0;
}

