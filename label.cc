#include <iostream>

int main() {
    int i = 0;

start_loop://
    if (i < 5) {
        std::cout << i << std::endl;
        i++;
        goto start_loop; // Jump back to the start_loop label
    }
    return 0;
}
