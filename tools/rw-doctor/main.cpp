#include "rw/rw.hpp"
#include <iostream>

int main() {
    try {
        rw::Runtime runtime;
        std::cout << runtime.capabilities() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
