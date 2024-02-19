#include <typeinfo>
#include <iostream>

class Base {
    virtual void dummy() {}
};

class Derived : public Base {
};

int main() {

    Base* b = new Derived; 
    
    // Show name of object's actual type
    std::cout << "Object is of type: " << typeid(*b).name() << std::endl;

    // Safe downcast from Base to Derived
    Derived* d = dynamic_cast<Derived*>(b); 
    
    // Check if downcast worked
    if(d == nullptr) {
        std::cout << "Cast failed" << std::endl;
    }
    else {
        std::cout << "Cast succeeded" << std::endl;
    }
    
    return 0;
}