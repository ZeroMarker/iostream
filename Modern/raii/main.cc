#include <iostream>
#include <fstream>

class File {
  std::string name;
  std::fstream stream;
  
  public:
  File(const std::string& name) : name{name} { 
    // Acquire resource
    stream.open(name);  
    std::cout << "File opened" << std::endl;
  } 
  
  ~File() {
    // Release resource 
    stream.close(); 
    std::cout << "File closed" << std::endl;
  }
};

int main() {

  {
    File myFile{"data.txt"}; 
    // File instance created  
    // Resource acquired at construction

    // Use file stream

    // Stream automatically closed
    // Resource released at destruction
  }

  std::cout << "End of Scope" << std::endl; 

  return 0;
}