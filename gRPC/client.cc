// greeter_client.cc 
#include <grpcpp/grpcpp.h>

int main() {
 
  GreeterClient greeter(grpc::CreateChannel(
     "localhost:50051", grpc::InsecureChannelCredentials()));

  std::string user("world");
  std::string reply = greeter.SayHello(user); 
  
  std::cout << "Greeter received: " << reply << std::endl;
  
  return 0;
}