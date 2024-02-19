// greeter_server.cc
#include <grpcpp/grpcpp.h>

class GreeterServiceImpl final : public Greeter::Service {

  std::string SayHello(std::string name) override {
    return "Hello " + name; 
  }
};

void RunServer() {

  std::string server_address("0.0.0.0:50051");

  GreeterServiceImpl service;

  grpc::ServerBuilder builder;

  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  
  server->Wait();
}

int main() {
  RunServer();

  return 0;
}
