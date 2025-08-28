cmake_minimum_required(VERSION 3.14)
project(my_json_app)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.2 # Use the desired version tag
)

FetchContent_MakeAvailable(json)

add_executable(my_json_app main.cpp)

# Link your executable against the fetched library
target_link_libraries(my_json_app PRIVATE nlohmann_json::nlohmann_json)

