sudo snap install cmake --classic

cmake -S . -B build   # configure
cmake --build build   # build
./build/my_program    # run (adjust executable name)

