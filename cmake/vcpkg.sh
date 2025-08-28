sudo apt update
sudo apt install build-essential cmake git curl tar zip unzip

git clone https://github.com/microsoft/vcpkg.git

cd vcpkg
./bootstrap-vcpkg.sh

