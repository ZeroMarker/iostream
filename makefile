all: hello run clean
hello:
	g++ -std=c++20 main.cc
run:
	./a.exe
clean:
	rm *.exe