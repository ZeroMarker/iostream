all: hello run clean
hello:
	g++ -std=c++23 main.cc
run:
	./a.exe
clean:
	rm *.exe

