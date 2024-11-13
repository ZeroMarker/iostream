cc = nvcc
src = hello
ext = cu

all: $(src) run clean
$(src): $(src).$(ext)
	$(cc) $^ -o $(src)
run:
	./$(src)
clean:
	rm ./$(src)