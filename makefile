build:
	nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -o finalmakbiliproject  main.o cudaFunctions.o /usr/local/cuda-11.0/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./finalmakbiliproject

run:
	mpiexec -np 2 ./finalmakbiliproject <input3.txt >output.txt
