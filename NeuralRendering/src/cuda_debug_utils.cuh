#include <iostream>
#include <functional>
#include "cuda.h"
#include "HeaderThatSupressesWarnings.h"
#include "cuda_runtime.h"
#include "HeaderThatReenablesWarnings.h"
template <typename T>
inline void printGPUMemory(void* dev, int size, int block_size = 1) {
	T* loc = new T[size];
	cudaError_t err = cudaMemcpy(loc, dev, sizeof(T) * size,cudaMemcpyDeviceToHost);
	if (err) { std::cerr << "error: \"" << cudaGetErrorString(err) << "\" in \"printGPUMemory\"\n"; return; }
	for (int i = 0; i < size;) {
		while ((i+1) % block_size != 0) {
			std::cerr << loc[i++]<< ' ';
		};
		std::cerr << loc[i++] << '\n';
	}
	delete[] loc;
}

template <typename T>
inline void printGPUMemoryCoord(void* dev, int total, std::function<bool(T)> cond, int n = 1, int m = 1, int q = 1) {
	T* loc = new T[total];
	cudaError_t err = cudaMemcpy(loc, dev, sizeof(T) * total, cudaMemcpyDeviceToHost);
	bool printed_something = false;
	if (err) { std::cerr << "error: \"" << cudaGetErrorString(err) << "\" in \"printGPUMemory\"\n"; return; }
	for (int i = 0; i < total;++i) {
		if (cond(loc[i])) {
			printed_something = true;
			std::cerr << loc[i] << " at " << i << '(' << i % n << ' ' << (int)(i / n) % (m) << ' ' << (int)(i / (m)) % q << ' ' << (int)(i / (q)) << ")\n";
		}
	}
	delete[] loc;
	if (!printed_something)std::cerr << "there was nothing.\n";
}