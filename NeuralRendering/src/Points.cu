#include "Points.hpp"
#include <fstream>
#include <vector>
#include "CudaIncludesAndMacros.cuh"
#include "stream_binary_utils.hpp"
#include "CudaMath.cuh"
#if __CUDA_ARCH__ < 600
static_assert(false, "only 6.0 or greater currently supported.")
#endif
cudaError_t GPUPoints::writeToFile(const char* path, int vers)
{
	cudaError_t cudaStatus = cudaSuccess;
	switch (vers) {
	case 0x600:case 0x610: {
		//Positions file
		{
			size_t num_bytes = sizeof(float)* 3 * num_entries;
			std::unique_ptr<float> fvalues = std::unique_ptr<float>((float*)malloc(num_bytes));
			if (fvalues == NULL) return cudaError::cudaErrorMemoryAllocation;
			cudaStatus = cudaMemcpy(fvalues.get(), position_memory_start, num_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			STATUS_CHECK();
			std::ofstream f{ path,std::ios::binary };
			f.write((char*)fvalues.get(), num_bytes);
		}
		//Colors file
		{
			size_t num_bytes = sizeof(byte) * 3 * num_entries;
			std::unique_ptr<byte> cvalues = std::unique_ptr<byte>((byte*)malloc(num_bytes));
			if (cvalues == NULL) return cudaError::cudaErrorMemoryAllocation;
			cudaStatus = cudaMemcpy(cvalues.get(), color_memory_start, num_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			STATUS_CHECK();
			std::ofstream f{ std::string(path) + ".colors",std::ios::binary };
			f.write((char*)cvalues.get(), num_bytes);
		}
		//environment file
		if (environment != nullptr) {
			int size = environment->resolution * environment->resolution * (environment->data_is_bytes ? sizeof(char) : sizeof(float)) * environment->dimensions * 6;
			auto to_output = std::unique_ptr<char>((char*)malloc(size));
			cudaMemcpy(to_output.get(), environment->data, size, cudaMemcpyDeviceToHost);
			std::ofstream f{ std::string(path) + ".environment",std::ios::binary };
			int tmp = environment->resolution;
			f.write((char*)&tmp, sizeof(tmp));
			tmp = (environment->data_is_bytes ? 0 : 1);
			f.write((char*)&tmp, sizeof(tmp));
			f.write((char*)to_output.get(), size);
		}
	}break;
	default: {
		std::cerr << "Point cloud not saved! Version " << std::hex << vers << std::dec << " not supported!\n";
	}break;
	}
Error:
	return cudaStatus;
}

cudaError_t GPUPoints::read(const char* path, GPUPoints& dest, int vers)
{
	std::shared_ptr<Environment> environment = nullptr;
	cudaError_t cudaStatus = cudaSuccess;
	std::vector<float> positions;
	std::vector<unsigned char> colors;
	switch (vers)
	{
	case 0x500: {
		std::ifstream f{ path };
		if (f.fail())return cudaError::cudaErrorFileNotFound;
		int size = -1, c = 0;
		if (!(f >> size >> c))return cudaError::cudaErrorUnknown;
		if (size != -1) { positions.reserve(size); colors.reserve(size); }
		if (c == 0) { c = 1; }
		float x, y, z; int r, g, b;

		while (f >> x >> y >> z >> r >> g >> b)
		{
			positions.push_back(x);
			positions.push_back(y);
			positions.push_back(z);
			colors.push_back(clamp(r / c, 0, 255));
			colors.push_back(clamp(g / c, 0, 255));
			colors.push_back(clamp(b / c, 0, 255));
		}
		if (f.fail() && !f.eof())return cudaError::cudaErrorUnknown;
	}break;
	case 0x600:case 0x610: {
		std::ifstream f1{ path, std::ios::binary };
		if (f1.fail())return cudaError::cudaErrorFileNotFound;
		readBinaryIntoArray(positions, f1, -1);
		f1.close();
		std::ifstream f2{ std::string(path) + ".colors", std::ios::binary };
		if (f2.fail())return cudaError::cudaErrorFileNotFound;
		readBinaryIntoArray(colors, f2, -1);
		//There may also be a .environment file
		std::ifstream f3{ std::string(path) + ".environment", std::ios::binary };
		if (!f3.fail()) {
			int resolution = readOneBinary<int>(f3);
			bool as_bytes = (0==readOneBinary<int>(f3));
			environment = std::make_shared<Environment>(resolution,as_bytes);
			if (as_bytes) {
				std::vector<unsigned int> bytes;
				readBinaryIntoArray(bytes, f3, -1);
				const int expected = sizeof(uchar4) * environment->resolution * environment->resolution * 6;
				if (bytes.size() * sizeof(unsigned int) != expected) {
					std::cerr << "Byte environment invalid\n";
					return cudaError::cudaErrorUnknown;
				}
				cudaMemcpy(environment->data, &bytes[0], expected, cudaMemcpyHostToDevice);
			} else {
				std::vector<float> floats;
				readBinaryIntoArray(floats, f3, -1);
				const int expected = sizeof(float4) * environment->resolution * environment->resolution * 6;
				if (floats.size() * sizeof(float) != expected) {
					std::cerr << "Float environment invalid\n";
					return cudaError::cudaErrorUnknown;
				}
				cudaMemcpy(environment->data, &floats[0], expected, cudaMemcpyHostToDevice);
			}
		}
	}break;
	default:
		break;
	}
#ifdef _DEBUG
	if (positions.size() != colors.size()) {
		std::cerr << "Not the same number of points as colors\n";
		return cudaError::cudaErrorUnknown;
	}
#endif
	positions.shrink_to_fit();
	colors.shrink_to_fit();
	int sz = (int) positions.size();
	void* gpu_pos;
	void* gpu_col;
	cudaStatus = cudaMalloc(&gpu_pos, sz * sizeof(float));
	if (cudaStatus != cudaSuccess) { cudaFree(gpu_pos); }
	STATUS_CHECK();
	cudaStatus = cudaMemcpy(gpu_pos, positions.data(), sz * sizeof(float),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { cudaFree(gpu_pos); }
	STATUS_CHECK();
	cudaStatus = cudaMalloc(&gpu_col, sz * sizeof(char));
	if (cudaStatus != cudaSuccess) { cudaFree(gpu_pos); }
	STATUS_CHECK();
	cudaStatus = cudaMemcpy(gpu_col, colors.data(), sz * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { cudaFree(gpu_pos); cudaFree(gpu_col); }
	STATUS_CHECK();
	dest = GPUPoints(gpu_pos, gpu_col, sz / 3, true, environment);
Error:
	return cudaStatus;
}
GPUPoints::~GPUPoints()
{
	if (should_free) {
		cudaFree(color_memory_start);
		cudaFree(position_memory_start);
	}
}

Environment::Environment(int resolution, bool as_bytes, int dimensions) :resolution{ resolution }, data_is_bytes{ as_bytes }, dimensions{ dimensions }
{
	if (data_is_bytes) {
		grad = NULL;
		cudaError_t error = cudaMalloc(&data, sizeof(char) * dimensions * resolution * resolution * 6);//6 faces on a cubemap.
		if (error != cudaSuccess) { this->resolution = 0; data = 0; }
		error = cudaMemset(data, 0, sizeof(char) * dimensions * resolution * resolution * 6);//6 faces on a cubemap.
		if (error != cudaSuccess) { this->resolution = 0; data = 0; }
	} else {
		grad = NULL;
		cudaError_t error = cudaMalloc(&data, sizeof(float) * dimensions * resolution * resolution * 6);//6 faces on a cubemap.
		if (error != cudaSuccess) { this->resolution = 0; data = 0; }
		error = cudaMemset(data, 0, sizeof(float) * dimensions * resolution * resolution * 6);//6 faces on a cubemap.
		if (error != cudaSuccess) { this->resolution = 0; data = 0; }
	}
}
cudaError_t Environment::ensure_grad() {
	if (grad == NULL) {
		cudaError_t error = cudaMalloc(&grad, sizeof(float) * dimensions * resolution * resolution * 6);//6 faces on a cubemap.
		zero_grad();
		return error;
	}
	return cudaSuccess;
}

cudaError_t Environment::zero_grad()
{
	if (grad != NULL)
		return set_all_to_zeros(grad, sizeof(float) * dimensions * resolution * resolution * 6);
	return cudaError::cudaErrorIllegalState;
}

#include "cuda_debug_utils.cuh"

cudaError_t Environment::merge_grad()
{
	if (grad != NULL) {
		//printGPUMemoryCoord<float>(grad, 4 * resolution * resolution * 6, [](float x) {return x!=0.0f; }, 4, 4 * resolution, 4 * resolution * resolution);
		return apply_gradient(data, grad, (size_t) dimensions * resolution * resolution * 6, this->data_is_bytes);
	}
	return cudaError::cudaErrorIllegalState;
}


Environment::~Environment()
{
	if (grad != NULL) {
		cudaFree(grad);
	}
	if (data != NULL) {
		cudaFree(data);
	}
}
