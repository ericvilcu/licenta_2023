#pragma once
#include "cuda.h"
#include <memory>
#include "HeaderThatSupressesWarnings.h"
#include "cuda_runtime.h"
#include "HeaderThatReenablesWarnings.h"


//Contains far points or backgrounds.
//These would usually be uninitialized and then get their values from training.
//todo: make more things potentially have multiple channels.
class Environment {
public:
	int resolution;
	void* data;
	void* grad;
	bool data_is_bytes;//This would really overcomplicate things and may be worth it.
	int dimensions;
	Environment(int resolution = 1024, bool as_bytes = false, int dimensions = 4);
	cudaError_t ensure_grad();
	cudaError_t zero_grad();
	cudaError_t merge_grad();
	~Environment();
};
//-todo: separate into chunks? May be a little too much for now.
class GPUPoints {
public:
	std::shared_ptr<Environment> environment = nullptr;
	int num_entries=-1;
	void* position_memory_start=NULL;
	void* color_memory_start=NULL;
	bool should_free = false;
	bool colors_are_bytes = true;
	void* position_grad_memory_start = NULL;
	void* color_grad_memory_start = NULL;

public:
	GPUPoints() noexcept
		:num_entries{ 0 }, position_memory_start{ NULL }, color_memory_start{ NULL }, should_free{ false } {};
	GPUPoints(void* gpu_position_memory, void* gpu_color_memory, int num_entries, bool should_free=true, std::shared_ptr<Environment> environment = nullptr) noexcept
		:num_entries{ num_entries }, position_memory_start{ gpu_position_memory }, color_memory_start{ gpu_color_memory }, should_free{ should_free }, environment{ environment } {};
	GPUPoints(GPUPoints&& ot) noexcept :num_entries{ ot.num_entries }, position_memory_start{ ot.position_memory_start }, color_memory_start{ ot.color_memory_start }, should_free{ ot.should_free } {ot.should_free = false; };
	GPUPoints(GPUPoints& ot) = delete;//todo?
	GPUPoints(const char* pth, int vers) {
		GPUPoints::read(pth, *this, vers);
	}
	GPUPoints& operator=(GPUPoints& ot) = delete;
	GPUPoints& operator=(GPUPoints&& ot) noexcept { 
		this->~GPUPoints();
		num_entries = ot.num_entries;
		position_memory_start = ot.position_memory_start;
		color_memory_start = ot.color_memory_start;
		should_free = ot.should_free;
		ot.should_free = false;
		return *this;
	}

	cudaError_t writeToFile(const char* path, int vers);
	static cudaError_t read(const char* path, GPUPoints& dest, int vers);
	

	//toto: switch to using tensors and autograd, this is ridiculous.
	//UNIMPLEMENTED adds gradient to the points and the background points
	//currently only implemented for environment
	void merge_grad() {
		if (environment != nullptr)environment->merge_grad();
	}
	//UNIMPLEMENTED resets gradient of the points and the background points to 0
	//currently only implemented for environment
	void zero_grad() {
		if (environment != nullptr)environment->zero_grad();
	}
	//UNIMPLEMENTED ensures a gradient exists for the points and the background points
	//currently only implemented for environment
	void ensure_grad() {
		if (environment != nullptr)environment->ensure_grad();
	}

	void ensure_environment() {
		if (environment == nullptr)environment = std::make_shared<Environment>();
	}
	~GPUPoints();
};
