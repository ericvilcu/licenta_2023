#include "PlotterModule.hpp"

#include "PlotPointsBackwardsPasses.cuh"

class PlotFunction :public torch::autograd::Function<PlotFunction> {
public:

	static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor points, torch::Tensor environment, std::shared_ptr<CameraDataItf> view) {
		void* tmp_plot;
		void* tmp_weights;
		int ndim = points.size(-1) - 3;
		//TODO: check if environment is defined
		int environment_resolution = environment.size(1);
		plotPointsToGPUMemory_v2(view, ndim,
			points.data_ptr<float>(), points.size(0),
			environment.data_ptr<float>(), environment_resolution,
			(float**)&tmp_plot, false, (float**)&tmp_weights, false);
		torch::Tensor ret = torch::from_blob(tmp_plot, { view->get_height(),view->get_width(), ndim + 1LL }, cudaFree, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA).requires_grad(points.requires_grad() || environment.requires_grad()));
		torch::Tensor weights = torch::from_blob(tmp_weights, { view->get_height(),view->get_width() }, cudaFree, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
		//this works but is even more stupid. the serialize trick might have been smarter.
		//ctx->saved_data["cam_tsr"] = torch::from_blob(new std::shared_ptr<CameraDataItf>(view), { 1 }, [](void* mem) {delete (std::shared_ptr<CameraDataItf>*)mem; }, torch::TensorOptions());
		ctx->saved_data["cam_bin"] = view->serialized(false);
		ctx->save_for_backward({ points,environment,ret,weights });
		return ret;
	}
	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list image_gradient) {
		//I WOULD say to try to write one for the camera as well, and that is what ADOP does, but I'm not sure how to manage that.
		auto saved_stuff = ctx->get_saved_variables();
		torch::Tensor points = saved_stuff[0];
		torch::Tensor environment = saved_stuff[1];
		torch::Tensor ret = saved_stuff[2];
		torch::Tensor weights = saved_stuff[2];
		//auto tsr_cam = ctx->saved_data["cam_tsr"].toTensor();
		//std::shared_ptr<CameraDataItf> camera = *((std::shared_ptr<CameraDataItf>*)tsr_cam.data_ptr<float>());
		std::shared_ptr<CameraDataItf> camera = CameraDataItf::from_serial(false, ctx->saved_data["cam_bin"].toString()->string());
		int ndim = points.size(-1) - 3;
		torch::Tensor gradient_points = torch::zeros(points.sizes(), torch::TensorOptions().device(torch::kCUDA)).contiguous();
		torch::Tensor gradient_environment = torch::zeros(environment.sizes(), torch::TensorOptions().device(torch::kCUDA)).contiguous();
		image_gradient[0] = image_gradient[0].cuda().contiguous();
		cudaError_t cudaStatus = cudaError::cudaSuccess;
		int environment_resolution = environment.size(1);
		cudaStatus=PlotPointsBackwardsPass_v2(camera, ndim,
			points.data_ptr<float>(), gradient_points.data_ptr<float>(), points.size(0),
			environment.data_ptr<float>(), gradient_environment.data_ptr<float>(), environment_resolution,
			ret.data_ptr<float>(), weights.data_ptr<float>(), image_gradient[0].data_ptr<float>());
		torch::Tensor undefined;
		if (cudaStatus != cudaError::cudaSuccess) {
			std::cerr << "CUDA ERROR IN BACKWARDS PASS:" << cudaGetErrorString(cudaStatus) << '\n';
			return { undefined, undefined, undefined };
		}
		return { gradient_points, gradient_environment, undefined };
	}
};

torch::Tensor PlotterImpl::forward(torch::Tensor what, std::shared_ptr<CameraDataItf> view, bool train) {
	void* tmp_plot;
	int idx = what[0].item<int>();
	auto points = sources[idx]->pointData(what);
	auto environment = sources[idx]->environmentData(what);
	auto ret = PlotFunction::apply(points, environment, view);
	return ret;
}