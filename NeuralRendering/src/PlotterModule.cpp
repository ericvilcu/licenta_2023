#include "PlotterModule.hpp"

#include "PlotPointsBackwardsPasses.cuh"

class PlotFunction :public torch::autograd::Function<PlotFunction> {
public:

	static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor points, torch::Tensor environment, std::shared_ptr<CameraDataItf> view) {
		void* tmp_plot;
		int ndim = points.size(-1) - 3;
		plotPointsToGPUMemory_v2(tmp_plot, -1, -1, ndim, points.data_ptr<float>(), points.size(0), environment.data_ptr<float>(), environment.size(0), view, false);
		torch::Tensor ret = torch::from_blob(tmp_plot, { view->get_height(),view->get_width(), ndim+1LL }, cudaFree, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA).requires_grad(points.requires_grad() || environment.requires_grad()));
		//I should eventually modify it to be able to save a bitmap representing which are background points and such.
		//todo: modify to save more data;
		//todo: use intrusive pointers;
		//ctx->saved_data["cam"] = torch::IValue::make_capsule(torch::intrusive_ptr<CameraDataItf>(view->scaleTo(view->get_width(), view->get_height())));
		ctx->saved_data["cam"] = view->serialized(true);
		ctx->save_for_backward({ points,environment,ret });//bad.
		return ret;
	}
	static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list image_gradient) {
		//I WOULD say to try to write one for the camera as well, but that may be too much...
		auto saved_stuff = ctx->get_saved_variables();
		torch::Tensor points = saved_stuff[0];
		torch::Tensor environment = saved_stuff[1];
		torch::Tensor ret = saved_stuff[2];
		std::shared_ptr<CameraDataItf> camera = CameraDataItf::from_serial(true, ctx->saved_data["cam"].toString()->string());
		//std::cerr << camera->serialized(true) << '\n';

		int ndim = points.size(-1) - 3;
		torch::Tensor gradient_points = torch::zeros(points.sizes(), torch::TensorOptions().device(torch::kCUDA)).contiguous();
		torch::Tensor gradient_environment = torch::zeros(environment.sizes(), torch::TensorOptions().device(torch::kCUDA)).contiguous();

		image_gradient[0] = image_gradient[0].cuda().contiguous();
		PlotPointsBackwardsPass_v2(points.data_ptr<float>(), gradient_points.data_ptr<float>(), points.size(0),
			environment.data_ptr<float>(), gradient_environment.data_ptr<float>(), environment.size(1),
			camera, camera->get_height(), camera->get_width(), ndim,
			image_gradient[0].data_ptr<float>(), ret.data_ptr<float>());
		return { gradient_points,gradient_environment, torch::Tensor()/* aka undefined*/ };
	}
};

torch::Tensor PlotterImpl::forward(torch::Tensor what, std::shared_ptr<CameraDataItf> view, bool train) {
	void* tmp_plot;
	int idx = what[0].item<int>();
	auto points = sources[idx]->pointData(what);
	auto environment = sources[idx]->environmentData(what);
	auto ret = PlotFunction::apply(points, environment, view);
	//auto ret = torch::zeros({ view->get_height(),view->get_width(),points.size(-1) - 3 +1 });
	return ret;
}
