#pragma once
#include "DataSet.hpp"
#include "PlotPoints.cuh"

class PlotterImpl:public torch::nn::Module {
	std::vector<DataModule> sources;
public:
	PlotterImpl(std::shared_ptr<DataSet> initial_sources) {
		train(true);
		for (int i = 0; i < initial_sources->num_scenes(); ++i) {
			this->sources.emplace_back(initial_sources->scene(i).dataModule());
		}
		for (int idx = 0; idx < sources.size(); ++idx) {
			register_module(std::string("Data_") + std::to_string((int)sources[idx]->getType()) + "_" + std::to_string(idx), sources[idx]);
		}
	}

	PlotterImpl(std::vector<DataModule>&& initial_sources) {
		train(true);
		this->sources = std::move(initial_sources);
		for (int idx = 0; idx < sources.size(); ++idx) {
			register_module(std::string("Data_") + std::to_string((int)sources[idx]->getType()) + "_" + std::to_string(idx), sources[idx]);
		}
	}

	torch::Tensor forward(torch::Tensor what, std::shared_ptr<CameraDataItf> view, bool train = true);

	void load(torch::serialize::InputArchive& in) override {
		sources.clear();
		for (const auto& key : in.keys()) if(key.find("Data") == 0){

			torch::serialize::InputArchive sub_archive;
			in.read(key, sub_archive);
			size_t first_ = key.find('_');
			size_t second_ = key.find('_', first_ + 1ULL);
			DATA_MODULE_TYPE_T type = std::stoi(key.substr(first_ + 1ULL, second_ - first_ - 1ULL));
			int idx = std::stoi(key.substr(second_ + 1ULL));
			switch (type)
			{
			case SIMPLE:
				sources.emplace_back(
					register_module(std::string("Data_") + std::to_string((int)sources[idx]->getType()) + "_" + std::to_string(idx), DataModule())
				);
				sources.back()->load(sub_archive);
				break;
			default:
				throw std::string("Unknown data module type: ") + std::to_string((int)type);
				break;
			}
			

		}
	}

	void save(torch::serialize::OutputArchive& out) const override {
		for (auto& data : named_modules()) {
			torch::serialize::OutputArchive sub_archive;
			(*data)->save(sub_archive);
			out.write(data.key(), sub_archive);
		}
	}

	void set_train(bool train) {
		for (auto& s : sources)
			s->set_train(train);
		this->train(train);
	}
};

TORCH_MODULE(Plotter);


