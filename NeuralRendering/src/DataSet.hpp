#pragma once
#include <vector>
#include <random>
#include "Points.hpp"
#include "TrainImage.hpp"
#include "BlockingQueue.hpp"
#if __NVCC__
static_assert(false, "DO NOT INCLUDE THIS FILE FROM .cu FILES! IT HEAVILY UTILIZES 'typedef' WHICH CURRENTLY BREAKS NVCC COMPILED FILES!")
#endif

typedef char DATA_MODULE_TYPE_T;
enum DATA_MODULE_TYPE :char {
	INVALID = 0,
	SIMPLE,
};

//Made as there could potentially be multiple types of Images or Image generators
class IDataModuleImpl : public torch::nn::Module {
public:
	virtual torch::Tensor pointData(torch::Tensor spec) { return {}; };
	virtual torch::Tensor environmentData(torch::Tensor spec) { return {}; };
	virtual DATA_MODULE_TYPE_T getType() { return 0; };
};

class DataModuleImpl : public IDataModuleImpl {
	torch::Tensor point_data;
	torch::Tensor environment_data;
	static torch::Tensor readBinFrom(std::string path, bool require_grad);
	static torch::Tensor readTxtFrom(std::string path, bool require_grad);
public:
	DataModuleImpl(const std::string path, fileType_t t, bool train = true) {
		switch (t) {
		case CUSTOM_BINARY:
			this->point_data = register_buffer("point_data", readBinFrom(path + "/points.bin", train));
			this->environment_data = register_buffer("environment_data", readBinFrom(path + "/environment.bin", train));
			break;
		case TEXT:
			this->point_data = register_buffer("point_data", readTxtFrom(path + "/points.txt", train));
			this->environment_data = register_buffer("environment_data", readTxtFrom(path + "/environment.txt", train));
			break;
		case TORCH_ARCHIVE: {
			torch::serialize::InputArchive archive; archive.load_from(path + "/points");
			archive.read("points", this->point_data);
			archive.read("environment", this->environment_data);
			this->point_data = register_buffer("point_data", this->point_data);
			this->environment_data = register_buffer("environment_data", this->environment_data);
		}break;
		default:
			std::cerr << t << "is invalid";
			throw "invalid t";
		}
		this->train(train);
	}
	DataModuleImpl(bool train = true) {
		this->point_data = register_buffer("point_data", torch::zeros({ 0 }));
		this->environment_data = register_buffer("environment_data", torch::zeros({ 0 }));
		this->train(train);
	}
	DataModuleImpl(torch::Tensor point_data, torch::Tensor environment_data, bool train = true) {
		this->point_data = register_buffer("point_data", point_data);
		this->environment_data = register_buffer("environment_data", environment_data);
		this->train(train);
	}
	virtual torch::Tensor pointData(torch::Tensor) override { return point_data; };
	virtual torch::Tensor environmentData(torch::Tensor) override { return environment_data; };
	virtual DATA_MODULE_TYPE_T getType() override { return SIMPLE; }
	bool is_valid() {
		//may want to add more stuff here...
		return point_data.sizes().size() == 2 && environment_data.sizes().size() == 4;//point/channel and cube_face/x/y/channel
	}
	static DataModuleImpl load_from_file(std::string file) {
		//TODO
	}
};

TORCH_MODULE(DataModule);


class Scene {
private:
	DataModule point_data;
	int number_of_train_images;
	const std::string path;
	fileType_t file_type;
public:
	int index = 0;
	//std::shared_ptr<GPUPoints> points;
	void load_points(bool quiet = true);
	void determine_image_count(bool quiet = true);
	Scene(const std::string path, int index, bool autoload = true, bool load_train_data = true, bool quiet = true)
		:number_of_train_images{ -1 },
		path{ path },
		point_data{ nullptr },
		index{ index },
		file_type{ detect_file_type() } {
		determine_image_count(quiet);
		if (autoload) {
			load_points(quiet);
		}
	}
	int num_train_images() { return number_of_train_images; };
	std::shared_ptr<TrainingImage> loadOne(int idx, bool autoload=true);
	fileType_t detect_file_type();
	void save(const std::string& path, fileType_t mode, bool save_training_data=false);
	bool isValid() {
		if (!point_data->is_valid())
			return false;
		return true;
	}
	DataModule dataModule() { return point_data; }
};


class DataSet {
	typedef BlockingQueue<std::shared_ptr<TrainingImage>> ImageQueue;
	typedef std::pair<int, int> img_id;//todo: dto?
	typedef ImageQueue::producer<img_id> ImageLoader;
	std::vector<Scene> scenes;
	//Note: keep these 4 in this order or else they do not get properly destroyed and throw ugly errors
	std::unique_ptr<ImageQueue> train_image_provider = nullptr, show_image_provider = nullptr;
	std::unique_ptr<ImageLoader> trainImageLoader = nullptr, showImageLoader = nullptr;
	bool loadsTrainData = true;
	DataSet(bool load_train_data)
		:scenes{},
		loadsTrainData{ load_train_data } {
	}
	void initializeLoaders();
public:
	std::pair<img_id, std::shared_ptr<TrainingImage>> loadImageOf(img_id id) {
		//restricts to 1 scene for now.
		return std::make_pair(img_id(0, (id.second + 1) % scene(0).num_train_images()), scene(0).loadOne(id.second));
	};
	std::pair<img_id, std::shared_ptr<TrainingImage>> loadImageOfRand(img_id id) {
		//restricts to 1 scene for now.
		return std::make_pair(img_id(0, ((unsigned int)rand()) % scene(0).num_train_images()), scene(0).loadOne(id.second));
	};

	DataSet(const std::vector<std::string>& paths, bool autoload, bool loadTrainData = true, bool quiet = true)
		:
		//restricts to 1 scene for now.
		scenes{ {paths[0], 0, autoload, loadsTrainData, quiet} },
		loadsTrainData{ loadTrainData } {
		initializeLoaders();
		/*for (const auto& path : paths) {
			scenes.emplace_back(path, autoload, loadsTrainData, max_samples_each, max_points_each);
		}*/
	}
	size_t num_scenes() { return scenes.size(); }
	std::pair<Scene&, std::shared_ptr<TrainingImage>> next_example() {
		//restricts to 1 scene for now.
		auto img = show_image_provider->pop();
#if _DEBUG
		if (img == nullptr)
			throw "IMAGE BUFFER GOT INVALID OUTPUT";
#endif
		return { scene(0), img };
	}
	std::pair<Scene&, std::shared_ptr<TrainingImage>> next_train() {
		//restricts to 1 scene for now.
		auto img = train_image_provider->pop();
#if _DEBUG
		if (img == nullptr)
			throw "IMAGE BUFFER GOT INVALID OUTPUT";
#endif
		return { scene(0), img };
	}
	//Expensive
	bool isValid() {//todo: check 1 train show as well or something
		if (scenes.size() < 0)return false;

		for (auto& scene : scenes) {
			if (!scene.isValid())return false;
		}
		return true;
	}
	bool hasTrainData() {
		return loadsTrainData;
	}
	Scene& scene(int idx) {
		return scenes[idx];
	}

	int save(const std::string& path, fileType_t mode, bool saveTrainingImages);
	static std::unique_ptr<DataSet> load(const std::string& path, bool loadTrainingimagesIfPresent = true, bool quiet = true);

};