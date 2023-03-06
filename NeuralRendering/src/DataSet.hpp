#pragma once
#include <vector>
#include <random>
#include "Points.hpp"
#include "TrainImage.hpp"
#include "BlockingQueue.hpp"
#include "stream_binary_utils.hpp"
#ifdef __NVCC__
static_assert(false, "DO NOT INCLUDE THIS FILE FROM .cu FILES! IT HEAVILY UTILIZES 'typedef' WHICH CURRENTLY BREAKS NVCC COMPILED FILES!");
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
	static torch::Tensor readBinFrom(std::string path);
	static torch::Tensor readTxtFrom(std::string path);
	static void writeBinTo(std::string path, torch::Tensor what);
	static void writeTxtTo(std::string path, torch::Tensor what);
public:
	DataModuleImpl(const std::string& path, fileType_t t, bool train = true) {
		switch (t) {
		case CUSTOM_BINARY:
			this->point_data = register_parameter("point_data", readBinFrom(path + "/points.bin"), train);
			if (existsAndIsFile(path + "/environment.bin"))
				this->environment_data = register_parameter("environment_data", readBinFrom(path + "/environment.bin"), train);
			break;
		case TEXT:
			this->point_data = register_parameter("point_data", readTxtFrom(path + "/points.txt"), train);
			if(existsAndIsFile(path + "/environment.txt"))
				this->environment_data = register_parameter("environment_data", readTxtFrom(path + "/environment.txt"), train);
			break;
		case TORCH_ARCHIVE: {
			bool has_environment;
			torch::serialize::InputArchive archive; archive.load_from(path + "/points");
			archive.read("points", this->point_data);
			has_environment=archive.try_read("environment", this->environment_data);
			this->point_data = register_parameter("point_data", this->point_data, train);
			if (has_environment)this->environment_data = register_parameter("environment_data", this->environment_data, train);
		}break;
		default:
			std::cerr << t << "is invalid";
			throw "invalid t";
		}
		this->train(train);
	}
	void save(const std::string& path, fileType_t mode) {
		switch (mode) {
			case CUSTOM_BINARY:
				writeBinTo(path + "/points.bin", this->point_data);
				if (environment_data.defined())writeBinTo(path + "/environment.bin", this->environment_data);
				break;
			case TEXT:
				writeTxtTo(path + "/points.txt", this->point_data);
				if (environment_data.defined())writeTxtTo(path + "/environment.txt", this->environment_data);
				break;
			case TORCH_ARCHIVE: {
				torch::serialize::OutputArchive archive;
				archive.write("points", this->point_data);
				if (environment_data.defined())archive.write("environment", this->environment_data);
				archive.save_to(path + "/points");
			}break;
			default:
				std::cerr << "mode " << mode << " is invalid\n";
				throw "invalid mode";
		}
	}
	DataModuleImpl(bool train = true) {
		this->point_data = register_parameter("point_data", torch::zeros({ 0 }), train);
		this->environment_data = register_parameter("environment_data", torch::zeros({ 0 }), train);
		this->train(train);
	}
	DataModuleImpl(torch::Tensor point_data, torch::Tensor environment_data, bool train = true) {
		this->point_data = register_parameter("point_data", point_data, train);
		this->environment_data = register_parameter("environment_data", environment_data, train);
		this->train(train);
	}
	void expand_to_ndim(int ndim);
	void set_train(bool train) {
		this->point_data.set_requires_grad(train);
		this->environment_data.set_requires_grad(train);
		this->train(train);
	}
	virtual torch::Tensor pointData(torch::Tensor) override { return point_data; };
	virtual torch::Tensor environmentData(torch::Tensor) override { return environment_data; };
	virtual DATA_MODULE_TYPE_T getType() override { return SIMPLE; }
	bool is_valid() {
		//may want to add more stuff here...
		return point_data.sizes().size() == 2 && environment_data.sizes().size() == 4;//point/channel and cube_face/x/y/channel
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
	void expand_to_ndim(int ndim) { point_data->expand_to_ndim(ndim); };
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
	int train_scenes = 0;
	//Note: keep these 4 in this order or else they do not get properly destroyed and throw ugly errors
	std::unique_ptr<ImageQueue> train_image_provider = nullptr, show_image_provider = nullptr;
	std::unique_ptr<ImageLoader> trainImageLoader = nullptr, showImageLoader = nullptr;
	bool loadsTrainData = true;
	DataSet(bool load_train_data)
		:scenes{},
		loadsTrainData{ load_train_data } {
	}
	void initializeLoaders();

	std::vector<img_id> shuffled_for_random_load;
public:
	//todo? extract to another file/class?
	img_id initLoadImageOfRand() {
		for (auto& sc : scenes) {
			for (int ti = 0; ti < sc.num_train_images(); ++ti)
				shuffled_for_random_load.emplace_back(img_id(sc.index, ti));
		}
		std::random_shuffle(shuffled_for_random_load.begin(), shuffled_for_random_load.end());
		img_id next = shuffled_for_random_load[shuffled_for_random_load.size() - 1];
		shuffled_for_random_load.pop_back();
		return next;
	}
	std::pair<img_id, std::shared_ptr<TrainingImage>> loadImageOfRand(img_id id) {
		const int scene_id = id.first, image_id = id.second;
		img_id next;
		if (shuffled_for_random_load.size() == 0) {
			next = initLoadImageOfRand();
		}
		else
			next = shuffled_for_random_load[shuffled_for_random_load.size() - 1];
		shuffled_for_random_load.pop_back();
		return std::make_pair(img_id(next.first, next.second), scene(scene_id).loadOne(image_id));
	};

	std::pair<img_id, std::shared_ptr<TrainingImage>> loadImageOf(img_id id) {
		const int scene_id = id.first, image_id = id.second;
		int next_scene_id = id.first, next_image_id = id.second + 1;
		while (next_image_id >= scene(next_scene_id).num_train_images()) {
			next_image_id = 0; next_scene_id = (next_scene_id + 1) % num_scenes();
		}
		return std::make_pair(img_id(next_scene_id, next_image_id), scene(scene_id).loadOne(image_id));
	};

	DataSet(const std::vector<std::string>& paths, bool autoload, bool loadTrainData = true, bool quiet = true)
		:
		scenes{},
		loadsTrainData{ loadTrainData } {
		for (int i = 0; i < paths.size(); ++i) {
			scenes.emplace_back(paths[i], i, autoload, loadsTrainData, quiet);
			if (scenes[i].num_train_images() > 0)train_scenes += 1;
		}
		if(loadsTrainData)initializeLoaders();
	}
	size_t num_scenes() { return scenes.size(); }
	std::pair<Scene&, std::shared_ptr<TrainingImage>> next_example() {
		auto img = show_image_provider->pop();
#if _DEBUG
		if (img == nullptr)
			throw "IMAGE BUFFER GOT INVALID OUTPUT";
#endif
		return { scene(img->scene_id), img };
	}
	std::pair<Scene&, std::shared_ptr<TrainingImage>> next_train() {
		auto img = train_image_provider->pop();
#if _DEBUG
		if (img == nullptr)
			throw "IMAGE BUFFER GOT INVALID OUTPUT";
#endif
		return { scene(img->scene_id), img };
	}

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

	void expand_to_ndim(int ndim) {
		for (auto& scene : scenes)scene.expand_to_ndim(ndim);
	}

	int save(const std::string& path, fileType_t mode, bool saveTrainingImages);
	static std::unique_ptr<DataSet> load(const std::string& path, bool loadTrainingimagesIfPresent = true, bool quiet = true);

};
