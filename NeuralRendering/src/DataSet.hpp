#pragma once
#include <vector>
#include <random>
#include "Points.hpp"
#include "TrainImage.hpp"
#include "BlockingQueue.hpp"
#if __NVCC__
static_assert(false, "DO NOT INCLUDE THIS FILE FROM .cu FILES! IT HEAVILY UTILIZES 'typedef' WHICH CURRENTLY BREAKS NVCC COMPILED FILES!")
#endif
class Scene {
private:
	int number_of_train_images;
	const std::string path;
	int version = -1;
public:
	std::shared_ptr<GPUPoints> points;
	void reload(bool quiet = true);
	Scene(const std::string path, bool autoload = true, bool load_train_data = true, bool quiet = true)
		:number_of_train_images{ -1 },
		path{ path },
		points{ nullptr } {
		if (autoload) {
			reload(quiet);
		}
	}
	int num_train_images() { return number_of_train_images; };
	std::shared_ptr<TrainingImage> loadOne(int idx);
	bool is_loaded() {
		return !(points == nullptr);
	}
	void ensure_loaded() {
		if (!is_loaded()) {
			reload();
		}
	}
	void save(const std::string& path, bool save_training_data=false);
	void unload() {
		points = nullptr;
	}
	bool isValid() {
		if (points->color_memory_start == NULL || points->num_entries <= 0)
			return false;
		return true;
	}
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

	DataSet(const std::vector<std::string>& paths, bool autoload, bool loadTrainData = true)
		//restricts to 1 scene for now.
		:scenes{ {paths[0], autoload, loadsTrainData} },
		loadsTrainData{ loadTrainData } {
		initializeLoaders();
		/*for (const auto& path : paths) {
			scenes.emplace_back(path, autoload, loadsTrainData, max_samples_each, max_points_each);
		}*/
	}
	int num_scenes() { return scenes.size(); }
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

		for (auto& scene : scenes) if(scene.is_loaded()){
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

	int save(const std::string& path, bool saveTrainingImages);
	static std::unique_ptr<DataSet> load(int vers, const std::string& path, bool loadTrainingimagesIfPresent = true, bool quiet = true);

};