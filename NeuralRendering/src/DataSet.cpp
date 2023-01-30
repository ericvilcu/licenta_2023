#include "DataSet.hpp"
#include <sys/stat.h>
#include "stream_binary_utils.hpp"
void Scene::reload(bool quiet)
{
    int vers = 0x500;//v5 does not have a .meta file.
    {
        std::ifstream f{ path + "/meta.txt" };
        if (!f.fail()) {
            f >> vers;
        }
    }
    version = vers;
    number_of_train_images = 0;
    points = std::make_shared<GPUPoints>();
    GPUPoints::read((path + "/points.txt").c_str(), *points, vers);
    if (!quiet)std::cout << "points loaded for " << path << "!\n";
    if (vers == 0x500) {
        std::ifstream f{ path + "/paths.txt" };
        int expected_count;
        if (!(f >> expected_count))return;
        if (expected_count > 0) {
        }
        int i = 1;
        float rotation[9];
        float translation[3];
        while (f >> rotation[0] >> rotation[1] >> rotation[2]
            >> rotation[3] >> rotation[4] >> rotation[5]
            >> rotation[6] >> rotation[7] >> rotation[8]
            >> translation[0] >> translation[1] >> translation[2]) {
            float4x4 rt{
                -rotation[0], -rotation[3], -rotation[6],translation[0],
                rotation[1], rotation[4], rotation[7],-translation[1],
                rotation[2], rotation[5], rotation[8],-translation[2],
                0,0,0,1,
            };
            number_of_train_images++;
        }
    }
    else { //0x600
        int i = 1;
        while (true) {
            auto pathn = path + "/train_images/" + std::to_string(i++) + ".txt";
            if ( !std::ifstream(pathn).good() ) {
                break;
            }
            number_of_train_images++;
        }
    }
}

std::shared_ptr<TrainingImage> Scene::loadOne(int idx)
{
#ifdef _DEBUG
    if (idx >= number_of_train_images) {
        std::cerr << "Image number " << idx << " requested, but images only go up to " << number_of_train_images - 1;
        throw "out of range";
    }
#endif
    return std::make_shared<TrainingImage>(path + "/train_images/" + std::to_string(idx+1) + ".txt",version);
}

void Scene::save(const std::string& path,bool save_training_data) {
    int vers = this->version;
    makeDirIfNotPresent(path);
    //metadata
    {
        std::ofstream f{ path + "/meta.txt" };
        f << vers;
    }
    //the points
    points->writeToFile((path + "/points.txt").c_str(), vers);
    //the training data
    if (save_training_data) {
        makeDirIfNotPresent(path + "/train_images");
        if(this->path!=path)
            for (int i = 1; i <= number_of_train_images; ++i) {
                std::ifstream  src(this->path + "/train_images/" + std::to_string(i) + ".txt", std::ios::binary);
                std::ofstream  dst(path + "/train_images/" + std::to_string(i) + ".txt", std::ios::binary);
                dst << src.rdbuf();
            }
    }
}

int DataSet::save(const std::string& path, bool saveTrainingImages) {
    makeDirIfNotPresent(path);
    for (int i = 0; i < scenes.size();++i) {
        scene(i).save(path + "/" + std::to_string(i), saveTrainingImages);
    }
    return 0;
}

void DataSet::initializeLoaders() {
    train_image_provider = std::make_unique<ImageQueue>(5, nullptr);
    show_image_provider = std::make_unique<ImageQueue>(5, nullptr);
    trainImageLoader = std::make_unique<ImageLoader>(img_id(0, 0), [this](img_id id) {return loadImageOfRand(id); }, *train_image_provider);
    showImageLoader = std::make_unique<ImageLoader>(img_id(0, 0), [this](img_id id) {return loadImageOf(id); }, *show_image_provider);
}
std::unique_ptr<DataSet> DataSet::load(int vers, const std::string& path, bool loadTrainingimagesIfPresent, bool quiet) {
    auto ptr = std::unique_ptr<DataSet>(new DataSet(loadTrainingimagesIfPresent));
    struct stat info;
    int scene = 0;
    while (true) {
        std::string scene_path = path + '/' + std::to_string(scene++);
        if (stat(scene_path.c_str(), &info))break;//if it is inaccessible
        if (!(bool)(info.st_mode & S_IFDIR))break;//or it is not a directory
        ptr->scenes.emplace_back(scene_path, true, loadTrainingimagesIfPresent, quiet);
    }
    if (loadTrainingimagesIfPresent && ptr->isValid()) {
        ptr->initializeLoaders();
    }
    return ptr;
}