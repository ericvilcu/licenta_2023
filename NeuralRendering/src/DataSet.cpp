#include <sys/stat.h>
#include "stream_binary_utils.hpp"
#include "DataSet.hpp"

void Scene::load_points(bool quiet)
{
    if (existsAndIsFile((path + "/points.bin"))) {
        point_data = DataModule(path, CUSTOM_BINARY);
        if (!quiet)std::cout << "points loaded for " << path << "!\n";
    }
    if (existsAndIsFile((path + "/points.txt"))) {
        point_data = DataModule(path, TEXT);
        if (!quiet)std::cout << "points loaded for " << path << "!\n";
    }
    if (existsAndIsFile((path + "/points"))) {
        point_data = DataModule(path, TORCH_ARCHIVE);
        if (!quiet)std::cout << "points loaded for " << path << "!\n";
    }
}

void Scene::determine_image_count(bool quiet)
{
    int i = 1; number_of_train_images = 0;
    while (true) {
        auto pathn = path + "/train_images/" + std::to_string(i++) + file_extension_from_type(file_type);
        if (!existsAndIsFile(pathn)) {
            break;
        }
        number_of_train_images++;
    }
}

std::shared_ptr<TrainingImage> Scene::loadOne(int idx, bool autoload)
{
#ifdef _DEBUG
    if (idx >= number_of_train_images) {
        std::cerr << "Image number " << idx << " requested, but images only go up to " << number_of_train_images - 1;
        throw "out of range";
    }
#endif
    return std::make_shared<TrainingImage>(path + "/train_images/" + std::to_string(idx + 1) + file_extension_from_type(file_type), autoload);
}

fileType_t Scene::detect_file_type()
{
    if (existsAndIsFile(path + "/train_images/" + std::to_string(1) + file_extension_from_type(TEXT)))return TEXT;
    if (existsAndIsFile(path + "/train_images/" + std::to_string(1) + file_extension_from_type(CUSTOM_BINARY)))return CUSTOM_BINARY;
    if (existsAndIsFile(path + "/train_images/" + std::to_string(1) + file_extension_from_type(TORCH_ARCHIVE)))return TORCH_ARCHIVE;
    return -1;
}

void Scene::save(const std::string& path, fileType_t mode, bool save_training_data) {
    makeDirIfNotPresent(path);
    /*metadata, currently there is none.
    {
        std::ofstream f{ path + "/meta.txt" };
        f << vers;
    }*/
    std::string postfix = file_extension_from_type(mode);
    //the points
    //the training data
    if (save_training_data) {
        makeDirIfNotPresent(path + "/train_images");
        if (this->path != path) {
            for (int i = 0; i < number_of_train_images; ++i) {
                loadOne(i, false)->copyTo(path + "/train_images/" + std::to_string(i + 1) + postfix);
            }
        }
    }
    point_data->save(path, mode);
}

int DataSet::save(const std::string& path, fileType_t mode, bool saveTrainingImages) {
    makeDirIfNotPresent(path);
    for (int i = 0; i < scenes.size();++i) {
        scene(i).save(path + "/" + std::to_string(i), mode, saveTrainingImages);
    }
    return 0;
}

void DataSet::initializeLoaders() {
    train_image_provider = std::make_unique<ImageQueue>(5, nullptr);
    show_image_provider = std::make_unique<ImageQueue>(5, nullptr);
    trainImageLoader = std::make_unique<ImageLoader>(img_id(0, 0), [this](img_id id) {return loadImageOfRand(id); }, *train_image_provider);
    showImageLoader = std::make_unique<ImageLoader>(img_id(0, 0), [this](img_id id) {return loadImageOf(id); }, *show_image_provider);
}
std::unique_ptr<DataSet> DataSet::load(const std::string& path, bool loadTrainingimagesIfPresent, bool quiet) {
    auto ptr = std::unique_ptr<DataSet>(new DataSet(loadTrainingimagesIfPresent));
    struct stat info;
    int scene = 0;
    while (true) {
        std::string scene_path = path + '/' + std::to_string(scene++);
        if (stat(scene_path.c_str(), &info))break;//if it is inaccessible
        if (!(bool)(info.st_mode & S_IFDIR))break;//or it is not a directory
        ptr->scenes.emplace_back(scene_path,/*index=*/scene-1,/*autoload=*/true, loadTrainingimagesIfPresent, quiet);
    }
    if (loadTrainingimagesIfPresent && ptr->isValid()) {
        ptr->initializeLoaders();
    }
    return ptr;
}

torch::Tensor DataModuleImpl::readBinFrom(std::string path)
{
    std::ifstream f{ path , std::ios::binary};
    int64_t ndim = readOneBinary<int64_t>(f);
    std::vector<int64_t> dimensions;
    //dimensions read;
    readBinaryIntoArray(dimensions, f, ndim);
    std::vector<float> data;
    int tmp; size_t std_size = 1;
    for (int i = 0; i < ndim; ++i)
        std_size *= dimensions[i];
    data.reserve(std_size);
    readBinaryIntoArray(data, f, -1);//read any number of floats
    float tmpf;
    if (data.size() != std_size) {
        if (data.size() % std_size != 0)throw "misaligned data";
        int factor = data.size() / std_size;
        dimensions.insert(dimensions.begin(), factor);
    }
    torch::Tensor tsr = torch::tensor(std::move(data), torch::TensorOptions()).reshape(torch::IntArrayRef(dimensions)).cuda();
    return tsr;
}

void DataModuleImpl::writeBinTo(std::string path, torch::Tensor what)
{
    std::ofstream f{ path , std::ios::binary };
    int64_t ndim = what.sizes().size();
    writeOneBinary(f, ndim);
    for (const int64_t& dim : what.sizes()) {
        writeOneBinary(f, dim);
    }
    torch::Tensor what_CPU = what.cpu();
    writeBinary(f, what_CPU.data_ptr<float>(), what.numel());
}

torch::Tensor DataModuleImpl::readTxtFrom(std::string path)
{
    std::ifstream f{ path };
    int ndim;
    f >> ndim;
    std::vector<int64_t> dimensions;
    std::vector<float> data;
    int tmp; size_t std_size = 1;
    for (int i = 0; i < ndim; ++i) {
        f >> tmp; dimensions.push_back(tmp);
        std_size *= tmp;
    }
    data.reserve(std_size);
    float tmpf;
    //dimensions read;
    while (f >> tmpf) data.push_back(tmpf);

    if (data.size() > std_size) {
        if (data.size() % std_size != 0)throw "misaligned data";
        int factor = data.size() % std_size;
        dimensions.insert(dimensions.begin(), factor);
    }
    torch::Tensor tsr = torch::tensor(std::move(data), torch::TensorOptions()).reshape(torch::IntArrayRef(dimensions)).cuda();
    return tsr;
}

void DataModuleImpl::writeTxtTo(std::string path, torch::Tensor what){
    std::ofstream f{ path };
    f << what.sizes().size() << '\n';
    for (const int64_t& dim : what.sizes()) {
        f << dim << ' ';
    }
    f << '\n';
    torch::Tensor what_CPU = what.cpu();
    float* float_ptr = what_CPU.data_ptr<float>();
    int64_t numel = what.numel();
    for (int64_t i = 0; i < numel; ++i) {
        //could do '\n's occasionally but this is only to test edge-case scenarios with hand-made files ayway.
        f << float_ptr[i] << ' ';
    }
}

