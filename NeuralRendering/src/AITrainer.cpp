#include "AITrainer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <fstream>
#include <chrono>
#include <memory.h>
#include "CameraData.hpp"
#include "PlotPointsBackwardsPasses.cuh"
#include "PlotPoints.cuh"
#include "PlotterModule.hpp"
#include "stream_binary_utils.hpp"
//Not the best solution, but the best I could think of.
//#include "HeaderThatSupressesWarnings.h"
//#include <torch/torch.h>
//#include "HeaderThatReenablesWarnings.h"

inline bool stringEndsWith(const std::string& src, const std::string& ot) {
    for (size_t i = 0; i < ot.size(); i++)
    {
        if (ot[i] != src[src.size() - ot.size() + i])
            return false;
    }
    return true;
}


//-todo: see if torchScript would be a viable alternative for portablility.
class mainModuleImpl : public torch::nn::Module {
private:
    int num_layers;
    torch::nn::AvgPool2d downsampler = nullptr;
    std::vector<std::vector<torch::nn::Conv2d>> convolutional_layers_in;
    std::vector<std::vector<torch::nn::Conv2d>> convolutional_layers_out;
    void rebuild_layers(bool delete_prev=true) {
        if(delete_prev)
            for (auto& m : named_modules("",false))
                unregister_module(m.key());
        convolutional_layers_in.clear();
        convolutional_layers_out.clear();
        int last_layer_out_channels = 0;
        downsampler = register_module("downsampler", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ 2,2 }).stride(2)));
        int std_padding = 1;
        torch::IntArrayRef std_kernel_size{ std_padding * 2LL + 1,std_padding * 2LL + 1 };
        for (int i = 0; i < num_layers; ++i) {
            convolutional_layers_in.emplace_back();
            auto options = torch::nn::Conv2dOptions(4LL + last_layer_out_channels, 32, std_kernel_size).padding(std_padding);
            convolutional_layers_in[i].emplace_back(register_module<torch::nn::Conv2dImpl>(std::string("conv2din_") + std::to_string(i) + "_1", torch::nn::Conv2d(options)));
            options = torch::nn::Conv2dOptions(32, 16, std_kernel_size).padding(std_padding);
            convolutional_layers_in[i].emplace_back(register_module<torch::nn::Conv2dImpl>(std::string("conv2din_") + std::to_string(i) + "_2", torch::nn::Conv2d(options)));
            options = torch::nn::Conv2dOptions(16, 8, std_kernel_size).padding(std_padding);
            convolutional_layers_in[i].emplace_back(register_module<torch::nn::Conv2dImpl>(std::string("conv2din_") + std::to_string(i) + "_3", torch::nn::Conv2d(options)));
            last_layer_out_channels = 8;
        }
        last_layer_out_channels = 0;
        for (int i = 0; i < num_layers; ++i) {
            convolutional_layers_out.emplace_back();
            auto options = torch::nn::Conv2dOptions(12LL + last_layer_out_channels, 32, std_kernel_size).padding(std_padding);
            convolutional_layers_out[i].emplace_back(register_module<torch::nn::Conv2dImpl>(std::string("conv2dout_") + std::to_string(i) + "_1", torch::nn::Conv2d(options)));
            options = torch::nn::Conv2dOptions(32, 16, std_kernel_size).padding(std_padding);
            convolutional_layers_out[i].emplace_back(register_module<torch::nn::Conv2dImpl>(std::string("conv2dout_") + std::to_string(i) + "_2", torch::nn::Conv2d(options)));
            options = torch::nn::Conv2dOptions(16, 4, std_kernel_size).padding(std_padding);
            convolutional_layers_out[i].emplace_back(register_module<torch::nn::Conv2dImpl>(std::string("conv2dout_") + std::to_string(i) + "_3", torch::nn::Conv2d(options)));
            last_layer_out_channels = 4;
        }
    }
public:

    mainModuleImpl(int layers = 4, bool empty = false, bool set_train = true) {
        num_layers = layers;
        train(set_train);
        if (!empty) {
            rebuild_layers(false);
        }
        this->to(torch::kCUDA);
    }
    //The fun part is backwards does not need to be implemented.
    //Note: const does not mean the tensors themselves are unchangeable, tensors are basically just memory references in libtorch
    torch::Tensor forward(const std::vector<torch::Tensor>& imgs) {
        std::vector<torch::Tensor> partials;
        torch::Tensor prev = torch::zeros({ 0 });
        
        for (int idx = 0; idx < convolutional_layers_in.size(); ++idx) {
            //needs to be tranpsosed so that the functions work properly, as channels are expected to be the first dimenstion after batch in libtorch.
            torch::Tensor img = imgs[idx].transpose(-3, -1).transpose(-2, -1);
            if (prev.size(0) != 0) {
                img = torch::cat({ img,prev }, -3);
            }
            for (auto& conv : convolutional_layers_in[idx]) {
                img = torch::elu(conv(img));
            }
            partials.push_back(img);
            prev = downsampler(img);
        }
        prev = torch::zeros({ 0 });
        for (int idx = 0; idx < convolutional_layers_out.size(); ++idx) {
            torch::Tensor img = partials.back();
            partials.pop_back();
            if (prev.size(0) != 0) {
                if (prev.sizes().size() == 3) {
                    prev = torch::nn::functional::interpolate(torch::stack(prev),
                        torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).align_corners(false).size(std::vector<int64_t>{ img.size(-2), img.size(-1) }))[0];
                }
                else {
                    prev = torch::nn::functional::interpolate(prev,
                        torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).align_corners(false).size(std::vector<int64_t>{ img.size(-2), img.size(-1) }));
                }
                img = torch::cat({ img,prev }, -3);
            }
            img = torch::cat({ img,imgs[partials.size()].transpose(-3, -1).transpose(-2, -1) }, -3);
            for (auto& conv : convolutional_layers_out[idx]) {
                img = torch::elu(conv(img));
            }
            prev = img;
        }

        //Now untranspose it.
        return prev.transpose(-2,-1).transpose(-3,-1);
    }
    void load_from(const std::string& file) {
        torch::serialize::InputArchive model_archive;
        model_archive.load_from(file);
        torch::Tensor t_num_layers;
        model_archive.read("num_layers", t_num_layers, false);
        num_layers = t_num_layers.item().toInt();
        rebuild_layers();
        load(model_archive);
        to(torch::kCUDA);
    }
    void save_to(const std::string& file) {
        torch::serialize::OutputArchive model_archive;
        save(model_archive);
        torch::Tensor t_num_layers = torch::tensor(num_layers);
        model_archive.write("num_layers", t_num_layers, false);
        model_archive.save_to(file);
    }
    int layers() {
        return num_layers;
    }
};

//Note to self: typedef breaks CUDA kernels.
//typedef torch::nn::ModuleHolder<mainModuleImpl> mainModule;
TORCH_MODULE(mainModule);
//This is the second level of indirection to the module class, which may no longer be needed as I now include torch basically everywhere.
class Network {
public:
    Plotter plotter = nullptr;
    NetworkPointer* parent;
    NetworkPointer::trainingStatus status;
    mainModule mdl;
    int batch_size;
    int remaining_in_batch;
    float accumulated_loss = 0;
    //torch::optim::LRScheduler
    torch::optim::Adam optim;
    Network(NetworkPointer* parent, int batch_size = 20)
        :parent{ parent },
        plotter{ parent->getDataSet() },
        batch_size{ batch_size },
        remaining_in_batch{ batch_size },
        mdl{},
        optim{ {mdl->parameters(),plotter->parameters()}, torch::optim::AdamOptions()/*.lr(0.00001)*/.eps(1e-8) },
        status(0, 0){
        
    }

    /// <summary>
    /// Plots an image from a set of points and a camera. Also plots smaller images (sizes 1/2,1/4,1//8,... if num>1)
    /// </summary>
    /// <param name="num">The number of "mips" to plot</param>
    /// <param name="points">The points to use.</param>
    /// <param name="camera">The camera to use.</param>
    /// <param name="w">Width of camera, -1 if not needed</param>
    /// <param name="h">Height of camera, -1 if not needed</param>
    /// <returns>A vector of images</returns>
    std::vector<torch::Tensor> plot_images_and_halves(int num, int scene_index, std::shared_ptr<CameraDataItf> camera, bool require_grad = true) {

        std::vector<torch::Tensor> imgs;
        int w = camera->get_width(), h = camera->get_height();
        std::shared_ptr<CameraDataItf> used_camera = camera->scaleTo(w, h);
        for (int i = 0; i < num; ++i) {
            torch::Tensor this_dim = plotter->forward(torch::tensor(std::vector<int>{ scene_index }), used_camera);
            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << err << ' ' << cudaGetErrorString(err) << '\n';
                throw cudaGetErrorString(err);
            }
            imgs.push_back(this_dim);
            w /= 2;
            h /= 2;
            used_camera = used_camera->scaleTo(w, h);
        }
        return std::move(imgs);
    }

    torch::Tensor process(const Scene& scene, std::shared_ptr<CameraDataItf> camera) {
        return mdl->forward(plot_images_and_halves(mdl->layers(), scene.index, camera));
    }

    torch::Tensor plot_one(const Scene& scene, std::shared_ptr<CameraDataItf> camera) {
        return plot_images_and_halves(1, scene.index, camera)[0];
    }

    torch::Tensor train_diff(torch::Tensor produced, torch::Tensor target) {
        //todo: use unbind instead
        //almost equivalent to trust = target[::-1] (the last dim.)
        torch::Tensor trust = torch::stack({ target.transpose(0, -1)[-1] }).transpose(0,-1);//selects only the last dim.
        trust = torch::cat({ trust,trust,trust,trust }, -1);
        produced = produced * trust;
        return torch::smooth_l1_loss(produced, target);// / batch_size;
    }

    void train1() {
        //todo: split into train and validation, somehow.
        if (remaining_in_batch == 0) {
            // Optimizer pass
            optim.step();
            //Here would go other gradient-based optimizations.
            status.epochs++;
            status.epoch_count = 1;
            optim.zero_grad();
            status.loss = accumulated_loss / batch_size;
            remaining_in_batch = batch_size;
            accumulated_loss = 0;
            return;
        }
        auto pair = parent->getDataSet()->next_train();
        auto& img = *pair.second;
        auto& scene = pair.first;
        remaining_in_batch = std::max(0, remaining_in_batch - 1);
        std::vector<torch::Tensor> plots = plot_images_and_halves(mdl->layers(), scene.index, img.cam());
        auto generated = mdl->forward(plots);
        if (img.target.dtype() == torch::kFloat) {
            if (generated.sizes() != img.target.sizes())
                assert(false);
            auto output = train_diff(generated, img.target);
            //std::cerr << img.target.mean().item<float>() << ' ' << generated.mean().item<float>() << '\n';
            accumulated_loss += output.item<float>();
            output.backward();
        }
        else if (img.target.dtype() == torch::kByte) {
            auto output = torch::smooth_l1_loss(generated, img.target.to(torch::kFloat, false, true) / 255.0f);
            //std::cerr << img.target.mean().item<float>() << ' ' << generated.mean().item<float>() << '\n';
            accumulated_loss += output.item<float>();
            output.backward();
        }
        else assert(false&&"unsupported datatype");
    }

    void train_batch() {
        std::vector<torch::Tensor> plots;
        std::vector<std::vector<torch::Tensor>> individual_plots;
        std::vector<std::pair<Scene&, std::shared_ptr<TrainingImage>>> image_data_pairs;
        for(int i=0;i< remaining_in_batch;++i)
        {
            std::pair<Scene&,std::shared_ptr<TrainingImage>> pair = parent->getDataSet()->next_train();
            image_data_pairs.emplace_back(pair);
            auto& img = *pair.second;
            auto& scene = pair.first;
            individual_plots.emplace_back(plot_images_and_halves(mdl->layers(), scene.index, img.cam()));
        }
        remaining_in_batch = 0;
        for (int i = 0; i < individual_plots[0].size(); ++i) {
            std::vector<torch::Tensor> tensorList;
            for (int j = 0; j < individual_plots.size(); ++j)tensorList.push_back(individual_plots[j][i]);
            plots.emplace_back(torch::stack(tensorList));
        }
        auto generated_multiple = mdl->forward(plots);
        std::vector<torch::Tensor> targets;
        //calculate losses.
        for (int i = 0; i < individual_plots.size(); ++i) {
            auto generated = generated_multiple[i];
            auto& pair = image_data_pairs[i];
            auto& img = *pair.second;
            auto& scene = pair.first;
            if (generated.sizes() != img.target.sizes())
                assert(false);
            if (img.target.dtype() == torch::kFloat) {
                targets.push_back(img.target);
            }
            else if (img.target.dtype() == torch::kByte) {
                targets.push_back(img.target.to(torch::kFloat, false, true) / 255.0f);
            }
            else assert(false && "unsupported datatype");
        }
        auto output = train_diff(generated_multiple, torch::stack(targets));
        output.backward();
        accumulated_loss += output.item<float>();
        remaining_in_batch = batch_size;
        {//if (remaining_in_batch == 0) {
            // Optimizer pass
            optim.step();
            //Here would go other gradient-based optimizations.
            status.epochs++;
            status.epoch_count++;
            optim.zero_grad();
            status.loss += accumulated_loss;
            accumulated_loss = 0;
            return;
        }
    }
};


NetworkPointer::NetworkPointer(std::shared_ptr<DataSet> dataSet) :dataSet{ dataSet }, network{ new Network(this) }{

}

//todo: implement properly;
NetworkPointer::trainingStatus NetworkPointer::getTrainingStatus() {
    return network->status;
}
//todo: implement properly;
void NetworkPointer::train_frame(unsigned long long ms) {
    if (ms > 5000) {
        //more than 5 seconds is a lot if we want to see frames in the meantime as well.
        ms = 5000;
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;
    int processed_frames = 0;
    while (((end - start).count() * 1e-6) <= (double)ms)
    {
        processed_frames++;
        network->train1();
        end = std::chrono::high_resolution_clock::now();
    }
}

void NetworkPointer::train_long(unsigned long long ms, unsigned long long report_frequency_ms){
    auto start = std::chrono::high_resolution_clock::now();
    auto last_report = std::chrono::high_resolution_clock::now() - std::chrono::milliseconds(report_frequency_ms);
    auto end = std::chrono::high_resolution_clock::now();
    int processed_frames = 0;
    while (((end - start).count() * 1e-6) <= (double)ms)
    {
        processed_frames++;
        if (network->batch_size < 5) network->train_batch();
        else network->train1();
        end = std::chrono::high_resolution_clock::now();
        if ((end - last_report).count() * 1e-6 >= (double)report_frequency_ms) {
            network->status.print_pretty(std::cout);
            last_report = end;
            const auto left = std::chrono::milliseconds(ms) - (end - start);
            const auto h = std::chrono::duration_cast<std::chrono::hours>(left);
            const auto m = std::chrono::duration_cast<std::chrono::minutes>(left - h);
            const auto s = std::chrono::duration_cast<std::chrono::seconds>(left - h - m);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(left - h - m - s);
            std::cout << "Time left:" << h.count() << "h " << m.count() << "m " << s.count() << "s " << ms.count() << "ms" << '\n';
            network->status.epoch_count = 0;
            network->status.loss = 0;
        }
    }
}

void NetworkPointer::plot_example(Renderer& r, Renderer::ViewType points, Renderer::ViewType target, Renderer::ViewType result)
{
    auto dat = dataSet->next_example();
    auto& img = *dat.second;
    auto& scene = dat.first;
    int w = img.width;
    int h = img.height;
    //Ensure all the views have enough space to place things in them.
    r.uploadRGBA8(points, NULL, w, h);
    r.uploadRGBA8(target, NULL, w, h);
    r.uploadRGBA8(result, NULL, w, h);
    //First, the simplest: the target image.
    {
        torch::Tensor tmp_tensor;
        void* tmp;
        tmp_tensor = (img.target * 255).clamp(0,255).to(caffe2::kByte, false, true).to_dense().contiguous();
        tmp = tmp_tensor.data_ptr<unsigned char>();
        //Todo: what is up with these ridiculous casts.
        bytesToView(*(const void**)&tmp, h, w, r, target);
    }

    //The largest point view, the AI may use some smaller ones as well.
    {
        //todo? add more stuff in INTERACTIVE camera to view different channels.
        auto tmp_tensor = network->plot_one(scene, img.cam()->scaleTo(w, h)).slice(-1, 0, 4);
        tmp_tensor = (tmp_tensor * 255).clamp(0, 255).to(caffe2::kByte, false, false).to_dense().contiguous();

        void* tmp;
        tmp = tmp_tensor.data_ptr<unsigned char>();
        bytesToView(*(const void**)&tmp, tmp_tensor.size(0), tmp_tensor.size(1), r, points);
    }

    //The output image.
    {
        torch::Tensor tmp_tensor;
        void* tmp;
        tmp_tensor = network->process(scene, img.cam());
        tmp_tensor = (tmp_tensor * 255).clamp(0, 255).to(caffe2::kByte, false, false).to_dense().contiguous();
        tmp = tmp_tensor.data_ptr<unsigned char>();
        bytesToView(*(const void**)&tmp, tmp_tensor.size(0), tmp_tensor.size(1), r, result);
    }
}

cudaError_t NetworkPointer::plotResultToRenderer(Renderer& renderer, const Scene& scene, std::shared_ptr<CameraDataItf> camera, const Renderer::ViewType viewType)
{
    int w, h;
    cudaError_t cudaStatus = cudaSuccess;
    const auto& view = renderer.getView(viewType);
    w = view.width;
    h = view.height;

    torch::Tensor tmp_tensor;
    void* tmp;
    tmp_tensor = network->process(scene, camera->scaleTo(w, h));
    tmp_tensor = (tmp_tensor * 255).clamp(0, 255).to(caffe2::kByte, false, false).to_dense().contiguous();

    tmp = tmp_tensor.data_ptr<unsigned char>();
    cudaStatus = bytesToView(*(const void**)&tmp, tmp_tensor.size(0), tmp_tensor.size(1), renderer, viewType);

    return cudaStatus;
}

//todo: include more options.
cudaError_t NetworkPointer::plotToRenderer(Renderer& renderer, const Scene& scene, std::shared_ptr<CameraDataItf> camera, const Renderer::ViewType viewType)
{
    //todo: add more stuff in INTERACTIVE camera to view different channels.
    int w, h;
    cudaError_t cudaStatus = cudaSuccess;
    const auto& view = renderer.getView(viewType);
    w = view.width;
    h = view.height;

    torch::Tensor tmp_tensor;

    //NOTE: removes all but the first 4 dimensions.
    tmp_tensor = network->plot_one(scene, camera->scaleTo(w, h)).slice(-1, 0, 4);
    tmp_tensor = (tmp_tensor * 255).clamp(0, 255).to(caffe2::kByte, false, false).to_dense().contiguous();

    void* tmp;
    tmp = tmp_tensor.data_ptr<unsigned char>();
    cudaStatus = bytesToView(*(const void**)&tmp, tmp_tensor.size(0), tmp_tensor.size(1), renderer, viewType);

    return cudaStatus;
}

#define MODEL_POSTFIX "/model"
#define OPTIM_POSTFIX "/optim"
#define DATA_POSTFIX "/data"
std::unique_ptr<NetworkPointer> NetworkPointer::load(const std::string& file, bool loadDatasetIfPresent, bool loadTrainImagesIfPresent, bool quiet)
{
    std::unique_ptr<NetworkPointer> ptr = nullptr;
    //first, if asked to, the dataSet
    if (loadDatasetIfPresent) {
        std::shared_ptr<DataSet> dataSet = DataSet::load(file + DATA_POSTFIX, loadTrainImagesIfPresent, quiet);
        if (dataSet == nullptr)goto Error;
        ptr = std::make_unique<NetworkPointer>(dataSet);

    } else {
        ptr = std::make_unique<NetworkPointer>();
    }
    //then, the model itself
    {
        {
            ptr->network = std::make_shared<Network>(ptr.get());
            ptr->network->mdl->load_from(file + MODEL_POSTFIX);
        }
        {
            torch::serialize::InputArchive optim_archive;
            optim_archive.load_from(file + OPTIM_POSTFIX);
            ptr->network->optim.load(optim_archive);
        }
    }
    return ptr;
Error:
    return nullptr;
}
int NetworkPointer::save(const std::string& file, fileType_t mode ,bool saveDataset, bool saveTrainImages)
{
    makeDirIfNotPresent(file);
    //First, the network itelf.
    {
        {
            this->network->mdl->save_to(file + MODEL_POSTFIX);
        }
        {
            torch::serialize::OutputArchive optim_archive;
            this->network->optim.save(optim_archive);
            optim_archive.save_to(file + OPTIM_POSTFIX);
        }
    }
    //then, if asked to, the dataSet
    if (saveDataset) {
        return dataSet->save(file + DATA_POSTFIX, mode ,saveTrainImages);
    }
    return 0;
}

NetworkPointer::~NetworkPointer(){}
