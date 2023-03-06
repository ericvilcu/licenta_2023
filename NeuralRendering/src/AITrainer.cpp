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
#include "cli_args.hpp"

inline bool stringEndsWith(const std::string& src, const std::string& ot) {
    for (size_t i = 0; i < ot.size(); i++)
    {
        if (ot[i] != src[src.size() - ot.size() + i])
            return false;
    }
    return true;
}
inline bool stringStartsWith(const std::string& src, const std::string& ot) {
    //return (src.find(ot) == 0);
    for (size_t i = 0; i < ot.size(); i++)
    {
        if (ot[i] != src[i])
            return false;
    }
    return true;
}

class mainModuleImpl : public torch::nn::Module {
private:
    //Here because I want to do these differently eventually https://arxiv.org/pdf/2205.05509.pdf
    typedef torch::nn::Conv2dOptions convModuleOptions;
    typedef torch::nn::Conv2d convModule;
    typedef torch::nn::Conv2dImpl convModuleImpl;
    int num_layers;
    int ndim;
    torch::nn::AvgPool2d downsampler = nullptr;
    std::vector<std::vector<convModule>> convolutional_layers_in;
    std::vector<std::vector<convModule>> convolutional_layers_out;
    convModule final_convolutional_layer_out = nullptr;
    void rebuild_layers(bool delete_prev=true) {
        if(delete_prev)
            for (auto& m : named_modules("",false))
                unregister_module(m.key());
        convolutional_layers_in.clear();
        convolutional_layers_out.clear();
        int last_layer_out_channels = 0;
        const int in_layer_out_channels = 32;
        const int out_layer_out_channels = 32;
        downsampler = register_module("downsampler", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ 2,2 }).stride(2)));
        int std_padding = 1;
        torch::ExpandingArray<2> std_kernel_size{ std_padding * 2LL + 1,std_padding * 2LL + 1 };
        //todo? maybe reduce the number of channels back down? this may be overkill.
        for (int i = 0; i < num_layers; ++i) {
            convolutional_layers_in.emplace_back();
            auto options = convModuleOptions((ndim+1LL) + last_layer_out_channels, 32, std_kernel_size).padding(std_padding);
            convolutional_layers_in[i].emplace_back(register_module<convModuleImpl>(std::string("conv2din_") + std::to_string(i) + "_1", convModule(options)));
            options = convModuleOptions(32, 32, std_kernel_size).padding(std_padding);
            convolutional_layers_in[i].emplace_back(register_module<convModuleImpl>(std::string("conv2din_") + std::to_string(i) + "_2", convModule(options)));
            options = convModuleOptions(32, in_layer_out_channels, std_kernel_size).padding(std_padding);
            convolutional_layers_in[i].emplace_back(register_module<convModuleImpl>(std::string("conv2din_") + std::to_string(i) + "_3", convModule(options)));
            last_layer_out_channels = in_layer_out_channels;
        }
        last_layer_out_channels = 0;
        for (int i = 0; i < num_layers; ++i) {
            convolutional_layers_out.emplace_back();
            auto options = convModuleOptions((ndim + 1LL) + in_layer_out_channels + last_layer_out_channels, 32, std_kernel_size).padding(std_padding);
            convolutional_layers_out[i].emplace_back(register_module<convModuleImpl>(std::string("conv2dout_") + std::to_string(i) + "_1", convModule(options)));
            options = convModuleOptions(32, 32, std_kernel_size).padding(std_padding);
            convolutional_layers_out[i].emplace_back(register_module<convModuleImpl>(std::string("conv2dout_") + std::to_string(i) + "_2", convModule(options)));
            options = convModuleOptions(32, out_layer_out_channels, std_kernel_size).padding(std_padding);
            convolutional_layers_out[i].emplace_back(register_module<convModuleImpl>(std::string("conv2dout_") + std::to_string(i) + "_3", convModule(options)));
            last_layer_out_channels = out_layer_out_channels;
        }
        
        {
            auto options = convModuleOptions(last_layer_out_channels, 4, 1).padding(0);
            final_convolutional_layer_out = register_module<convModuleImpl>("final_layer", convModule(options));
        }
    }
public:

    mainModuleImpl(const std::string& path) {
        train(true);
        this->to(torch::kCUDA);
        this->load_from(path);
    }

    mainModuleImpl(int ndim = 3, int layer_count = 4, bool empty = false, bool set_train = true) {
        this->ndim = ndim;
        num_layers = layer_count;
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

        prev = final_convolutional_layer_out(prev);
        //Now untranspose it.
        return prev.transpose(-2,-1).transpose(-3,-1);
    }
    void load_from(const std::string& file) {
        torch::serialize::InputArchive model_archive;
        model_archive.load_from(file);
        torch::Tensor t_num_layers;
        torch::Tensor t_ndim;
        model_archive.read("num_layers", t_num_layers, false);
        model_archive.read("ndim", t_ndim, false);
        num_layers = t_num_layers.item().toInt();
        ndim = t_ndim.item().toInt();
        rebuild_layers();
        load(model_archive);
        to(torch::kCUDA);
    }
    void save_to(const std::string& file) {
        torch::serialize::OutputArchive model_archive;
        save(model_archive);
        torch::Tensor t_ndim = torch::tensor(ndim);
        torch::Tensor t_num_layers = torch::tensor(num_layers);
        model_archive.write("num_layers", t_num_layers, false);
        model_archive.write("ndim", t_ndim, false);
        model_archive.save_to(file);
    }
    int layer_count() {
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
    int MAX_NUM_PIXELS=INT_MAX;
    int batch_size;
    int remaining_in_batch;
    float accumulated_loss = 0;
    //torch::optim::LRScheduler
    torch::optim::Adam optim;
    Network() = delete;
    Network(const Network&) = delete;
    Network(Network&&) = delete;
    Network(NetworkPointer* parent)
        :parent{ parent },
        plotter{ parent->getDataSet() },
        batch_size{ 1 },
        remaining_in_batch{ 1 },
        mdl{ 1,1 },
        optim{ std::vector<torch::optim::OptimizerParamGroup>{}, torch::optim::AdamOptions() },
        status(0, 0) {
    }
    Network(NetworkPointer* parent, int ndim, int depth, int batch_size = 20)
        :parent{ parent },
        plotter{ parent->getDataSet() },
        batch_size{ batch_size },
        remaining_in_batch{ batch_size },
        mdl{ ndim,depth } ,
        optim{ std::vector<torch::optim::OptimizerParamGroup>{plotter->parameters(), mdl->parameters()}, torch::optim::AdamOptions() },
        status(0, 0){
    }


    Network(NetworkPointer* parent, const std::string& network_path, const std::string& optim_path)
        :parent{ parent },
        plotter{ parent->getDataSet() },
        batch_size{ 1 },
        remaining_in_batch{ 1 },
        mdl{ network_path },
        optim{ std::vector<torch::optim::OptimizerParamGroup>{plotter->parameters(), mdl->parameters()}, torch::optim::AdamOptions() },
        status(0, 0){
        torch::serialize::InputArchive optim_arc; optim_arc.load_from(optim_path); optim.load(optim_arc);
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
        return mdl->forward(plot_images_and_halves(mdl->layer_count(), scene.index, camera));
    }

    torch::Tensor plot_one(const Scene& scene, std::shared_ptr<CameraDataItf> camera) {
        return plot_images_and_halves(1, scene.index, camera)[0];
    }

    torch::Tensor train_diff(torch::Tensor produced, torch::Tensor target) {
        //todo: use unbind instead
        //almost equivalent to trust = target[::-1] (the last dim.)
        torch::Tensor trust = torch::stack({ target.transpose(0, -1)[-1] }).transpose(0,-1);//selects only the last dim.
        float mt = trust.sum().item().toFloat();
        trust = torch::cat({ trust,trust,trust,trust }, -1);
        produced = produced * trust;
        return torch::smooth_l1_loss(produced, target, torch::Reduction::Sum, 0.5) / batch_size / mt;//Normally, this should not happen, but since we're using the ADAM optimizer and want to change the batch size between trainings sometimes, it should be here.
    }

    //NOTE: cannot be used for training.
    torch::Tensor size_safe_forward(int scene_id, std::shared_ptr<CameraDataItf> camera) {
        torch::NoGradGuard guard;
        int w = camera->get_width(), h = camera->get_height();
        float total_loss = 0;
        int MW = w, MH = h;
        try {
            std::vector<std::shared_ptr<CameraDataItf>> cameras;
            std::vector<int4> subsections;
            std::vector<torch::Tensor> sub_plots;
            int scaleX=1,scaleY=1;
            int const_padding = 0;//tbh the normal images could use padding as well, but that sounds like something that should be done on another abstraction layer.
            if (w * h > MAX_NUM_PIXELS) {
                //split needed
                const_padding = 8;
                //todo? use another method? this is dumb
                scaleX = scaleY = ceil(sqrt(w * h / MAX_NUM_PIXELS));
                int w0 = ceil(1.0 * w / scaleX);
                int h0 = ceil(1.0 * h / scaleY);
                while ((w0 + const_padding * 2) * (h0 + const_padding * 2) > MAX_NUM_PIXELS) {
                    if (scaleX <= scaleY)scaleX++;
                    else scaleY++;
                    w0 = ceil(1.0 * w / scaleX);
                    h0 = ceil(1.0 * h / scaleY);
                }
                for (int i = 0; i < scaleX; ++i) {
                    for (int j = 0; j < scaleY; ++j) {
                        int4 subsection = make_int4(i * w / scaleX, j * h / scaleY, (1 + i) * w / scaleX, (1 + j) * h / scaleY);
                        subsections.push_back(subsection);
                        cameras.push_back(camera->subSection(subsection.x - const_padding, subsection.y - const_padding, subsection.z + const_padding, subsection.w + const_padding));
                    }
                }
            }
            else {
                //no split neccesarry;
                cameras.push_back(camera);
                subsections.push_back(make_int4(0, 0, camera->get_width(), camera->get_height()));
            }
            for (int i = 0; i < cameras.size(); ++i) {
                MW = cameras[i]->get_width();
                MH = cameras[i]->get_height();
                std::vector<torch::Tensor> plots = plot_images_and_halves(mdl->layer_count(), scene_id, cameras[i]);
                auto generated = mdl->forward(plots);

                if (const_padding > 0) {
                    generated = generated.slice(0, const_padding, -const_padding);
                    generated = generated.slice(1, const_padding, -const_padding);
                }
                if (generated.requires_grad()) {
                    generated.set_requires_grad(false);
                }
                sub_plots.push_back(generated);
            }
            torch::Tensor pathed_all;
            for (int i = 0; i < scaleX; ++i) {
                torch::Tensor pathed_column;
                for (int j = 0; j < scaleY; ++j) {
                    if (pathed_column.defined())pathed_column = torch::cat({ pathed_column,sub_plots[i * scaleY + j] }, 0);
                    else pathed_column = sub_plots[i * scaleY + j];
                    sub_plots[i * scaleY + j] = torch::Tensor();
                }
                if(pathed_all.defined())pathed_all = torch::cat({ pathed_all,pathed_column }, 1);
                else pathed_all = pathed_column;
            }
            return pathed_all;
        }
        catch (torch::OutOfMemoryError error) {
            MAX_NUM_PIXELS = MW * MH - 1;
            std::string err{ error.what_without_backtrace() };
            if (!global_args->quiet) {
                std::cerr << "Plotting an image of size " << MW * MH << "(" << MW << "x" << MH << ") caused error:\n" << error.what_without_backtrace() << '\n';
                std::cerr << "Assuming this is a memory error and decreasing max pixel count on an image... (will be applied next training cycle)\n";
            }
            return size_safe_forward(scene_id, camera);
        }
    }

    float train_image_with_splitting(int scene_id, std::shared_ptr<CameraDataItf> camera, torch::Tensor target) {
        int w = camera->get_width(), h = camera->get_height();
        float total_loss=0;
        int MW = w, MH = h;
        try {
            std::vector<std::shared_ptr<CameraDataItf>> cameras;
            std::vector<int4> subsections;
            int const_padding = 0;//tbh the normal images could use padding as well, but that sounds like something that should be done on another abstraction layer.
            if (w * h > MAX_NUM_PIXELS) {
                //split needed
                const_padding = 8;
                //todo? use another method? this is dumb
                int scale = ceil(sqrt(w * h / MAX_NUM_PIXELS));
                int w0 = ceil(1.0 * w / scale);
                int h0 = ceil(1.0 * h / scale);
                while ((w0 + const_padding * 2) * (h0 + const_padding * 2) > MAX_NUM_PIXELS) {
                    w0 = ceil(1.0 * w / scale);
                    h0 = ceil(1.0 * h / scale);
                    scale++;
                }
                for (int i = 0; i < scale; ++i) {
                    for (int j = 0; j < scale; ++j) {
                        int4 subsection=make_int4(i*w/scale,j*h/scale,(1+i)*w/scale,(1+j)*h/scale);
                        subsections.push_back(subsection);
                        cameras.push_back(camera->subSection(subsection.x - const_padding, subsection.y - const_padding, subsection.z + const_padding, subsection.w + const_padding));
                    }
                }
            }
            else {
                //no split neccesarry;
                cameras.push_back(camera);
                subsections.push_back(make_int4(0, 0, camera->get_width(), camera->get_height()));
            }
            for (int i = 0; i < cameras.size(); ++i) {
                MW = cameras[i]->get_width();
                MH = cameras[i]->get_height();
                std::vector<torch::Tensor> plots = plot_images_and_halves(mdl->layer_count(), scene_id, cameras[i]);
                auto generated = mdl->forward(plots);
                auto local_target = target;
                if (const_padding > 0) {
                    generated = generated.slice(0, const_padding, -const_padding);
                    generated = generated.slice(1, const_padding, -const_padding);
                }
                if (cameras.size() != 1) {
                    local_target = local_target.slice(0, subsections[i].y, subsections[i].w);
                    local_target = local_target.slice(1, subsections[i].x, subsections[i].z);
                }
                if (local_target.dtype() == torch::kFloat) {
                    if (generated.sizes() != local_target.sizes())
                        assert(false);
                    auto output = train_diff(generated, local_target);
                    //std::cerr << img.target.mean().item<float>() << ' ' << generated.mean().item<float>() << '\n';
                    total_loss += output.item<float>();
                    output.backward();
                }
                else if (local_target.dtype() == torch::kByte) {
                    auto output = torch::smooth_l1_loss(generated, local_target.to(torch::kFloat, false, true) / 255.0f);
                    //std::cerr << img.target.mean().item<float>() << ' ' << generated.mean().item<float>() << '\n';
                    total_loss += output.item<float>();
                    output.backward();
                }
                else assert(false && "unsupported datatype");
            }
        } catch (torch::OutOfMemoryError error) {
            MAX_NUM_PIXELS = MW * MH - 1;
            std::string err{ error.what_without_backtrace() };
            if (!global_args->quiet) {
                std::cerr << "Plotting an image of size " << MW * MH << "(" << MW << "x" << MH << ") caused error:\n" << error.what_without_backtrace() << '\n';
                std::cerr << "Assuming this is a memory error and decreasing max pixel count on an image... (will be applied next training cycle)\n";
            }
        }
        return total_loss;
    }

    void train1() {
        //todo: split into train and validation, somehow.
        if (remaining_in_batch == 0) {
            // Optimizer pass
            optim.step();
            //Here would go other gradient-based optimizations.
            status.epochs++;
            status.epoch_count += 1;
            optim.zero_grad();
            status.loss += accumulated_loss;
            remaining_in_batch = batch_size;
            accumulated_loss = 0;
            return;
        }
        auto pair = parent->getDataSet()->next_train();
        auto& img = *pair.second;
        auto& scene = pair.first;
        //auto r=PartialCameraDataTemplate<PinholeCameraData>::check_internal_consistency(*(PinholeCameraData*)img.cam().get());
        //std::cerr << r.first << ' ' << r.second << '\n';
        remaining_in_batch = std::max(0, remaining_in_batch - 1);
        accumulated_loss += train_image_with_splitting(scene.index, img.cam(), img.target);
    }

    /*void train_batch() {
        std::vector<torch::Tensor> plots;
        std::vector<std::vector<torch::Tensor>> individual_plots;
        std::vector<std::pair<Scene&, std::shared_ptr<TrainingImage>>> image_data_pairs;
        for(int i=0;i< remaining_in_batch;++i)
        {
            std::pair<Scene&,std::shared_ptr<TrainingImage>> pair = parent->getDataSet()->next_train();
            image_data_pairs.emplace_back(pair);
            auto& img = *pair.second;
            auto& scene = pair.first;
            individual_plots.emplace_back(plot_images_and_halves(mdl->layer_count(), scene.index, img.cam()));
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
    }*///did not work that welll anyway
};


NetworkPointer::NetworkPointer(std::shared_ptr<DataSet> dataSet, int ndim, int depth) :dataSet{ dataSet }, network{ new Network(this,ndim,depth) }{
}

NetworkPointer::NetworkPointer(std::shared_ptr<DataSet> dataSet, std::string network_path, std::string optim_path) : dataSet{ dataSet }, network{ new Network(this,network_path,optim_path) }{}

//todo: implement properly;
NetworkPointer::trainingStatus& NetworkPointer::getTrainingStatus() {
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
    network->status.epoch_count = 0;
    network->status.loss = 0;
    while (((end - start).count() * 1e-6) <= (double)ms)
    {
        processed_frames++;
        network->train1();
        end = std::chrono::high_resolution_clock::now();
    }
}

void NetworkPointer::train_long(unsigned long long ns, unsigned long long save_ns, const std::string& workspace_folder, bool quiet, unsigned long long report_ns) {
    auto start = std::chrono::high_resolution_clock::now();
    auto last_report = std::chrono::high_resolution_clock::now() - std::chrono::milliseconds(report_ns);
    auto last_save = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    int processed_frames = 0;
    while (std::chrono::nanoseconds(end - start).count() <= ns)
    {
        processed_frames++;
        //if (network->batch_size < 3) network->train_batch();//prevents crashing due to insufficient memory to hold all of those images
        //else 
            network->train1();
        end = std::chrono::high_resolution_clock::now();
        if (!quiet && std::chrono::nanoseconds(end - last_report).count() >= report_ns) {
            std::stringstream ss{ std::ios::app | std::ios::out };
            if (network->status.epoch_count != 0) {
                network->status.print_pretty(ss);
                global_args->log(ss.str());
            }

            last_report = end;
            const auto left = std::chrono::nanoseconds(ns) - (end - start);
            const auto h = std::chrono::duration_cast<std::chrono::hours       >(left);
            const auto m = std::chrono::duration_cast<std::chrono::minutes     >(left - h);
            const auto s = std::chrono::duration_cast<std::chrono::seconds     >(left - h - m);
            const auto ms= std::chrono::duration_cast<std::chrono::milliseconds>(left - h - m - s);
            ss << "Time left:" << h.count() << "h " << m.count() << "m " << s.count() << "s " << ms.count() << "ms" << '\n';

            std::cout << ss.str() << '\n';
            network->status.epoch_count = 0;
            network->status.loss = 0;
        }
        if (save_ns > -1 && std::chrono::nanoseconds(end - last_save).count() >= save_ns) {
            last_save = end;
            if (!quiet)std::cout << "(autosave) Saving to: \"" << workspace_folder << "...\n";
            this->save(workspace_folder, fileType::CUSTOM_BINARY, true, true);
            if (!quiet)std::cout << "(autosave) Saved!\n\n";
        }
    }
}

void NetworkPointer::plot_example(Renderer& r, Renderer::ViewType points, Renderer::ViewType target, Renderer::ViewType result, std::shared_ptr<InteractiveCameraData> cd)
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
        for (int i=0; i < (cd->show_mips?network->mdl->layer_count():1); ++i) {
            auto tmp_tensor = network->plot_one(scene, img.cam()->scaleTo(w/(1<<i), h/(1<<i))).slice(-1, 0, 4);
            tmp_tensor = (tmp_tensor * 255).clamp(0, 255).to(caffe2::kByte, false, false).to_dense().contiguous();
            
            void* tmp;
            tmp = tmp_tensor.data_ptr<unsigned char>();
            bytesToSubView(*(const void**)&tmp, 0, 0, tmp_tensor.size(0), tmp_tensor.size(1), r, points);
        }
    }

    //The output image.
    {
        torch::Tensor tmp_tensor;
        void* tmp;
        tmp_tensor = network->size_safe_forward(scene.index, img.cam());
        tmp_tensor = (tmp_tensor * 255).clamp(0, 255).to(caffe2::kByte, false, false).to_dense().contiguous();
        tmp = tmp_tensor.data_ptr<unsigned char>();
        bytesToView(*(const void**)&tmp, tmp_tensor.size(0), tmp_tensor.size(1), r, result);
    }
}

//Should have really just separate camera data and options..
cudaError_t NetworkPointer::plotResultToRenderer(Renderer& renderer, const Scene& scene, std::shared_ptr<InteractiveCameraData> camera, const Renderer::ViewType viewType)
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
cudaError_t NetworkPointer::plotToRenderer(Renderer& renderer, const Scene& scene, std::shared_ptr<InteractiveCameraData> camera, const Renderer::ViewType viewType)
{
    int w, h;
    cudaError_t cudaStatus = cudaSuccess;
    const auto& view = renderer.getView(viewType);
    w = view.width;
    h = view.height;

    torch::Tensor tmp_tensor;

    tmp_tensor = network->plot_one(scene, camera->scaleTo(w, h));

    int channels = tmp_tensor.size(-1);//Note: one is the depth channel. The weight channel is the one thing not included.
    //Now slice it to the proper dimensions...
    if (camera->debug_channels == 1) {
        int p = (channels + (camera->debug_channel_start) % channels) % channels;
        tmp_tensor = tmp_tensor.slice(-1, p, p + 1);
        //std::cerr << mn << ' ' << mx << ' ' << mean << ' ' << dt << '\n';
        float mn = std::min(tmp_tensor.min().item().toFloat(), -1e-12f);
        float mx = std::max(tmp_tensor.max().item().toFloat(), 1e-12f);
        float mean = tmp_tensor.mean().item().toFloat();
        float dt = std::abs((tmp_tensor - mean).abs().mean().item().toFloat());
        mn = std::max(mn, (mean - 3 * dt));
        mx = std::min(mx, (mean + 3 * dt));
        tmp_tensor = ((tmp_tensor - mn) / (mx - mn));
        tmp_tensor = torch::cat({ tmp_tensor,tmp_tensor,tmp_tensor }, -1);
    } else { // if (camera->debug_channels == 3)
        int pr = (channels + (camera->debug_channel_start    ) % channels) % channels;
        int pg = (channels + (camera->debug_channel_start + 1) % channels) % channels;
        int pb = (channels + (camera->debug_channel_start + 2) % channels) % channels;
        tmp_tensor = torch::cat({ tmp_tensor.slice(-1, pr, pr + 1) ,tmp_tensor.slice(-1, pg, pg + 1),tmp_tensor.slice(-1, pb, pb + 1) }, -1);
    }
    //TODO? based on some camera option, normalise/tonemap?

    //Set the alpha channel to 1
    tmp_tensor = torch::nn::functional::pad(tmp_tensor, torch::nn::functional::PadFuncOptions({ 0,1 }).value(1));
    //Modify it to the bytes to project to the renderer and make it contignuous
    tmp_tensor = (tmp_tensor * 255).clamp(0, 255).to(caffe2::kByte, false, false).to_dense().contiguous();

    void* tmp;
    tmp = tmp_tensor.data_ptr<unsigned char>();
    cudaStatus = bytesToView(*(const void**)&tmp, tmp_tensor.size(0), tmp_tensor.size(1), renderer, viewType);
    return cudaStatus;
}

torch::Tensor NetworkPointer::forward(int sceneId, std::shared_ptr<CameraDataItf> camera)
{
    return network->process(dataSet->scene(sceneId), camera);
}

torch::Tensor NetworkPointer::size_safe_forward(int sceneId, std::shared_ptr<CameraDataItf> camera)
{
    return network->size_safe_forward(sceneId, camera);
}

constexpr auto MODEL_POSTFIX = "/model";
constexpr auto OPTIM_POSTFIX = "/optim";
constexpr auto DATA_POSTFIX  = "/data" ;
std::unique_ptr<NetworkPointer> NetworkPointer::load(const std::string& file, bool loadDatasetIfPresent, bool loadTrainImagesIfPresent, bool quiet)
{
    std::unique_ptr<NetworkPointer> ptr = nullptr;
    //first, if asked to, the dataSet
    if (loadDatasetIfPresent) {
        std::shared_ptr<DataSet> dataSet = DataSet::load(file + DATA_POSTFIX, loadTrainImagesIfPresent, quiet);
        if (dataSet == nullptr)goto Error;
        ptr = std::unique_ptr<NetworkPointer>(new NetworkPointer(dataSet, file + MODEL_POSTFIX, file + OPTIM_POSTFIX));
    } else {//branch may be broken
        ptr = std::make_unique<NetworkPointer>();
        {
            ptr->network = std::make_shared<Network>(ptr.get(), 3, 4);
            ptr->network->mdl->load_from(file + MODEL_POSTFIX);
        }
        {
            torch::serialize::InputArchive optim_archive;
            optim_archive.load_from(file + OPTIM_POSTFIX);
            ptr->network->optim.load(optim_archive);
            ptr->network->optim.add_param_group(ptr->network->mdl->parameters());
            ptr->network->optim.add_param_group(ptr->network->plotter->parameters());
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

void NetworkPointer::setBatchSize(int new_size){
    this->network->optim.zero_grad();
    this->network->batch_size = new_size;
    this->network->remaining_in_batch = new_size;
}

void NetworkPointer::train_images(bool train)
{
    network->plotter->set_train(train);
}

NetworkPointer::~NetworkPointer(){}
