#pragma once
#include <iostream>
#include <memory>
#include <string>
#include "Points.hpp"
#include "Renderer.hpp"
#include "DataSet.hpp"
//The reason I have this indirection is that I do not want to have to include tensor libraries from main and other source files. It should only serve as an api that can turn a shitty image into a prettier one.
class Network;
class NetworkPointer {
private:
    std::shared_ptr<DataSet> dataSet;//dataSet above Network, as it should be initialized beforehand.
    std::shared_ptr<Network> network;
    //creates an empty optimizer for proper loading.
    NetworkPointer(std::shared_ptr<DataSet> dataSet, std::string network_path, std::string optim_path);
public:
    NetworkPointer() {};
    struct trainingStatus {
        int epoch_count = 0;
        int epochs;
        float loss;
        trainingStatus(int ep, float ls) {
            epochs = ep;
            loss = ls;
        }
        void print_pretty(std::ostream& out) {
            out << "total epochs=" << epochs << '\n';
            out << "epochs for this report=" << epoch_count << '\n';
            out << "loss=" << loss / std::max(1,epoch_count) << '\n';
        }
    };
    NetworkPointer(std::shared_ptr<DataSet> dataSet, int ndim, int depth);
    trainingStatus& getTrainingStatus();

    void train_frame(unsigned long long ms);
                                                                //every minute seems about fair.
    void train_long(unsigned long long nano_time, unsigned long long nano_autosave_time, const std::string& workspace_folder, bool quiet = false, unsigned long long report_frequency_ms = 1e9 * 60);
    /**
     * @brief Gets a before/after/from image. If you change these every frame you will be unwell.
     * 
     * @param r renderer to place the data in.
     */
    void plot_example(Renderer& r, Renderer::ViewType points, Renderer::ViewType target, Renderer::ViewType result, std::shared_ptr<InteractiveCameraData> cd);
    /**
    * @brief Plots to renderer a scene from the perspective of a camera, using the neural renderer.
    */
    cudaError_t plotResultToRenderer(Renderer& renderer, const Scene& scene, const std::shared_ptr<InteractiveCameraData> camera, const Renderer::ViewType viewType);

    cudaError_t plotToRenderer(Renderer& renderer, const Scene& scene, const std::shared_ptr<InteractiveCameraData> camera, const Renderer::ViewType viewType);
    std::shared_ptr<DataSet> getDataSet() { return dataSet; }
    void setDataSet(std::shared_ptr<DataSet> dataSet_new) { dataSet = dataSet_new; }
    static std::unique_ptr<NetworkPointer> load(const std::string& file, bool loadDatasetIfPresent = true, bool loadTrainImagesIfPresent = false, bool quiet = true);
    int save(const std::string& file, fileType_t mode, bool saveDataset = true, bool saveTrainImages = false);
    //todo: some way to add more data
    void setBatchSize(int new_size);
    //todo: some way to combine scenes
    void train_images(bool train);
    //This may be needed as shared_ptr's destructor may not be visible from any other place.
    torch::Tensor forward(int sceneId, std::shared_ptr<CameraDataItf> camera);
    torch::Tensor size_safe_forward(int sceneId, std::shared_ptr<CameraDataItf> camera);
    ~NetworkPointer();
};


