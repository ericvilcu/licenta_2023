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
    std::shared_ptr<Network> network;
    std::shared_ptr<DataSet> dataSet;
    NetworkPointer() {};
public:
    struct trainingStatus {
        int epoch_count = 1;
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
    NetworkPointer(std::shared_ptr<DataSet> dataSet);
    trainingStatus getTrainingStatus();

    void train_frame(unsigned long long ms);
                                                                //every minute seems about fair.
    void train_long(unsigned long long ms, unsigned long long report_frequency_ms=1000 * 60);
    /**
     * @brief Gets a before/after/from image. If you change these every frame you will be unwell.
     * 
     * @param r renderer to place the data in.
     */
    void plot_example(Renderer& r, Renderer::ViewType points, Renderer::ViewType target, Renderer::ViewType result);
    /**
    * @brief Plots to renderer a scene from the perspective of a camera, using the neural renderer.
    */
    cudaError_t plotToRenderer(Renderer& renderer, const GPUPoints& points, const CameraDataItf& camera, const Renderer::ViewType viewType);
    std::shared_ptr<DataSet> getDataSet() { return dataSet; }
    void setDataSet(std::shared_ptr<DataSet> dataSet_new) { dataSet = dataSet_new; }
    static std::unique_ptr<NetworkPointer> load(int vers, const std::string& file, bool loadDatasetIfPresent = true, bool loadTrainImagesIfPresent = false);
    int save(const std::string& file, bool saveDataset = true, bool saveTrainImages = false);
    //todo: some way to add more data
    //todo: some way to combine scenes
    //This may be needed as shared_ptr's destructor may not be visible from any other place.
    ~NetworkPointer();
};


