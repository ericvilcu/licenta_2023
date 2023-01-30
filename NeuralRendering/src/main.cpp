#include <chrono>
#include <iostream>
#include <thread>
#include "Renderer.hpp"
#include "PlotPoints.cuh"
#include "CameraController.hpp"
#include "AITrainer.hpp"

int main(int argc, char** argv)
{
    //todo: more options (disable GUI, etc.)
    //these would all be read from argv in a theoretical final version.
    std::cout << "All libraries loaded.\n";//let me just load 4GB of libraries
    bool quiet = false;
    const char* default_path = "C:/...";
    bool timeout = true; float timeout_s = 60*30;
    const char* default_path_AI = "C:/...";
    bool load_from_dataset_only = false;//otherwise it loads from the entire AI model.
    const char* default_output_path = default_path_AI;
    bool NO_LIVE_RENDER = true;//false
    bool save_results = true; bool save_train_images = true;
    bool train = true;
    float example_refresh_rate = (train?2:0.3);

    try {
        std::shared_ptr<DataSet> dataSet = nullptr;
        std::shared_ptr<NetworkPointer> network = nullptr;
        if (load_from_dataset_only) {
            if (!quiet)std::cout << "Started reading dataset...\n";
            dataSet = std::make_shared<DataSet>(/*paths = */std::vector<std::string>{ default_path }, /*autoload = */ true, /*loadTrainData = */ true);
            if (!dataSet->isValid()) { std::cerr << "DataSet Invalid!"; return (int)std::errc::invalid_argument; }
            if (!quiet)std::cout << "Finished reading dataset...\n";

            if (!quiet)std::cout << "Initializing network...\n";
            network = std::make_shared<NetworkPointer>(dataSet);
            if (!quiet)std::cout << "Network initialized...\n";
        }
        else {
            network = NetworkPointer::load(0x600, default_path_AI, true, true);
            dataSet = network->getDataSet();
        }

        NetworkPointer& nw = *network;
        if (NO_LIVE_RENDER) {
            nw.train_long((unsigned long long)(timeout_s*1e3));
        }
        else {
            if (train) {
                if (!quiet)std::cout << "First train (initializes some torch things)\n";
                nw.train_frame(0);
                if (!quiet)std::cout << "First train done\n";
            }
            CameraGLData cam_data{ 1,PI * 1 / 3,0.1f,1e9f };
            if (!quiet)std::cout << "Initializing renderer...\n";
            Renderer r{ "Main Window" };
            r.update();
            //std::this_thread::sleep_for(std::chrono::milliseconds(1));//allow idle time.
            r.ensureWH(1664, 512);
            auto old_frame = std::chrono::high_resolution_clock::now();
            //std::this_thread::sleep_for(std::chrono::milliseconds(1));
            CameraController controller{ r,cam_data };//       x0  x1  y0  y1  z flexible-resolution
            r.createView(Renderer::ViewTypeEnum::MAIN_VIEW   , .0, .5, .0, .5, 0, true );
            r.createView(Renderer::ViewTypeEnum::TRAIN_VIEW_1, .5, 1., .0, .5, 0, false);
            r.createView(Renderer::ViewTypeEnum::TRAIN_VIEW_2, .5, 1., .5, 1., 0, false);
            r.createView(Renderer::ViewTypeEnum::POINTS_VIEW , .0, .5, .5, 1., 0, false);
            if (!quiet)std::cout << "Renderer initialized...\n";
            auto last_example_update = std::chrono::high_resolution_clock::now().operator-=(std::chrono::nanoseconds((long long)5e9));
            auto example_refresh_s = example_refresh_rate;
            r.update();
            auto all_start = std::chrono::high_resolution_clock::now();
            while (!r.shouldClose()) {
                if (train) {
                    nw.train_frame(100);
                    if (!quiet)nw.getTrainingStatus().print_pretty(std::cout);
                }
                else {
                    //std::cout << cam_data.translation.x << ' ' << cam_data.translation.y << ' ' << cam_data.translation.z << '\n';
                    //std::cout << cam_data.transform << '\n';
                }
                auto current_frame = std::chrono::high_resolution_clock::now();
                controller.processMovements();
                if (cam_data.use_neural == false) {
                    plotPointsToRenderer(r, *dataSet->scene(0).points, cam_data, Renderer::ViewTypeEnum::MAIN_VIEW);
                }
                else {
                    void* memory;
                    nw.plotToRenderer(r, *dataSet->scene(0).points, cam_data, Renderer::ViewTypeEnum::MAIN_VIEW);
                }

                if ((current_frame - last_example_update).count() * 1e-9 > example_refresh_s) {
                    nw.plot_example(r, Renderer::ViewTypeEnum::POINTS_VIEW, Renderer::ViewTypeEnum::TRAIN_VIEW_2, Renderer::ViewTypeEnum::TRAIN_VIEW_1);

                    last_example_update = current_frame;
                }
                
                r.update();
                auto delta = current_frame - old_frame;
                //std::cout << "FPS:" << delta.count() / 1e12 << '\n';
                old_frame = std::chrono::high_resolution_clock::now();
                //todo: some kind of system for calculating the average framerate.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));//allow idle time.

                if (timeout) {
                    auto curr = std::chrono::high_resolution_clock::now();
                    if ((curr - all_start).count() * 1e-9 > timeout_s)
                        break;
                }
            }
        }
        if (save_results)
            nw.save(default_output_path, true, save_train_images);
        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        auto cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }
    }
    catch (std::exception x) {
        std::cerr << "Uncaught error:\n" << x.what();
        return -1;
    }
    catch (std::string dat) {
        std::cerr << "Uncaught error:" << dat;
        return -1;
    }
    catch (const char* dat) {
        std::cerr << "Uncaught error:" << dat;
        return -1;
    }
    return 0;
}