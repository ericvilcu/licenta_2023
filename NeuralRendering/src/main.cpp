#include <chrono>
#include <iostream>
#include <thread>
#include "Renderer.hpp"
#include "PlotPoints.cuh"
#include "CameraController.hpp"
#include "AITrainer.hpp"

struct cli_args {
    typedef char tasks_t;
    enum tasks:tasks_t {
        TRAIN_OR_RENDER,
        CONSTRUCT_DATASET,
    };
    bool quiet = false;
    std::vector<std::string> datasets_path;
    bool timeout = false; float timeout_s = 60 * 30;
    std::string workspace = "";
    bool NO_LIVE_RENDER = false;
    bool save_results = false; bool save_train_images = true;
    bool train = true;
    bool train_environment = false;
    float example_refresh_rate = -1;
    int batch_size = 3;
    int ndim = 3;
    int nn_depth = 4;
    tasks_t task = TRAIN_OR_RENDER;

    cli_args(int argc, char** argv) {
        if (argc == 1) {
            std::cerr << "no arguments provided. Use --help to get help."; exit(-1);
        }
        bool save_auto = false;
        bool load_set = false;
        for (int i = 1; i < argc; ++i) {
            std::string v = argv[i];
            //not a great implemenntation...
            if (v == "--quiet")quiet = true;
            else if (v == "--timeout") { timeout = true; timeout_s = atof(argv[++i]); }
            else if (v == "--batch") { batch_size = atoi(argv[++i]); }
            else if (v == "--train-environment") { train_environment = true; }
            else if (v == "--autosave") { save_auto = save_results = true; }
            else if (v == "--dataset") { load_set = true; datasets_path.push_back(argv[++i]);}
            else if (v == "--workspace") { load_set = true; workspace = argv[++i]; }
            else if (v == "--no-render") { NO_LIVE_RENDER = true; }
            else if (v == "--no-train") { train = false; }
            else if (v == "--extra-channels") { ndim = atoi(argv[++i]) + 3; }
            else if (v == "--nn-depth") { nn_depth = atoi(argv[++i]); }
            else if (v == "--make-workspace") { task = CONSTRUCT_DATASET; }
            else if (v == "--example-refresh") { example_refresh_rate = atof(argv[++i]); }
            else if (v == "--help" || v == "-h" || v == "-?") {
                std::cerr << "todo: write help\n";
                exit(-1);
            }
            else {
                std::cerr << "No idea what argument `" << v << "` means.\n"; exit(-1);
            }
        }
        if (example_refresh_rate <= 0) {
            example_refresh_rate = (train ? 2 : 0.3);
        }
        return;
    }
};

int main(int argc, char** argv)
{
    //todo: more options (disable GUI, etc.)
    //these would all be read from argv in a theoretical final version.
    cli_args args{ argc,argv };
    if(!args.quiet)std::cout << "All libraries loaded.\n";
    try {
        std::shared_ptr<DataSet> dataSet = nullptr;
        std::shared_ptr<NetworkPointer> network = nullptr;
        if (args.task == cli_args::CONSTRUCT_DATASET) {
            if (!args.quiet)std::cout << "Started reading dataset...\n";
            dataSet = std::make_shared<DataSet>(args.datasets_path, /*autoload = */ true, /*loadTrainData = */ false, args.quiet);
            if (!dataSet->isValid()) { std::cerr << "DataSet Invalid!"; return (int)std::errc::invalid_argument; }
            if (!args.quiet)std::cout << "Finished reading dataset...\n";
            dataSet->expand_to_ndim(args.ndim);
            if (!args.quiet)std::cout << "Initializing network...\n";
            network = std::make_shared<NetworkPointer>(dataSet,args.ndim,args.nn_depth);
            if (!args.quiet)std::cout << "Network initialized...\n";
        }
        else {
            if (!args.quiet)std::cout << "Initializing network and data...\n";
            network = NetworkPointer::load(args.workspace, true, true, args.quiet);
            if (!args.quiet)std::cout << "Network and data initialized...\n";
            dataSet = network->getDataSet();
        }

        NetworkPointer& nw = *network;

        if (args.task == cli_args::TRAIN_OR_RENDER) {
            nw.setBatchSize(args.batch_size);
            nw.train_images(args.train_environment);
            if (args.NO_LIVE_RENDER) {
                if (args.timeout)
                    nw.train_long((unsigned long long)(args.timeout_s * 1e3), args.quiet);
                else std::cerr << "Please provide a timeout value.";
            }
            else {
                if (args.train) {
                    if (!args.quiet)std::cout << "First train (initializes some torch things)\n";
                    nw.train_frame(0);
                    if (!args.quiet)std::cout << "First train done\n";
                }
                std::shared_ptr<InteractiveCameraData> cam_data = std::make_shared<InteractiveCameraData>(1, PI * 1 / 3, 0.01f, 1e9f);
                if (!args.quiet)std::cout << "Initializing renderer...\n";
                Renderer r{ "Main Window" };
                r.update();
                //std::this_thread::sleep_for(std::chrono::milliseconds(1));//allow idle time.
                r.ensureWH(1664, 512);
                auto old_frame = std::chrono::high_resolution_clock::now();
                //std::this_thread::sleep_for(std::chrono::milliseconds(1));
                CameraController controller{ r,cam_data };//       x0  x1  y0  y1  z flexible-resolution
                r.createView(Renderer::ViewTypeEnum::MAIN_VIEW   , .0, .5, .0, .5, 0, true);
                r.createView(Renderer::ViewTypeEnum::TRAIN_VIEW_1, .5, 1., .0, .5, 0, false);
                r.createView(Renderer::ViewTypeEnum::TRAIN_VIEW_2, .5, 1., .5, 1., 0, false);
                r.createView(Renderer::ViewTypeEnum::POINTS_VIEW , .0, .5, .5, 1., 0, false);
                if (!args.quiet)std::cout << "Renderer initialized...\n";
                auto last_example_update = std::chrono::high_resolution_clock::now().operator-=(std::chrono::nanoseconds((long long)5e9));
                auto example_refresh_s = args.example_refresh_rate;
                r.update();
                auto all_start = std::chrono::high_resolution_clock::now();
                while (!r.shouldClose()) {
                    if (args.train) {
                        nw.train_frame(100);
                        if (!args.quiet)nw.getTrainingStatus().print_pretty(std::cout);
                    }
                    else {
                        //std::cout << cam_data.translation.x << ' ' << cam_data.translation.y << ' ' << cam_data.translation.z << '\n';
                        //std::cout << cam_data.transform << '\n';
                    }
                    auto current_frame = std::chrono::high_resolution_clock::now();
                    controller.processMovements();
                    if (cam_data->use_neural) {
                        //todo? get scene from camera data?
                        nw.plotResultToRenderer(r, dataSet->scene(0), cam_data, Renderer::ViewTypeEnum::MAIN_VIEW);
                    }
                    else {
                        nw.plotToRenderer(r, dataSet->scene(0), cam_data, Renderer::ViewTypeEnum::MAIN_VIEW);
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

                    if (args.timeout) {
                        auto curr = std::chrono::high_resolution_clock::now();
                        if ((curr - all_start).count() * 1e-9 > args.timeout_s)
                            break;
                    }
                }
            }
        }
        if (!args.quiet)std::cout << "Saving to: " << args.workspace << "...\n";
        nw.save(args.workspace, fileType::CUSTOM_BINARY, true, args.save_train_images);
        if (!args.quiet)std::cout << "Saved!\n";

        if (!args.quiet)std::cout << "Shutting down...\n";
        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        auto cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaDeviceReset failed!";
            return 1;
        }
    }
    catch (std::runtime_error x) {
        std::cerr << "Uncaught error:\n" << x.what();
        return -1;
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