#include <chrono>
#include <iostream>
#include <thread>
#include "Renderer.hpp"
//#include "PlotPoints.cuh"
#include "cli_args.hpp"
#include "CameraController.hpp"
#include "AITrainer.hpp"
#include <opencv2/opencv.hpp>

cli_args::cli_args(int argc, char** argv) {
    if (argc == 1) {
        std::cerr << "no arguments provided. Use --help to get help."; exit(-1);
    }
    bool load_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string v = argv[i];
        //not a great implemenntation...
        if (v == "--quiet")quiet = true;
        else if (v == "--timeout") { timeout = true; timeout_s = atof(argv[++i]); }
        else if (v == "--batch") { batch_size = atoi(argv[++i]); }
        else if (v == "--train-environment") { train_environment = true; }
        else if (v == "--dataset") { load_set = true; datasets_path.push_back(argv[++i]); }
        else if (v == "--workspace") { load_set = true; workspace = argv[++i]; }
        else if (v == "--no-render") { NO_LIVE_RENDER = true; }
        else if (v == "--autosave") { autosave_freq = atof(argv[++i]); }
        else if (v == "--no-train") { train = false; }
        else if (v == "--extra-channels") { ndim = atoi(argv[++i]) + 3; }
        else if (v == "--nn-depth") { nn_depth = atoi(argv[++i]); }
        else if (v == "--make-workspace") { new_workspace = true; }
        else if (v == "--example-refresh") { example_refresh_rate = atof(argv[++i]); }
        else if (v == "--save-samples") { sample_save_path = argv[++i]; }
        else if (v == "--random-train") { random_train = std::string(argv[++i]) == "true"; }
        else if (v == "--help" || v == "-h" || v == "-?") {
            std::cerr << "todo: write help\n";
            exit(0);
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
cli_args* global_args;
int main(int argc, char** argv)
{
    //todo: more options (disable GUI, etc.)
    //these would all be read from argv in a theoretical final version.
    cli_args args{ argc,argv };
    global_args = &args;
    if(!args.quiet)std::cout << "All libraries loaded.\n";
    try {
        if (args.new_workspace) {
            if (!args.quiet)std::cout << "Started reading dataset...\n";
            auto _dataSet = std::make_shared<DataSet>(args.datasets_path, /*autoload = */ true, /*loadTrainData = */ false, args.quiet);
            if (!_dataSet->isValid()) { std::cerr << "DataSet Invalid!"; return (int)std::errc::invalid_argument; }
            if (!args.quiet)std::cout << "Finished reading dataset...\n";
            _dataSet->expand_to_ndim(args.ndim);

            if (!args.quiet)std::cout << "Initializing new network...\n";
            auto _network = std::make_shared<NetworkPointer>(_dataSet,args.ndim,args.nn_depth);
            if (!args.quiet)std::cout << "Network initialized...\n";

            if (!args.quiet)std::cout << "Saving new workspace...\n";
            _network->save(args.workspace, fileType::CUSTOM_BINARY, true, args.save_train_images);
            if (!args.quiet)std::cout << "New workspace initialized...\n";
        }

        std::shared_ptr<DataSet> dataSet = nullptr;
        std::shared_ptr<NetworkPointer> network = nullptr;
        if (!args.quiet)std::cout << "Loading network and data from workspace...\n";
        network = NetworkPointer::load(args.workspace, true, true, args.quiet);
        if (!args.quiet)std::cout << "Network and data loaded...\n";
        dataSet = network->getDataSet();

        NetworkPointer& nw = *network;
        nw.setBatchSize(args.batch_size);
        nw.train_images(args.train_environment);
        if (args.NO_LIVE_RENDER) {
            if (args.timeout)
                nw.train_long((unsigned long long)(args.timeout_s * 1e9), (unsigned long long)(args.autosave_freq * 1e9), args.workspace, args.quiet);
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
            auto last_autosave = std::chrono::high_resolution_clock::now();
            auto example_refresh_s = args.example_refresh_rate;
            r.update();
            auto all_start = std::chrono::high_resolution_clock::now();
            while (!r.shouldClose()) {
                if (args.train) {
                    nw.train_frame(100);
                    if (!args.quiet) {
                        auto& status = nw.getTrainingStatus();
                        if (status.epoch_count != 0)status.print_pretty(std::cout);
                    }
                }
                else {
                    //std::cout << cam_data.translation.x << ' ' << cam_data.translation.y << ' ' << cam_data.translation.z << '\n';
                    //std::cout << cam_data.transform << '\n';
                }
                auto current_frame = std::chrono::high_resolution_clock::now();
                controller.processMovements();
                if (cam_data->use_neural) {
                    nw.plotResultToRenderer(r, dataSet->scene(cam_data->selected_scene%dataSet->num_scenes()), cam_data, Renderer::ViewTypeEnum::MAIN_VIEW);
                }
                else {
                    nw.plotToRenderer(r, dataSet->scene(cam_data->selected_scene%dataSet->num_scenes()), cam_data, Renderer::ViewTypeEnum::MAIN_VIEW);
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
                    if ((curr - all_start).count() * 1e-9 > args.timeout_s) {
                        break;
                    }
                }

                //check if it should autosave
                if ((current_frame - last_autosave).count() * 1e-9 > args.autosave_freq && args.autosave_freq >= 0) {
                    //note: making the autosave crash-resistant is not a bad idea, but may be too much work.
                    //by the way, I mean BSOD-resistant, since that's happened to me twice already, and that would require modifying the loading as well.
                    if (!args.quiet)std::cout << "(autosave) Saving to: " << args.workspace << "...\n";
                    nw.save(args.workspace, fileType::CUSTOM_BINARY, true, args.save_train_images);
                    if (!args.quiet)std::cout << "(autosave) Saved!\n";
                    last_autosave = std::chrono::high_resolution_clock::now();
                }
            }
            if (!args.quiet)std::cout << "Shutting down live render...\n";
        }
        if (!args.quiet)std::cout << "Saving to: " << args.workspace << "...\n";
        nw.save(args.workspace, fileType::CUSTOM_BINARY, true, args.save_train_images);
        if (!args.quiet)std::cout << "Saved!\n";

        if (args.sample_save_path != "") {
            if (!args.quiet)std::cout << "Saving some samples to: \"" << args.sample_save_path << "\"\n";
            for (int i = 0; i < dataSet->num_scenes(); ++i) {
                if (!args.quiet)std::cout << "Saving samples from scene " << i << "\n";
                for (int j = 0; j < dataSet->scene(i).num_train_images(); ++j) {
                    if (!args.quiet)std::cout << "Rendering sample " << j+1 << "/" << dataSet->scene(i).num_train_images() << "\n";
                    const auto& d=dataSet->scene(i).loadOne(j);
                    auto cam = d->cam();
                    if (cam->type() == PINHOLE_PROJECTION) {
                        //maybe something that slightly alters position/angle would be nice as well? to get some non-train sample images as well
                        std::shared_ptr<PinholeCameraData> full_cam = std::static_pointer_cast<PinholeCameraData>(cam);
                        auto& randomly_nudge_transform = [](const float4x4& data) {
                            auto& signed_rand = []() {return (float)(rand() - (RAND_MAX >> 1)) / (RAND_MAX >> 1); };
                            float4x4 mod;
                            mod.identity();
                            //not sure if this is the right order of operations.
                            if (rand() % 2) {
                                float4x4 rotation = CameraDataItf::transform_from((2 * PI) * signed_rand(), 0.5f * signed_rand(), 0);
                                mod = mod * rotation;
                            }
                            if (rand()%2) {
                                //how exactly does this almost result in infinity and the camera being close to the far plane?
                                float4x4 translation; translation.identity();
                                //NOTE: .1 is kinda random and dependant on the scene's scale
                                constexpr float scale = 0.1f;
                                translation[3][0] += scale * signed_rand();
                                translation[3][1] += scale * signed_rand();
                                translation[3][2] += scale * signed_rand();
                                mod = mod * translation;
                            }
                            return mod * data;
                        };
                        full_cam->transform = randomly_nudge_transform(full_cam->transform);
                    }
                    auto tsr = torch::clamp(nw.forward(i, cam) * 255, 0, 255).to(torch::kByte);
                    if (!args.quiet)std::cout << "Saving sample " << j + 1 << "/" << dataSet->scene(i).num_train_images() << "\n";
                    auto cpu_tensor = tsr.cpu().contiguous();
                    auto ptr=cpu_tensor.data_ptr<unsigned char>();
                    int width = cpu_tensor.sizes()[0];
                    int height = cpu_tensor.sizes()[1];
                    //NOTE: this is the ONE use of OpenCV. Maybe try to find a smaller alternative later.
                    cv::Mat output_mat(cv::Size{ height, width }, CV_MAKETYPE(CV_8U, cpu_tensor.sizes()[2]), ptr);
                    cv::imwrite(args.sample_save_path+"/sav"+std::to_string(i)+"-"+std::to_string(j)+".png", output_mat);
                }
                if (!args.quiet)std::cout << "Samples from scene " << i << " saved\n";
            }
            if (!args.quiet)std::cout << "Samples rendered and saved!\n";
        }
        if (!args.quiet)std::cout << "Shutting down...\n";
        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        auto cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaDeviceReset failed!";
            return 1;
        }
    }
    catch (torch::Error x) {
        std::cerr << x.what() << '\n';
        return -1;
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