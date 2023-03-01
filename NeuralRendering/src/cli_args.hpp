#pragma once
#include <string>
#include <vector>
struct cli_args;
extern cli_args* global_args;
struct cli_args {
    bool quiet = false;
    std::vector<std::string> datasets_path;
    bool timeout = false; float timeout_s = 60 * 30;
    std::string workspace = "";
    std::string sample_save_path = "";
    bool NO_LIVE_RENDER = false;
    bool save_train_images = true;
    bool train = true;
    bool train_environment = false;
    float example_refresh_rate = -1;
    int batch_size = 3;
    int ndim = 3;
    int nn_depth = 4;
    bool new_workspace = false;
    float autosave_freq = -1;//I somehow lost 3hrs of training so I'm putting this option in.
    bool random_train = false;
    cli_args(int argc, char** argv);
private:
    std::unique_ptr<std::ostream> logger;
public:
    template <typename T>
    inline void log(T data) {
        if (logger != nullptr) {
            (*logger) << data << '\n';
        }
    }
};
