#pragma once
#include <memory>
#include <string>
#include <vector>
#include "CameraData.hpp"
//This may make for some bloated files.
//todo? (unimportant) find a way to not include this giant mess in headers.
#include "HeaderThatSupressesWarnings.h"
#include <torch/torch.h>
#include "HeaderThatReenablesWarnings.h"
//todo: comments.
class TrainingImage {
    std::shared_ptr<CameraDataItf> camera_data;
    std::string pth;
public:
    int scene_id;
    int width = 0; int height = 0;
    torch::Tensor target;
    bool is_loaded = false;
    TrainingImage(const std::string& fn, int scene_id, bool autoload = true);
    void load_image();
    void unload();
    void copyTo(const std::string& ot);
    //deprecated, todo: remove when no longer used.
    CameraDataItf& camera() {
        return *camera_data;
    }
    std::shared_ptr<CameraDataItf> cam() { return camera_data; };
};
#ifdef __NVCC__
#define fileType_t char
#else
typedef char fileType_t;
#endif
enum fileType :fileType_t {
    TEXT, CUSTOM_BINARY, TORCH_ARCHIVE
};

inline const char* file_extension_from_type(fileType_t mode) {
    return (mode == TEXT ? ".txt" : (mode == CUSTOM_BINARY ? ".bin" : ""));
}
