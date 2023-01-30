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
    std::unique_ptr<CameraDataItf> camera_data;
    std::string pth; int vers;
public:
    int width = 0; int height = 0;
    torch::Tensor target;
    bool is_loaded = false;
    TrainingImage(const std::string& fn, int vers, bool autoload = true);
    void reload();
    void unload();
    CameraDataItf& camera() {
        return *camera_data;
    }
};
