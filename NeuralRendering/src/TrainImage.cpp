#include "TrainImage.hpp"
#include "stream_binary_utils.hpp"
#include "string_utils.hpp"
#include <sstream>

fileType_t getFileType(const std::string& pth) {
    if (endswith(pth, ".txt")) return fileType::TEXT;
    if (endswith(pth, ".bin")) return fileType::CUSTOM_BINARY;
    return fileType::TORCH_ARCHIVE;
}

//todo: rewrite this entire function slowly.
TrainingImage::TrainingImage(const std::string& fn, bool autoload) : pth{ fn } {
    if (autoload)load_image();
}

void TrainingImage::load_image(){
    switch (getFileType(pth)) {
        case TEXT: {
            std::ifstream f{ pth.c_str() };
            camera_data = CameraDataItf::from_serial(/*text = */true, f);
            width = camera_data->get_width();
            height = camera_data->get_height();
            int clr;
            std::vector<float> data;
            data.reserve(4ULL * width * height);
            while (f >> clr) {
                float v = (float)clr / (16 * 255.0f);
                data.push_back(v);
                data.push_back(v);
                data.push_back(v);
                data.push_back(1.0f);
            }
            //#if _DEBUG //Actually, this is fairly cheap. Might as well do it during non-debug as well.
            if (data.size() != 4LL * width * height) {
                std::cerr << "Only " << data.size() / 4 << " pixels found; expected " << width * height << '\n';
                std::cerr << "Found " << data.size() << " numbers.\n";
                assert(false);
            }
            //#endif // _DEBUG
            target = torch::tensor(std::move(data));
            target = target.reshape({ height,width,4 }).cuda();
            //std::cerr << target;
        }break;
        case CUSTOM_BINARY: {
            std::ifstream f{ pth, std::ios::binary };
            if (!f.is_open()) {
                std::cerr << "invalid image file: not found:" << pth << '\n';
                throw "file not found";//todo: figure something out with these exceptions
            }

            camera_data = CameraDataItf::from_serial(/*text = */false, f);
            width = camera_data->get_width();
            height = camera_data->get_height();
            std::vector<unsigned char> data;
            readBinaryIntoArray(data, f, -1);
            //#if _DEBUG //Actually, this is fairly cheap. Might as well do it during non-debug as well.
            if (data.size() != 4LL * width * height) {
                std::cerr << "Only " << data.size() / 4 << " pixels found; expected " << width * height << '\n';
                std::cerr << "Found " << data.size() << " numbers.\n";
                assert(false);
            }
            //#endif // _DEBUG
            //todo?: use unbind instead
            target = torch::tensor(std::move(data), torch::TensorOptions().dtype(torch::kByte));
            target = target.cuda().reshape({ height,width,4 }).to(torch::kFloat).divide(255);
            target.transpose(0, 2)[0] *= target.transpose(0, 2)[-1];
            target.transpose(0, 2)[1] *= target.transpose(0, 2)[-1];
            target.transpose(0, 2)[2] *= target.transpose(0, 2)[-1];
        }break;
        case TORCH_ARCHIVE: {
            torch::serialize::InputArchive archive;
            archive.load_from(pth.c_str());
            torch::IValue cam_data;
            archive.read("cam", cam_data);
            std::string s_cam_data = cam_data.toString()->string();
            std::stringstream ss_cam_data{ s_cam_data };
            camera_data = CameraDataItf::from_serial(false, ss_cam_data);
            width = camera_data->get_width();
            height = camera_data->get_height();
            archive.read("data", target, true); target = target.cuda();
        }break;
        default:
            std::cerr << "unknown format at path: " << pth << '\n';
            target = torch::zeros({});
            break;
    }
}

void TrainingImage::unload(){
    camera_data = nullptr;
    is_loaded = false;
    target = torch::tensor(0);
}

void TrainingImage::copyTo(const std::string& dest){
    fileType_t type = getFileType(dest);
    fileType_t myType = getFileType(pth);
    if (type == myType) {
        //https://en.cppreference.com/w/cpp/filesystem/copy is c++17 only...
        std::ifstream  src(pth, std::ios::binary);
        std::ofstream  dst(dest, std::ios::binary);
        dst << src.rdbuf();
        return;
    }
    if (!is_loaded)load_image();
    switch (type) {
    case TEXT:{
        std::ofstream  g(dest);
        torch::Tensor tmp = target.cpu().contiguous();
        float* data = tmp.contiguous().data_ptr<float>();
        int idx=0;
        if (camera_data->type() == PINHOLE_PROJECTION) {
            PinholeCameraData* cam = (PinholeCameraData*)camera_data.get();
            g << 0 << '\n' << cam->ppy << ' ' << cam->ppx << ' ' << cam->fy << ' ' << cam->fx << '\n';
        } else throw "can't save camera data:";
        g << tmp.size(0) << ' ' << tmp.size(1) << '\n';
        for (int i = 0; i < tmp.size(0); ++i) {
            for (int j = 0; j < tmp.size(1); ++j) {
                for (int k = 0; k < tmp.size(2); ++k) {
                    g << data[idx] << ' ';
                }
            }
            g << data[idx] << '\n';
        }
    }
    case CUSTOM_BINARY:{
        //todo
    }
    case TORCH_ARCHIVE: {
        torch::serialize::OutputArchive archive;
        auto cam_serial = cam()->serialized(false);
        archive.write("data", target);
        archive.write("cam", cam()->serialized(false));
        archive.save_to(dest);
    }
    }
}
