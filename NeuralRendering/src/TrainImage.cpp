#include "TrainImage.hpp"
#include "stream_binary_utils.hpp"
float4x4 read_transform_as_rotation_and_translation(std::istream& f) {
    float rotation[9];
    float translation[3];
    if (f >>  rotation[0]   >>  rotation[1]   >>  rotation[2]
          >>  rotation[3]   >>  rotation[4]   >>  rotation[5]
          >>  rotation[6]   >>  rotation[7]   >>  rotation[8]
          >> translation[0] >> translation[1] >> translation[2]) {
        float4x4 r{
            rotation[0], rotation[1], rotation[2], 0,
            rotation[3], rotation[4], rotation[5], 0,
            rotation[6], rotation[7], rotation[8], 0,
            0,0,0,1,
        };
        float4x4 t{
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            translation[0],-translation[1],-translation[2],1,
        };
        return t * r;
    }
    return float4x4();
}

float4x4 readBinaryTransform(std::istream& f) {
    float rotation[9];
    float translation[3];
    readBinary(f, rotation, 9);
    readBinary(f, translation, 3);
    float4x4 r{
        rotation[0], rotation[1], rotation[2], 0,
        rotation[3], rotation[4], rotation[5], 0,
        rotation[6], rotation[7], rotation[8], 0,
        0,0,0,1,
    };
    float4x4 t{
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        translation[0],-translation[1],-translation[2],1,
    };
    return t * r;
}


//todo: rewrite this entire function slowly.
TrainingImage::TrainingImage(const std::string& fn, int vers, bool autoload) :pth{ fn }, vers{ vers } {
    if (autoload)reload();
}
void TrainingImage::reload(){
    switch(vers&0xf00){
    case 0x500:{
        std::ifstream f{ pth.c_str() };
        int t; float ppx, ppy, fx, fy;
        float4x4 transform = read_transform_as_rotation_and_translation(f);
        f >> t >> ppy >> ppx >> fy >> fx >> height >> width;
        PinholeCameraData camera_data_MP;
        //todo: rewrite this in the files themselves.
        camera_data_MP.fx  = fx;
        camera_data_MP.ppx = ppx ;
        camera_data_MP.fy  = fy;
        camera_data_MP.ppy = ppy ;
        //camera_data_MP.fov_x = width / fx;
        //camera_data_MP.fov_rad = 2 * atanf(camera_data_MP.fov_x);//not necessary to calculate?
        camera_data_MP.transform = transform;//Note: this needs to be read from a separate file, that sucks and may be changed at a later date.
        camera_data_MP.near_clip = 1e-9f; camera_data_MP.far_clip = 1e9f;//should just be infinity
        camera_data_MP.w = width; camera_data_MP.h = height;
        camera_data = std::make_unique<PinholeCameraData>(camera_data_MP);
        //Only contains a grayscale image stored as an array of values with a maximum of around 4,080
        if (t == 0) {
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
        }
        else {
            target = torch::zeros({ height,width,4 }).cuda();
        }
    }break;
    case 0x600: {
        std::ifstream f{ pth, std::ios::binary };
        if (!f.is_open()) {
            std::cerr << "invalid image file: not found:" << pth << '\n';
            throw "file not found";//todo: figure something out with these exceptions
        }
        float4x4 transform = readBinaryTransform(f);
        int t; float ppx, ppy, fx, fy;
        readBinary(f, &t);
        readBinary(f, &ppy); readBinary(f, &ppx); readBinary(f, &fy); readBinary(f, &fx);
        readBinary(f, &height); readBinary(f, &width);
        PinholeCameraData camera_data_MP;
        camera_data_MP.fx = fx;
        camera_data_MP.ppx = ppx;
        camera_data_MP.fy = fy;
        camera_data_MP.ppy = ppy;
        camera_data_MP.transform = transform;
        camera_data_MP.near_clip = 1e-9f; camera_data_MP.far_clip = 1e9f;
        camera_data_MP.w = width; camera_data_MP.h = height;
        camera_data = std::make_unique<PinholeCameraData>(camera_data_MP);
        if (t == 0) {
            if ((vers & 0x0f0) == 0x010) {
                std::vector<unsigned char> data;
                readBinaryIntoArray(data, f, -1);
                //#if _DEBUG //Actually, this is fairly cheap. Might as well do it during non-debug as well.
                if (data.size() != 4LL * width * height) {
                    std::cerr << "Only " << data.size() / 4 << " pixels found; expected " << width * height << '\n';
                    std::cerr << "Found " << data.size() << " numbers.\n";
                    assert(false);
                }
                //#endif // _DEBUG
                //todo: use unbind instead
                target = torch::tensor(std::move(data), torch::TensorOptions().dtype(torch::kByte));
                target = target.cuda().reshape({ height,width,4 }).to(torch::kFloat).divide(255);
                target.transpose(0, 2)[0] *= target.transpose(0, 2)[-1];
                target.transpose(0, 2)[1] *= target.transpose(0, 2)[-1];
                target.transpose(0, 2)[2] *= target.transpose(0, 2)[-1];
            }
            else {
                int clr;
                std::vector<unsigned char> data;
                readBinaryIntoArray(data, f, -1);
                //#if _DEBUG //Actually, this is fairly cheap. Might as well do it during non-debug as well.
                if (data.size() != 3LL * width * height) {
                    std::cerr << "Only " << data.size() / 3 << " pixels found; expected " << width * height << '\n';
                    std::cerr << "Found " << data.size() << " numbers.\n";
                    assert(false);
                }
                //#endif // _DEBUG
                //todo: use unbind instead
                target = torch::tensor(std::move(data), torch::TensorOptions().dtype(torch::kByte));
                target = target.cuda().reshape({ height,width,3 }).to(torch::kFloat).divide(255);
                target = torch::cat({ target, torch::ones({ height,width,1}).cuda() }, -1);
            }
        }
        else {
            target = torch::zeros({ height,width,4 }).cuda();
        }
    }break;
    default:
        std::cerr << "Could not read training image from " << pth << ", with version " << std::hex << vers << std::dec << "due to invalid version" << '\n';
        assert(false);
    }
}

void TrainingImage::unload(){
    camera_data = nullptr;
    is_loaded = false;
    target = torch::tensor(0);
}
