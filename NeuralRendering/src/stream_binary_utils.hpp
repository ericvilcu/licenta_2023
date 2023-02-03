#pragma once
#include <sys/stat.h>
#ifdef WIN32
#include <direct.h>
inline int MakeDirectory_(const char* location) {
    return _mkdir(location);
}
#endif
#include <istream>
#include <string>
#include <vector>
#include <typeinfo>
//Note: endianness matters, you're probably fine so long as it was saved from the same OS.
template <typename T>
void readBinary(std::istream& f, T* data, int num = 1) {
    f.read((char*)data, sizeof(T) * num);
}
template <typename T>
void writeBinary(std::ostream& f, T* data, int num = 1) {
    f.write((char*)data, sizeof(T) * num);
}

template <typename T>
T readOneBinary(std::istream& f) {
    T data;
    f.read((char*)&data, sizeof(T));
    return data;
}

template <typename T>
void writeOneBinary(std::ostream& f, const T& data) {
    f.write((char*)&data, sizeof(T));
}

template <typename T>
void readBinaryIntoArray(std::vector<T>& destination, std::istream& f, int max_size = -1) {
    size_t num_bytes_left = 0; 
    {
        auto first_pos = f.tellg();
        auto last_pos = f.seekg(0, f.end).tellg();
        f.seekg(first_pos);
        num_bytes_left = last_pos - first_pos;
    }
    if (max_size != -1 && max_size * sizeof(T) < num_bytes_left) {
        num_bytes_left = max_size * sizeof(T);
    }
    auto num_elements_left = num_bytes_left / sizeof(T);
#if _DEBUG
    if (num_bytes_left % sizeof(T) != 0) {
        std::cerr << "Data misaligned. " << num_bytes_left << " bytes were found for " << typeid(T).name() << " with length " << sizeof(T) << '\n';
        throw "misaligned";//todo: figure out how to proparly handle these things
    }
#endif
    auto last = destination.size();
    destination.resize(destination.size() + num_elements_left);
    f.read((char*)&destination[last], num_bytes_left);
}

template <typename T>
void writeBinaryArray(std::vector<T>& data, std::ostream& f, size_t start = 0, size_t limit = SIZE_MAX) {
    if (data.size() < limit) {
        limit = data.size();
    }
    f.write((char*)&data[0], limit * sizeof(T));
}

//returns true if it was already present
inline bool makeDirIfNotPresent(const std::string& path) {

    struct stat info;
    if (!(bool)stat((path).c_str(), &info)) {//if it is accessible
        if (!(bool)(info.st_mode & S_IFDIR)) {//if it is not a directory
            throw "Already exists, but not a dir.";//seriously figure something out with exceptions, you can't just keep throwing const char*
        }
        return true;
    }
    MakeDirectory_(path.c_str());
    return false;
}

inline bool existsAndIsFile(const std::string& path) {
    struct stat info;
    if (!(bool)stat((path).c_str(), &info)) {//if it is accessible
        if (!(bool)(info.st_mode & S_IFDIR)) {//if it is not a directory
            return true;//throw "Already exists and not a dir.";
        }
        return false;
    }
    return false;
}

#include "dtype4x4.cuh"

inline float4x4 readBinaryTransform(std::istream& f) {
    float rotation[9];
    float translation[3];
    readBinary(f, rotation, 9);
    readBinary(f, translation, 3);
    /*float4x4 r{
        rotation[0], rotation[1], rotation[2], 0,
        rotation[3], rotation[4], rotation[5], 0,
        rotation[6], rotation[7], rotation[8], 0,
        0,0,0,1,
    };

    float4x4 t{
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        translation[0],translation[1],translation[2],1,
    };
    return t * r;*/
    float4x4 rt{
        rotation[0], rotation[1], rotation[2], translation[0],
        rotation[3], rotation[4], rotation[5], translation[1],
        rotation[6], rotation[7], rotation[8], translation[2],
        0,0,0,1,
    };
    return rt;
}

inline float4x4 read_transform(std::istream& f, bool text = true) {
    if (!text)return readBinaryTransform(f);
    float rotation[9];
    float translation[3];
    if (f >> rotation[0] >> rotation[1] >> rotation[2]
        >> rotation[3] >> rotation[4] >> rotation[5]
        >> rotation[6] >> rotation[7] >> rotation[8]
        >> translation[0] >> translation[1] >> translation[2]) {
        float4x4 rt{
            rotation[0], rotation[1], rotation[2], translation[0],
            rotation[3], rotation[4], rotation[5], translation[1],
            rotation[6], rotation[7], rotation[8], translation[2],
            0,0,0,1,
        };
        return rt;
    }
    return float4x4();
}

inline void write_transform(std::ostream& f, float4x4 data, bool text=true) {
    float rotation[9];
    float translation[3];
    rotation[0] = data[0][0]; rotation[1] = data[1][0]; rotation[2] = data[2][0];
    rotation[3] = data[0][1]; rotation[4] = data[1][1]; rotation[5] = data[2][1];
    rotation[6] = data[0][2]; rotation[7] = data[1][2]; rotation[8] = data[2][2];

    translation[0] = data[3][0];
    translation[1] = data[3][1];
    translation[2] = data[3][2];
    if (text) {
        f << rotation[0] << ' ' << rotation[1] << ' ' << rotation[2] << '\n'
            << rotation[3] << ' ' << rotation[4] << ' ' << rotation[5] << '\n'
            << rotation[6] << ' ' << rotation[7] << ' ' << rotation[8] << '\n'
            << translation[0] << ' ' << translation[1] << ' ' << translation[2] << '\n';
    } else {
        writeBinary(f, rotation, 9);
        writeBinary(f, translation, 3);
    }
}

