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
#include <typeinfo>
//Note: endianness matters, you're probably fine so long as it was saved from the same OS.
template <typename T>
void readBinary(std::istream& f, T* data, int num = 1) {
    f.read((char*)data, sizeof(T) * num);
}

template <typename T>
T readOneBinary(std::istream& f) {
    T data;
    f.read((char*)&data, sizeof(T));
    return data;
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
        num_bytes_left = max_size;
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