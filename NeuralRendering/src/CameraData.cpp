#include <iostream>
#include <sstream>
#include "stream_binary_utils.hpp"
#include "CameraData.hpp"


std::string InteractiveCameraData::serialized(bool text) const
{
	//InteractiveCameraData is meant to represent a live, controllable camera with some extra options, it should not be used to store data about images.
	//This would only be useful if I wanted to store data about the interactive view's last position, which I do not.
	return "";
}


std::string PinholeCameraData::serialized(bool text) const
{
	if (text) {
		std::stringstream ss{};
		ss << CameraType::PINHOLE_PROJECTION << '\n';
		write_transform(ss, transform, /*text = */ true);
		ss << ppy << ' ' << ppx << ' ' << fy << ' ' << fx << ' ' << h << ' ' << w << ' ';
		return ss.str();
	}
	else {
		std::stringstream ss{ std::ios::binary };
		writeOneBinary(ss, (int)CameraType::PINHOLE_PROJECTION);
		write_transform(ss, transform, /*text = */ false);
		writeOneBinary(ss, ppy);writeOneBinary(ss, ppx);writeOneBinary(ss, fy);writeOneBinary(ss, fx);writeOneBinary(ss, h);writeOneBinary(ss, w);
		return ss.str();
	}
	return "";
}

std::string PinholeCameraData::debug_log() const
{
	return std::string();
}

std::unique_ptr<PinholeCameraData> readPinholeProjection(bool text, std::istream& f) {
	if (text) {
		int t, width, height; float ppx, ppy, fx, fy;
		float4x4 transform = read_transform(f, /*text = */ true);
		f >> ppy >> ppx >> fy >> fx >> height >> width;
		PinholeCameraData camera_data_MP;
		camera_data_MP.fx = fx;
		camera_data_MP.ppx = ppx;
		camera_data_MP.fy = fy;
		camera_data_MP.ppy = ppy;
		camera_data_MP.transform = transform;
		camera_data_MP.near_clip = 1e-9f; camera_data_MP.far_clip = 1e9f;//should not just be infinity
		camera_data_MP.w = width; camera_data_MP.h = height;
		return std::make_unique<PinholeCameraData>(camera_data_MP);
	}
	else {

		int width, height; float ppx, ppy, fx, fy;
		float4x4 transform = read_transform(f, /*text = */ false);
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
		return std::make_unique<PinholeCameraData>(camera_data_MP);
	}
	return nullptr;
}

std::unique_ptr<CameraDataItf> CameraDataItf::from_serial(bool text, std::istream& f){
	int t;
	if (text) {
		f >> t;
	}
	else t = readOneBinary<int>(f);
	switch (t) {
		case PINHOLE_PROJECTION: {
			return readPinholeProjection(text,f);
		}
		default:
			return nullptr;
	}
}

std::unique_ptr<CameraDataItf> CameraDataItf::from_serial(bool text, const std::string& data)
{
	std::stringstream ss{data};
	return from_serial(text, ss);
}
