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
		std::stringstream ss{ std::stringstream::out };
		ss << CameraType::PINHOLE_PROJECTION << '\n';
		write_transform(ss, transform, /*text = */ true);
		ss << ppy << ' ' << ppx << ' ' << fy << ' ' << fx << ' ' << h << ' ' << w << ' ';
		return ss.str();
	}
	else {
		std::stringstream ss{ std::stringstream::out|std::ios::binary };
		writeOneBinary(ss, (int)CameraType::PINHOLE_PROJECTION);
		write_transform(ss, transform, /*text = */ false);
		writeOneBinary(ss, ppy);writeOneBinary(ss, ppx);writeOneBinary(ss, fy);writeOneBinary(ss, fx);
		writeOneBinary(ss, h);writeOneBinary(ss, w);
		ss.sync();
		return ss.str();
	}
	return "";
}

std::string RadialCameraData::serialized(bool text) const
{
	if (text) {
		std::stringstream ss{ std::stringstream::out };
		ss << CameraType::RADIAL << '\n';
		write_transform(ss, transform, /*text = */ true);
		ss << ppy << ' ' << ppx << ' ' << fy << ' ' << fx << ' ' << k1 << ' ' << k2 << ' ' << h << ' ' << w << ' ';
		return ss.str();
	}
	else {
		std::stringstream ss{ std::stringstream::out | std::ios::binary };
		writeOneBinary(ss, (int)CameraType::RADIAL);
		write_transform(ss, transform, /*text = */ false);
		writeOneBinary(ss, ppy); writeOneBinary(ss, ppx); writeOneBinary(ss, fy); writeOneBinary(ss, fx); writeOneBinary(ss, k1); writeOneBinary(ss, k2);
		writeOneBinary(ss, h); writeOneBinary(ss, w);
		ss.sync();
		return ss.str();
	}
	return "";
}

std::unique_ptr<RadialCameraData> readRadial(bool text, std::istream& f) {
	if (text) {
		int t, width, height; float ppx, ppy, fx, fy, k1, k2;
		float4x4 transform = read_transform(f, /*text = */ true);
		f >> ppy >> ppx >> fy >> fx >> k1 >> k2 >> height >> width;
		RadialCameraData camera_data_radial;
		camera_data_radial.fx = fx;
		camera_data_radial.ppx = ppx;
		camera_data_radial.fy = fy;
		camera_data_radial.ppy = ppy;
		camera_data_radial.transform = transform;
		camera_data_radial.near_clip = 1e-9f; camera_data_radial.far_clip = 1e9f;//should not just be infinity
		camera_data_radial.k1 = k1; camera_data_radial.k2 = k2;
		camera_data_radial.w = width; camera_data_radial.h = height;
		return std::make_unique<RadialCameraData>(camera_data_radial);
	}
	else {

		int width, height; float ppx, ppy, fx, fy, k1, k2;
		float4x4 transform = read_transform(f, /*text = */ false);
		readBinary(f, &ppy); readBinary(f, &ppx); readBinary(f, &fy); readBinary(f, &fx); readBinary(f, &k1); readBinary(f, &k2);
		readBinary(f, &height); readBinary(f, &width);
		RadialCameraData camera_data_radial;
		camera_data_radial.fx = fx;
		camera_data_radial.ppx = ppx;
		camera_data_radial.fy = fy;
		camera_data_radial.ppy = ppy;
		camera_data_radial.transform = transform;
		camera_data_radial.near_clip = 1e-9f; camera_data_radial.far_clip = 1e9f;
		camera_data_radial.k1 = k1; camera_data_radial.k2 = k2;
		camera_data_radial.w = width; camera_data_radial.h = height;
		return std::make_unique<RadialCameraData>(camera_data_radial);
	}
	return nullptr;
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
		case RADIAL: {
			return readRadial(text, f);
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