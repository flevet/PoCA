#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Cuda/BasicOperationsImage.h>
#include <Cuda/ConnectedComponents.h>
#include <General/Engine.hpp>
#include <General/Image.hpp>
#include <Interfaces/MyObjectInterface.hpp>

namespace {
	uint32_t idx2(uint32_t _x, uint32_t _y, uint32_t _width)
	{
		return _y * _width + _x;
	}

	uint32_t idx3(uint32_t _x, uint32_t _y, uint32_t _z, uint32_t _width, uint32_t _height)
	{
		return (_z * _height + _y) * _width + _x;
	}

	poca::core::Image<uint8_t>* makeRawImage(std::vector<uint8_t>&& _pixels, uint32_t _w, uint32_t _h, uint32_t _d)
	{
		poca::core::Image<uint8_t>* image = new poca::core::Image<uint8_t>(poca::core::RAW);
		image->pixels() = std::move(_pixels);
		image->finalizeImage(_w, _h, _d);
		image->setType(poca::core::UINT8);
		return image;
	}

	poca::core::Image<uint32_t>* makeLabelImage(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _h, uint32_t _d)
	{
		poca::core::Image<uint32_t>* image = new poca::core::Image<uint32_t>(poca::core::LABEL);
		std::vector<uint32_t>& labels = image->pixels();
		labels.resize(_raw.size());
		run_face_connected_component_pipeline(_raw.data(), labels.data(), _w, _h, _d);
		image->finalizeImage(_w, _h, _d);
		image->setType(poca::core::UINT32);
		computeFeaturesLabelImage(image);
		return image;
	}

	poca::core::MyObjectInterface* createObject(
		const std::string& _name,
		poca::core::ImageInterface* _raw,
		poca::core::ImageInterface* _labels)
	{
		return poca::core::Engine::instance()->createObjectFromImages(
			".",
			_name,
			{ { _raw, "RAW uint8 problem cases" }, { _labels, "Face CCL labels" } });
	}
}

poca::core::MyObjectInterface* createFaceConnectedComponentTestDataset2D()
{
	const uint32_t w = 12, h = 10, d = 1;
	std::vector<uint8_t> raw(w * h * d, 0);
	const std::array<std::pair<uint32_t, uint32_t>, 18> foreground = { {
		{ 1, 1 }, { 2, 2 },
		{ 1, 4 }, { 2, 4 }, { 1, 5 }, { 2, 5 },
		{ 3, 1 }, { 4, 1 },
		{ 6, 1 }, { 6, 2 },
		{ 8, 2 }, { 9, 3 },
		{ 8, 6 }, { 9, 6 }, { 10, 6 }, { 9, 5 }, { 9, 7 },
		{ 4, 7 }
	} };

	for (const auto& p : foreground)
		raw[idx2(p.first, p.second, w)] = 255;

	poca::core::Image<uint32_t>* labels = makeLabelImage(raw, w, h, d);
	poca::core::Image<uint8_t>* image = makeRawImage(std::move(raw), w, h, d);
	return createObject("Face_connected_components_2D_test", image, labels);
}

poca::core::MyObjectInterface* createFaceConnectedComponentTestDataset3D()
{
	const uint32_t w = 6, h = 6, d = 4;
	std::vector<uint8_t> raw(w * h * d, 0);
	const std::array<std::tuple<uint32_t, uint32_t, uint32_t>, 17> foreground = { {
		{ 0, 0, 0 }, { 1, 1, 1 },
		{ 1, 3, 0 }, { 2, 3, 0 },
		{ 4, 1, 0 }, { 4, 2, 0 },
		{ 0, 4, 1 }, { 0, 4, 2 },
		{ 2, 2, 2 }, { 3, 3, 2 },
		{ 4, 4, 1 }, { 5, 4, 1 }, { 5, 5, 1 }, { 5, 5, 2 },
		{ 2, 0, 3 }, { 3, 0, 3 }, { 3, 1, 3 }
	} };

	for (const auto& p : foreground)
		raw[idx3(std::get<0>(p), std::get<1>(p), std::get<2>(p), w, h)] = 255;

	poca::core::Image<uint32_t>* labels = makeLabelImage(raw, w, h, d);
	poca::core::Image<uint8_t>* image = makeRawImage(std::move(raw), w, h, d);
	return createObject("Face_connected_components_3D_test", image, labels);
}
