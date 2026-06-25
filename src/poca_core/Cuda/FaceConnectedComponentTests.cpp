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
#include <General/TestRegistry.hpp>
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
	void drawHorizontalLine2D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _y, uint32_t _x0, uint32_t _x1)
	{
		for (uint32_t x = _x0; x <= _x1; x++)
			_raw[idx2(x, _y, _w)] = 255;
	}

	void drawVerticalLine2D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _x, uint32_t _y0, uint32_t _y1)
	{
		for (uint32_t y = _y0; y <= _y1; y++)
			_raw[idx2(_x, y, _w)] = 255;
	}

	void drawRectangleOutline2D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _x0, uint32_t _y0, uint32_t _x1, uint32_t _y1)
	{
		drawHorizontalLine2D(_raw, _w, _y0, _x0, _x1);
		drawHorizontalLine2D(_raw, _w, _y1, _x0, _x1);
		drawVerticalLine2D(_raw, _w, _x0, _y0, _y1);
		drawVerticalLine2D(_raw, _w, _x1, _y0, _y1);
	}

	void drawSerpentine2D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _x0, uint32_t _x1, uint32_t _y0, uint32_t _y1, uint32_t _step)
	{
		bool leftToRight = true;
		for (uint32_t y = _y0; y <= _y1; y += _step) {
			drawHorizontalLine2D(_raw, _w, y, _x0, _x1);
			uint32_t nextY = y + _step;
			if (nextY <= _y1)
				drawVerticalLine2D(_raw, _w, leftToRight ? _x1 : _x0, y, nextY);
			leftToRight = !leftToRight;
		}
	}

	void addLargeStressStructures2D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _h)
	{
		drawSerpentine2D(_raw, _w, 20, _w - 21, 20, 980, 8);

		for (uint32_t y = 1030; y < _h - 20; y += 5) {
			for (uint32_t x = 20; x < 720; x += 5) {
				_raw[idx2(x, y, _w)] = 255;
				_raw[idx2(x + 1, y + 1, _w)] = 255;
			}
		}

		for (uint32_t n = 0; n < 24; n++) {
			const uint32_t x0 = 850 + n * 18;
			const uint32_t y0 = 1060 + n * 12;
			const uint32_t x1 = _w - 40 - n * 18;
			const uint32_t y1 = _h - 40 - n * 12;
			drawRectangleOutline2D(_raw, _w, x0, y0, x1, y1);
		}

		for (uint32_t y = 1120; y < _h - 80; y += 80) {
			for (uint32_t x = 780; x < 1480; x++)
				_raw[idx2(x, y, _w)] = 255;
			for (uint32_t yy = y; yy < y + 54; yy++)
				_raw[idx2(780 + ((y / 80) % 2) * 699, yy, _w)] = 255;
		}
	}

	void drawLineX3D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _h, uint32_t _y, uint32_t _z, uint32_t _x0, uint32_t _x1)
	{
		for (uint32_t x = _x0; x <= _x1; x++)
			_raw[idx3(x, _y, _z, _w, _h)] = 255;
	}

	void drawLineY3D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _h, uint32_t _x, uint32_t _z, uint32_t _y0, uint32_t _y1)
	{
		for (uint32_t y = _y0; y <= _y1; y++)
			_raw[idx3(_x, y, _z, _w, _h)] = 255;
	}

	void drawLineZ3D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _h, uint32_t _x, uint32_t _y, uint32_t _z0, uint32_t _z1)
	{
		for (uint32_t z = _z0; z <= _z1; z++)
			_raw[idx3(_x, _y, z, _w, _h)] = 255;
	}

	void fillBox3D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _h, uint32_t _x0, uint32_t _y0, uint32_t _z0, uint32_t _x1, uint32_t _y1, uint32_t _z1)
	{
		for (uint32_t z = _z0; z <= _z1; z++)
			for (uint32_t y = _y0; y <= _y1; y++)
				for (uint32_t x = _x0; x <= _x1; x++)
					_raw[idx3(x, y, z, _w, _h)] = 255;
	}

	void drawBoxShell3D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _h, uint32_t _x0, uint32_t _y0, uint32_t _z0, uint32_t _x1, uint32_t _y1, uint32_t _z1)
	{
		fillBox3D(_raw, _w, _h, _x0, _y0, _z0, _x1, _y1, _z0);
		fillBox3D(_raw, _w, _h, _x0, _y0, _z1, _x1, _y1, _z1);
		fillBox3D(_raw, _w, _h, _x0, _y0, _z0, _x1, _y0, _z1);
		fillBox3D(_raw, _w, _h, _x0, _y1, _z0, _x1, _y1, _z1);
		fillBox3D(_raw, _w, _h, _x0, _y0, _z0, _x0, _y1, _z1);
		fillBox3D(_raw, _w, _h, _x1, _y0, _z0, _x1, _y1, _z1);
	}

	void drawSerpentine3D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _h, uint32_t _x0, uint32_t _x1, uint32_t _y0, uint32_t _y1, uint32_t _z0, uint32_t _z1)
	{
		uint32_t connectorX = _x0, connectorY = _y0, previousZ = _z0;
		bool hasPreviousSlice = false;
		for (uint32_t z = _z0; z <= _z1; z += 16) {
			bool leftToRight = true;
			for (uint32_t y = _y0; y <= _y1; y += 8) {
				drawLineX3D(_raw, _w, _h, y, z, _x0, _x1);
				uint32_t nextY = y + 8;
				if (nextY <= _y1)
					drawLineY3D(_raw, _w, _h, leftToRight ? _x1 : _x0, z, y, nextY);
				connectorX = leftToRight ? _x1 : _x0;
				connectorY = nextY <= _y1 ? nextY : y;
				leftToRight = !leftToRight;
			}
			if (hasPreviousSlice)
				drawLineZ3D(_raw, _w, _h, connectorX, connectorY, previousZ, z);
			hasPreviousSlice = true;
			previousZ = z;
		}
	}

	void addLargeStressStructures3D(std::vector<uint8_t>& _raw, uint32_t _w, uint32_t _h, uint32_t _d)
	{
		drawSerpentine3D(_raw, _w, _h, 16, _w - 17, 16, _h - 17, 4, _d - 5);

		for (uint32_t z = 10; z < _d - 8; z += 17)
			for (uint32_t y = 12; y < 500; y += 5)
				for (uint32_t x = 12; x < 500; x += 5) {
					_raw[idx3(x, y, z, _w, _h)] = 255;
					_raw[idx3(x + 1, y + 1, z + 1, _w, _h)] = 255;
				}

		for (uint32_t n = 0; n < 18; n++) {
			const uint32_t x0 = 560 + (n % 3) * 130;
			const uint32_t y0 = 120 + ((n / 3) % 3) * 130;
			const uint32_t z0 = 18 + n * 24;
			drawBoxShell3D(_raw, _w, _h, x0, y0, z0, x0 + 72, y0 + 72, z0 + 18);
		}

		for (uint32_t z = 32; z < _d - 32; z += 64)
			fillBox3D(_raw, _w, _h, 720, 680, z, 950, 690, z + 2);
		for (uint32_t x = 720; x <= 950; x += 46)
			drawLineZ3D(_raw, _w, _h, x, 690, 32, _d - 33);
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

poca::core::MyObjectInterface* createFaceConnectedComponentLargeTestDataset2D()
{
	const uint32_t w = 2000, h = 2000, d = 1;
	std::vector<uint8_t> raw(static_cast<size_t>(w) * h * d, 0);
	addLargeStressStructures2D(raw, w, h);

	poca::core::Image<uint32_t>* labels = makeLabelImage(raw, w, h, d);
	poca::core::Image<uint8_t>* image = makeRawImage(std::move(raw), w, h, d);
	return createObject("Face_connected_components_large_2D_stress_test", image, labels);
}

poca::core::MyObjectInterface* createFaceConnectedComponentLargeTestDataset3D()
{
	const uint32_t w = 1028, h = 1028, d = 512;
	std::vector<uint8_t> raw(static_cast<size_t>(w) * h * d, 0);
	addLargeStressStructures3D(raw, w, h, d);

	poca::core::Image<uint32_t>* labels = makeLabelImage(raw, w, h, d);
	poca::core::Image<uint8_t>* image = makeRawImage(std::move(raw), w, h, d);
	return createObject("Face_connected_components_large_3D_stress_test", image, labels);
}

void registerFaceConnectedComponentTests(poca::core::TestRegistry& _registry)
{
	_registry.add({
		"face_connected_components_2d",
		"face connected components",
		"2D image",
		"Create a 2D face connected components test dataset",
		true,
		createFaceConnectedComponentTestDataset2D
	});
	_registry.add({
		"face_connected_components_3d",
		"face connected components",
		"3D image",
		"Create a 3D face connected components test dataset",
		true,
		createFaceConnectedComponentTestDataset3D
	});
	_registry.add({
		"face_connected_components_large_2d",
		"face connected components",
		"Large 2D image",
		"Create a large 2D face connected components stress test dataset",
		true,
		createFaceConnectedComponentLargeTestDataset2D
	});
	_registry.add({
		"face_connected_components_large_3d",
		"face connected components",
		"Large 3D image",
		"Create a large 3D face connected components stress test dataset",
		true,
		createFaceConnectedComponentLargeTestDataset3D
	});
}
