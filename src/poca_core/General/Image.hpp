/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Image.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*
* PoCA is a free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 3 of the License, or (at your option) any later version.
*
* The algorithms that underlie PoCA have required considerable
* development. They are described in the original SR-Tesseler paper,
* doi:10.1038/nmeth.3579. If you use PoCA as part of work (visualization,
* manipulation, quantification) towards a scientific publication, please include
* a citation to the original paper.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program; if not, write to the Free Software Foundation,
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#ifndef Image_hpp__
#define Image_hpp__

#include <execution>
//#include <tinytiffwriter.h>

#include <Interfaces/ImageInterface.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/MyData.hpp>
#include <General/ArrayStatistics.hpp>
#include <General/Misc.h>
#include <Cuda/CoreMisc.h>
#include <General/Histogram.hpp>

namespace poca::core {

	template <class T>
	class Image : public ImageInterface {
	public:
		Image(const ImageType = RAW);
		~Image();

		BasicComponentInterface* copy();

		void finalizeImage(const uint32_t, const uint32_t, const uint32_t);
		void addFeatureLabels();

		void uint8_normalisedData(std::vector <unsigned char>&) const;
		void uint16_normalisedData(std::vector <uint16_t>&) const;
		void uint16_labeledData(std::vector <uint16_t>&) const;
		void float_normalisedData(std::vector <float>&) const;

		const float* getImage(const uint32_t) const;

		inline const T* data() const;
		inline T* data();
		inline const std::vector<T>& pixels() const;
		inline std::vector<T>& pixels();

		void save(const std::string&) const;

		const void* getImagePtr(const uint32_t) const;

	protected:
	};

	//maxValue is used for shaders. uint8_t & uint16_t textures are normalized so need to know the maxValue to find back the pixel value
	//float texture are not normalized, so we set maxValue at 1 to keep pixel value unchanged in shader
	template <class T>
	Image<T>::Image(const ImageType _typeImage) :ImageInterface(_typeImage)
	{
		m_data.insert(std::make_pair("intensity", new poca::core::MyData(new poca::core::Histogram<T>(), false)));
		std::string type = typeid(T).name();
		m_maxValue = (type == "float" || type == "unsigned int" || type == "int") ? 1 : std::numeric_limits<T>::max();
	}

	template <class T>
	Image<T>::~Image()
	{
	}

	template <class T>
	BasicComponentInterface* Image<T>::copy()
	{
		return new Image(*this);
	}

	template <class T>
	void Image<T>::finalizeImage(const uint32_t _w, const uint32_t _h, const uint32_t _d)
	{
		m_width = _w; m_height = _h; m_depth = _d;
		const std::vector<T>& pixels = this->pixels();
		clock_t t1 = clock(), t2, t3 = clock(), t4;
		t4 = clock();
		long elapsed = ((double)t4 - t3) / CLOCKS_PER_SEC * 1000;
		std::cout << "Time for finding max & min " << elapsed << std::endl;
		t3 = clock();
		m_bbox.set(0, 0, 0, m_width, m_height, m_depth);
		m_data["intensity"]->finalizeData();// = new poca::core::MyData(m_pixels, false);
		t4 = clock();
		elapsed = ((double)t4 - t3) / CLOCKS_PER_SEC * 1000;
		std::cout << "Time for creating my data " << elapsed << std::endl;
		t3 = clock();
		m_selection.clear();// .resize(pixels.size(), true);
		setCurrentHistogramType("intensity");
		m_min = getCurrentHistogram()->getMin();
		m_max = getCurrentHistogram()->getMax();
		t4 = clock();
		elapsed = ((double)t4 - t3) / CLOCKS_PER_SEC * 1000;
		std::cout << "Time forsetting the histogram " << elapsed << std::endl;
		t3 = clock();
		std::cout << "Bounding box image " << m_bbox << std::endl;
		t2 = clock();
		elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
		std::cout << "Time for creating image " << elapsed << std::endl;
		if(m_typeImage == poca::core::LABEL)
			addFeatureLabels();
	}

	template <class T>
	void Image<T>::addFeatureLabels()
	{
		std::cout << __LINE__ << std::endl;
		if (m_volumes.empty()) return;
		std::cout << __LINE__ << std::endl;
		std::vector <float> labels(m_volumes.size());
		std::iota(std::begin(labels), std::end(labels), 1);
		addFeature("label", poca::core::generateDataWithLog(labels));
		addFeature("volume", poca::core::generateDataWithLog(m_volumes));
		setCurrentHistogramType("label");
		std::cout << __LINE__ << std::endl;
	}

	template <class T>
	void Image<T>::uint8_normalisedData(std::vector <unsigned char>& _normalData) const
	{
		// Normalise and cast
		const std::vector<T>& pixels = this->pixels();
		_normalData.clear();
		_normalData.resize(pixels.size());
		float minV = (float)*std::min_element(pixels.begin(), pixels.end()), maxV = (float)*std::max_element(pixels.begin(), pixels.end()), inter = maxV - minV;
#pragma omp parallel for
		for (auto i = 0; i < pixels.size(); ++i) {
			_normalData[i] = static_cast<unsigned char>(255 * ((float)pixels[i] - minV) / inter);
			//_normalData[i] = static_cast<unsigned char>(m_pixels[i]);
		}
	}

	template <class T>
	void Image<T>::uint16_normalisedData(std::vector <uint16_t>& _normalData) const
	{
		const std::vector<T>& pixels = this->pixels();
		// Normalise and cast
		_normalData.clear();
		_normalData.resize(pixels.size());
		float minV = (float)*std::min_element(pixels.begin(), pixels.end()), maxV = (float)*std::max_element(pixels.begin(), pixels.end()), inter = maxV - minV;
#pragma omp parallel for
		for (auto i = 0; i < pixels.size(); ++i) {
			_normalData[i] = static_cast<uint16_t>(65535 * ((float)pixels[i] - minV) / inter);
		}
	}

	template <class T>
	void Image<T>::uint16_labeledData(std::vector <uint16_t>& _data) const
	{
		const std::vector<T>& pixels = this->pixels();
		_data.clear();
		_data.resize(pixels.size());
#pragma omp parallel for
		for (auto i = 0; i < pixels.size(); ++i) {
			_data[i] = static_cast<uint16_t>(pixels[i]);
		}
	}

	template <class T>
	void Image<T>::float_normalisedData(std::vector <float>& _normalData) const
	{
		const std::vector<T>& pixels = this->pixels();
		// Normalise and cast
		_normalData.clear();
		_normalData.resize(pixels.size());
		float minV = (float)*std::min_element(pixels.begin(), pixels.end()), maxV = (float)*std::max_element(pixels.begin(), pixels.end()), inter = maxV - minV;
#pragma omp parallel for
		for (auto i = 0; i < pixels.size(); ++i) {
			_normalData[i] = ((float)pixels[i] - minV) / inter;
			//_normalData[i] = static_cast<unsigned char>(m_pixels[i]);
		}
	}

	template <class T>
	const float* Image<T>::getImage(const uint32_t _index) const
	{
		const std::vector<T>& pixels = this->pixels();
		auto wh = m_width * m_height;
		return pixels.data() + _index * wh;
	}

	template <class T>
	inline const T* Image<T>::data() const
	{
		return pixels().data();
	}

	template <class T>
	inline T* Image<T>::data()
	{
		return pixels().data();
	}

	template <class T>
	inline const std::vector<T>& Image<T>::pixels() const
	{
		return dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"))->getValues();
	}

	template <class T>
	inline std::vector<T>& Image<T>::pixels()
	{
		return dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"))->getValues();
	}

	template <class T>
	void Image<T>::save(const std::string& _filename) const
	{
		/*std::pair <uint16_t, TinyTIFFWriterSampleFormat> infos = getTinyTiffHeaderInfo(m_type);
		TinyTIFFWriterFile* tif = TinyTIFFWriter_open(filename.toStdString().c_str(), infos.first, infos.second, 1, image->width(), image->height(), TinyTIFFWriter_Greyscale);
		if (tif) {
			for (uint32_t frame = 0; frame < image->depth(); frame++) {
				const float* data = image->getImage(frame);
				TinyTIFFWriter_writeImage(tif, data);
			}
			TinyTIFFWriter_close(tif);
			std::cout << "Image " << filename.toStdString() << "saved" << std::endl;
		}*/
	}

	template <class T>
	const void* Image<T>::getImagePtr(const uint32_t _index) const
	{
		const std::vector<T>& pixels = this->pixels();
		auto wh = m_width * m_height;
		return (void *)(pixels.data() + _index * wh);
	}

	typedef Image<uint8_t> ImageU8;
	typedef Image<uint16_t> ImageU16;
	typedef Image<float> ImageF;
}

#endif

