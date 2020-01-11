/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "linemod.hpp"
#include <opencv2/opencv.hpp>

namespace matchHL
{
namespace linemod
{
	const int MAX_FEATURE_NUM = 63;//now only support 63
	template <class Type, class T> struct CmpCoordinate
	{
		CmpCoordinate(Type p)
		{
			this->p = p;
		}
		bool operator()(T b1, T b2)
		{
			Type dis = sqrt(pow(double(b1.x - b2.x), 2.0) + pow(double(b1.y - b2.y), 2.0));
			return dis < p;
		}
		Type p;
	};

// struct Feature

/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
  switch (quantized)
  {
    case 1:   return 0;
    case 2:   return 1;
    case 4:   return 2;
    case 8:   return 3;
    case 16:  return 4;
    case 32:  return 5;
    case 64:  return 6;
    case 128: return 7;
    default:
      return -1; //avoid warning
  }
}

void Feature::read(const cv::FileNode& fn)
{
  cv::FileNodeIterator fni = fn.begin();
  fni >> x >> y >> label;
}

void Feature::write(cv::FileStorage& fs) const
{
  fs << "[:" << x << y << label << "]";
}

// struct Template

/**
 * \brief Crop a set of overlapping templates from different modalities.
 *
 * \param[in,out] templates Set of templates representing the same object view.
 *
 * \return The bounding box of all the templates in original image coordinates.
 */
static cv::Rect cropTemplates(std::vector<Template>& templates)
{
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();

  // First pass: find min/max feature x,y over all pyramid levels and modalities
  for (int i = 0; i < (int)templates.size(); ++i)
  {
    Template& templ = templates[i];

    for (int j = 0; j < (int)templ.features.size(); ++j)
    {
      int x = templ.features[j].x << templ.pyramid_level;
      int y = templ.features[j].y << templ.pyramid_level;
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      max_x = std::max(max_x, x);
      max_y = std::max(max_y, y);
    }
  }

  /// @todo Why require even min_x, min_y?
  if (min_x % 2 == 1) --min_x;
  if (min_y % 2 == 1) --min_y;

  // Second pass: set width/height and shift all feature positions
  for (int i = 0; i < (int)templates.size(); ++i)
  {
    Template& templ = templates[i];
    templ.width = (max_x - min_x) >> templ.pyramid_level;
    templ.height = (max_y - min_y) >> templ.pyramid_level;
	templ.tl_x = min_x >> templ.pyramid_level;
	templ.tl_y = min_y >> templ.pyramid_level;

    for (int j = 0; j < (int)templ.features.size(); ++j)
    {
      templ.features[j].x -= templ.tl_x;
      templ.features[j].y -= templ.tl_y;
    }
  }

  return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

void Template::read(const cv::FileNode& fn)
{
  fn["width"] >> width;
  fn["height"] >> height;
  fn["pyramid_level"] >> pyramid_level;

  fn["scale"] >> scale;
  fn["angle"] >> angle;
  fn["tl_x"] >> tl_x;
  fn["tl_y"] >> tl_y;
  fn["templImg"] >> templImg;
  fn["templMask"] >> templMask;

  cv::FileNode features_fn = fn["features"];
  features.resize(features_fn.size());
  cv::FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
  for (int i = 0; it != it_end; ++it, ++i)
  {
    features[i].read(*it);
  }
}

void Template::write(cv::FileStorage& fs) const
{
  fs << "width" << width;
  fs << "height" << height;
  fs << "pyramid_level" << pyramid_level;

  fs << "scale" << scale;
  fs << "angle" << angle;
  fs << "tl_x" << tl_x;
  fs << "tl_y" << tl_y;
  fs << "templImg" << templImg;
  fs << "templMask" << templMask;

  fs << "features" << "[";
  for (int i = 0; i < (int)features.size(); ++i)
  {
    features[i].write(fs);
  }
  fs << "]"; // features
}

/****************************************************************************************\
*                             Modality interfaces                                        *
\****************************************************************************************/

bool QuantizedPyramid::selectScatteredFeatures(const std::vector<Candidate>& candidates,
                                               std::vector<Feature>& features,
                                               size_t num_features, float distance)
{
	//add by HuangLi 2019/08/07
  features.clear();
  float distance_sq = distance * distance;
  int i = 0;
  while (features.size() < num_features)
  {
    Candidate c = candidates[i];

    // Add if sufficient distance away from any previously chosen feature
    bool keep = true;
    for (int j = 0; (j < (int)features.size()) && keep; ++j)
    {
      Feature f = features[j];
      keep = (c.f.x - f.x)*(c.f.x - f.x) + (c.f.y - f.y)*(c.f.y - f.y) >= distance_sq;
    }
    if (keep)
      features.push_back(c.f);

    if (++i == (int)candidates.size())
    {
      // Start back at beginning, and relax required distance
      i = 0;
      distance -= 1.0f;
      distance_sq = distance * distance;
    }
  }

  if (features.size() == num_features)
  {
	  return true;
  }
  else
  {
	  std::cout << "this templ has no enough features" << std::endl;
	  return false;
  }
}

cv::Ptr<Modality> Modality::create(const cv::String& modality_type)
{
  if (modality_type == "ColorGradient")
    return cv::makePtr<ColorGradient>();
  else if (modality_type == "DepthNormal")
    return cv::makePtr<DepthNormal>();
  else
    return cv::Ptr<Modality>();
}

cv::Ptr<Modality> Modality::create(const cv::FileNode& fn)
{
  cv::String type = fn["type"];
  cv::Ptr<Modality> modality = create(type);
  modality->read(fn);
  return modality;
}

void colormap(const cv::Mat& quantized, cv::Mat& dst)
{
  std::vector<cv::Vec3b> lut(8);
  lut[0] = cv::Vec3b(  0,   0, 255);
  lut[1] = cv::Vec3b(  0, 170, 255);
  lut[2] = cv::Vec3b(  0, 255, 170);
  lut[3] = cv::Vec3b(  0, 255,   0);
  lut[4] = cv::Vec3b(170, 255,   0);
  lut[5] = cv::Vec3b(255, 170,   0);
  lut[6] = cv::Vec3b(255,   0,   0);
  lut[7] = cv::Vec3b(255,   0, 170);

  dst = cv::Mat::zeros(quantized.size(), CV_8UC3);
  for (int r = 0; r < dst.rows; ++r)
  {
    const uchar* quant_r = quantized.ptr(r);
    cv::Vec3b* dst_r = dst.ptr<cv::Vec3b>(r);
    for (int c = 0; c < dst.cols; ++c)
    {
      uchar q = quant_r[c];
      if (q)
        dst_r[c] = lut[getLabel(q)];
    }
  }
}

/****************************************************************************************\
*                             Color gradient modality                                    *
\****************************************************************************************/

// Forward declaration
//void hysteresisGradient(cv::Mat& magnitude, cv::Mat& angle,
//                        cv::Mat& ap_tmp, float threshold);

/**
 * \brief Compute quantized orientation image from color image.
 *
 * Implements section 2.2 "Computing the Gradient Orientations."
 *
 * \param[in]  src       The source 8-bit, 3-channel image.
 * \param[out] magnitude Destination floating-point array of squared magnitudes.
 * \param[out] angle     Destination 8-bit array of orientations. Each bit
 *                       represents one bin of the orientation space.
 * \param      threshold Magnitude threshold. Keep only gradients whose norms are
 *                       larger than this.
 */
void hysteresisGradient(cv::Mat& magnitude, cv::Mat& quantized_angle,
	cv::Mat& angle, float threshold)
{
	// Quantize 360 degree range of orientations into 16 buckets
	// Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
	// for stability of horizontal and vertical features.
	cv::Mat_<unsigned char> quantized_unfiltered;
	angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

	// Zero out top and bottom rows
	/// @todo is this necessary, or even correct?
	memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
	memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
	// Zero out first and last columns
	for (int r = 0; r < quantized_unfiltered.rows; ++r)
	{
		quantized_unfiltered(r, 0) = 0;
		quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
	}

	// Mask 16 buckets into 8 quantized orientations
	for (int r = 1; r < angle.rows - 1; ++r)
	{
		uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
		for (int c = 1; c < angle.cols - 1; ++c)
		{
			quant_r[c] &= 7;
		}
	}

	// Filter the raw quantized image. Only accept pixels where the magnitude is above some
	// threshold, and there is local agreement on the quantization.
	quantized_angle = cv::Mat::zeros(angle.size(), CV_8U);
	for (int r = 1; r < angle.rows - 1; ++r)
	{
		float* mag_r = magnitude.ptr<float>(r);

		for (int c = 1; c < angle.cols - 1; ++c)
		{
			if (mag_r[c] > threshold)
			{
				// Compute histogram of quantized bins in 3x3 patch around pixel
				int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

				uchar* patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;

				patch3x3_row += quantized_unfiltered.step1();
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;

				patch3x3_row += quantized_unfiltered.step1();
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;

				// Find bin with the most votes from the patch
				int max_votes = 0;
				int index = -1;
				for (int i = 0; i < 8; ++i)
				{
					if (max_votes < histogram[i])
					{
						index = i;
						max_votes = histogram[i];
					}
				}

				// Only accept the quantization if majority of pixels in the patch agree
				static const int NEIGHBOR_THRESHOLD = 5;
				if (max_votes >= NEIGHBOR_THRESHOLD)
					quantized_angle.at<uchar>(r, c) = uchar(1 << index);
			}
		}
	}
}

static void quantizedOrientations(const cv::Mat& src, cv::Mat& magnitude,
                           cv::Mat& angle, float threshold)
{ 
	//add by HuangLi 2019/08/07
	cv::Mat smoothed;
	// Compute horizontal and vertical image derivatives on all color channels separately
	static const int KERNEL_SIZE = 7;
	// For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
	GaussianBlur(src, smoothed, cv::Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, cv::BORDER_REPLICATE);

	if (src.channels() == 1)
	{
		cv::Mat sobel_dx, sobel_dy, sobel_ag;
		Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
		Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
		magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
		phase(sobel_dx, sobel_dy, sobel_ag, true);
		hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
	}
	else
	{
		magnitude.create(src.size(), CV_32F);

		// Allocate temporary buffers
		cv::Size size = src.size();
		cv::Mat sobel_3dx; // per-channel horizontal derivative
		cv::Mat sobel_3dy; // per-channel vertical derivative
		cv::Mat sobel_dx(size, CV_32F);      // maximum horizontal derivative
		cv::Mat sobel_dy(size, CV_32F);      // maximum vertical derivative
		cv::Mat sobel_ag;  // final gradient orientation (unquantized)

		Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
		Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

		short * ptrx = (short *)sobel_3dx.data;
		short * ptry = (short *)sobel_3dy.data;
		float * ptr0x = (float *)sobel_dx.data;
		float * ptr0y = (float *)sobel_dy.data;
		float * ptrmg = (float *)magnitude.data;

		const int length1 = static_cast<const int>(sobel_3dx.step1());
		const int length2 = static_cast<const int>(sobel_3dy.step1());
		const int length3 = static_cast<const int>(sobel_dx.step1());
		const int length4 = static_cast<const int>(sobel_dy.step1());
		const int length5 = static_cast<const int>(magnitude.step1());
		const int length0 = sobel_3dy.cols * 3;

		for (int r = 0; r < sobel_3dy.rows; ++r)
		{
			int ind = 0;

			for (int i = 0; i < length0; i += 3)
			{
				// Use the gradient orientation of the channel whose magnitude is largest
				int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
				int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
				int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

				if (mag1 >= mag2 && mag1 >= mag3)
				{
					ptr0x[ind] = ptrx[i];
					ptr0y[ind] = ptry[i];
					ptrmg[ind] = (float)mag1;
				}
				else if (mag2 >= mag1 && mag2 >= mag3)
				{
					ptr0x[ind] = ptrx[i + 1];
					ptr0y[ind] = ptry[i + 1];
					ptrmg[ind] = (float)mag2;
				}
				else
				{
					ptr0x[ind] = ptrx[i + 2];
					ptr0y[ind] = ptry[i + 2];
					ptrmg[ind] = (float)mag3;
				}
				++ind;
			}
			ptrx += length1;
			ptry += length2;
			ptr0x += length3;
			ptr0y += length4;
			ptrmg += length5;
		}

		// Calculate the final gradient orientations
		phase(sobel_dx, sobel_dy, sobel_ag, true);
		hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
	}

}

class ColorGradientPyramid : public QuantizedPyramid
{
public:
  ColorGradientPyramid(const cv::Mat& src, const cv::Mat& mask,
                       float weak_threshold, size_t num_features,
                       float strong_threshold);

  virtual void quantize(cv::Mat& dst) const;

  virtual bool extractTemplate(Template& templ) const;

  virtual void pyrDown();

protected:
  /// Recalculate angle and magnitude images
  void update();

  cv::Mat src;
  cv::Mat mask;

  int pyramid_level;
  cv::Mat angle;
  cv::Mat magnitude;

  float weak_threshold;
  size_t num_features;
  float strong_threshold;
};

ColorGradientPyramid::ColorGradientPyramid(const cv::Mat& _src, const cv::Mat& _mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold)
  : src(_src),
    mask(_mask),
    pyramid_level(0),
    weak_threshold(_weak_threshold),
    num_features(_num_features),
    strong_threshold(_strong_threshold)
{
  update();
}

void ColorGradientPyramid::update()
{
  quantizedOrientations(src, magnitude, angle, weak_threshold);
}

void ColorGradientPyramid::pyrDown()
{
  // Some parameters need to be adjusted
  num_features /= 2; /// @todo Why not 4?
  ++pyramid_level;

  // Downsample the current inputs
  cv::Size size(src.cols / 2, src.rows / 2);
  cv::Mat next_src;
  cv::pyrDown(src, next_src, size);
  src = next_src;
  if (!mask.empty())
  {
    cv::Mat next_mask;
    resize(mask, next_mask, size, 0.0, 0.0, cv::INTER_NEAREST);
    mask = next_mask;
  }

  update();
}

void ColorGradientPyramid::quantize(cv::Mat& dst) const
{
  dst = cv::Mat::zeros(angle.size(), CV_8U);
  angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template& templ) const
{
	if (1)
	{
		// Want features on the border to distinguish from background
		cv::Mat local_mask;
		if (!mask.empty())
		{
			erode(mask, local_mask, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
			//subtract(mask, local_mask, local_mask);
		}
		double time = cv::getTickCount();
		// Create sorted list of all pixels with magnitude greater than a threshold
		std::vector<Candidate> candidates;
		bool no_mask = local_mask.empty();
		float threshold_sq = strong_threshold*strong_threshold;
		for (int r = 0; r < magnitude.rows; ++r)
		{
			const uchar* angle_r = angle.ptr<uchar>(r);
			const float* magnitude_r = magnitude.ptr<float>(r);
			const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

			for (int c = 0; c < magnitude.cols; ++c)
			{
				if (no_mask || mask_r[c])
				{
					uchar quantized = angle_r[c];
					if (quantized > 0)
					{
						float score = magnitude_r[c];
						if (score > threshold_sq)
						{
							candidates.push_back(Candidate(c, r, getLabel(quantized), score));
						}
					}
				}
			}
		}
		// We require a certain number of features
		if (candidates.size() < num_features)
			return false;
		// NOTE: Stable sort to agree with old code, which used std::list::sort()
		std::stable_sort(candidates.begin(), candidates.end());
		time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
		std::cout << "stable_sort cost: " << time << std::endl;


		// Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
		float distance = static_cast<float>(candidates.size() / num_features + 1);
		//add by HuangLi 2019/08/07
		if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
		{
			return false;
		}

		// cv::Size determined externally, needs to match templates for other modalities
		templ.width = -1;
		templ.height = -1;
		templ.pyramid_level = pyramid_level;

		return true;
	}
	else//add nms for template features selection.  Turn on causing feature point reduction
	{
		// Want features on the border to distinguish from background
		cv::Mat local_mask;
		if (!mask.empty())
		{
			erode(mask, local_mask, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
			//        subtract(mask, local_mask, local_mask);
		}

		std::vector<Candidate> candidates;
		bool no_mask = local_mask.empty();
		float threshold_sq = strong_threshold * strong_threshold;

		int nms_kernel_size = 5;
		cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

		for (int r = 0 + nms_kernel_size / 2; r < magnitude.rows - nms_kernel_size / 2; ++r)
		{
			const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

			for (int c = 0 + nms_kernel_size / 2; c < magnitude.cols - nms_kernel_size / 2; ++c)
			{
				if (no_mask || mask_r[c])
				{
					float score = 0;
					if (magnitude_valid.at<uchar>(r, c)>0){
						score = magnitude.at<float>(r, c);
						bool is_max = true;
						for (int r_offset = -nms_kernel_size / 2; r_offset <= nms_kernel_size / 2; r_offset++){
							for (int c_offset = -nms_kernel_size / 2; c_offset <= nms_kernel_size / 2; c_offset++){
								if (r_offset == 0 && c_offset == 0) continue;

								if (score < magnitude.at<float>(r + r_offset, c + c_offset)){
									score = 0;
									is_max = false;
									break;
								}
							}
						}

						if (is_max){
							for (int r_offset = -nms_kernel_size / 2; r_offset <= nms_kernel_size / 2; r_offset++){
								for (int c_offset = -nms_kernel_size / 2; c_offset <= nms_kernel_size / 2; c_offset++){
									if (r_offset == 0 && c_offset == 0) continue;
									magnitude_valid.at<uchar>(r + r_offset, c + c_offset) = 0;
								}
							}
						}
					}

					if (score > threshold_sq && angle.at<uchar>(r, c) > 0)
					{
						candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
					}
				}
			}
		}
		// We require a certain number of features
		if (candidates.size() < num_features)
			return false;
		// NOTE: Stable sort to agree with old code, which used std::list::sort()
		std::stable_sort(candidates.begin(), candidates.end());

		// Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
		float distance = static_cast<float>(candidates.size() / num_features + 1);
		if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
		{
			return false;
		}

		// Size determined externally, needs to match templates for other modalities
		templ.width = -1;
		templ.height = -1;
		templ.pyramid_level = pyramid_level;

		return true;
	}
}

//ColorGradient::ColorGradient()
//  : weak_threshold(10.0f),
//    num_features(63),
//    strong_threshold(55.0f)
//{
//}

ColorGradient::ColorGradient()
  : weak_threshold(10.0f),
    num_features(63),
    strong_threshold(30.0f)
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
  : weak_threshold(_weak_threshold),
    num_features(_num_features),
    strong_threshold(_strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";

cv::String ColorGradient::name() const
{
  return CG_NAME;
}

cv::Ptr<QuantizedPyramid> ColorGradient::processImpl(const cv::Mat& src,
                                                     const cv::Mat& mask) const
{
  return cv::makePtr<ColorGradientPyramid>(src, mask, weak_threshold, num_features, strong_threshold);
}

void ColorGradient::read(const cv::FileNode& fn)
{
  cv::String type = fn["type"];
  CV_Assert(type == CG_NAME);

  weak_threshold = fn["weak_threshold"];
  num_features = int(fn["num_features"]);
  strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(cv::FileStorage& fs) const
{
  fs << "type" << CG_NAME;
  fs << "weak_threshold" << weak_threshold;
  fs << "num_features" << int(num_features);
  fs << "strong_threshold" << strong_threshold;
}

/****************************************************************************************\
*                               Depth normal modality                                    *
\****************************************************************************************/

// Contains GRANULARITY and NORMAL_LUT
#include "normal_lut.i"

static void accumBilateral(long delta, long i, long j, long * A, long * b, int threshold)
{
  long f = std::abs(delta) < threshold ? 1 : 0;

  const long fi = f * i;
  const long fj = f * j;

  A[0] += fi * i;
  A[1] += fi * j;
  A[3] += fj * j;
  b[0]  += fi * delta;
  b[1]  += fj * delta;
}

/**
 * \brief Compute quantized normal image from depth image.
 *
 * Implements section 2.6 "Extension to Dense Depth Sensors."
 *
 * \param[in]  src  The source 16-bit depth image (in mm).
 * \param[out] dst  The destination 8-bit image. Each bit represents one bin of
 *                  the view cone.
 * \param distance_threshold   Ignore pixels beyond this distance.
 * \param difference_threshold When computing normals, ignore contributions of pixels whose
 *                             depth difference with the central pixel is above this threshold.
 *
 * \todo Should also need camera model, or at least focal lengths? Replace distance_threshold with mask?
 */
static void quantizedNormals(const cv::Mat& src, cv::Mat& dst, int distance_threshold,
                      int difference_threshold)
{
  dst = cv::Mat::zeros(src.size(), CV_8U);

  const unsigned short * lp_depth   = src.ptr<ushort>();
  unsigned char  * lp_normals = dst.ptr<uchar>();

  const int l_W = src.cols;
  const int l_H = src.rows;

  const int l_r = 5; // used to be 7
  const int l_offset0 = -l_r - l_r * l_W;
  const int l_offset1 =    0 - l_r * l_W;
  const int l_offset2 = +l_r - l_r * l_W;
  const int l_offset3 = -l_r;
  const int l_offset4 = +l_r;
  const int l_offset5 = -l_r + l_r * l_W;
  const int l_offset6 =    0 + l_r * l_W;
  const int l_offset7 = +l_r + l_r * l_W;

  const int l_offsetx = GRANULARITY / 2;
  const int l_offsety = GRANULARITY / 2;

  for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y)
  {
    const unsigned short * lp_line = lp_depth + (l_y * l_W + l_r);
    unsigned char * lp_norm = lp_normals + (l_y * l_W + l_r);

    for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x)
    {
      long l_d = lp_line[0];

      if (l_d < distance_threshold)
      {
        // accum
        long l_A[4]; l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
        long l_b[2]; l_b[0] = l_b[1] = 0;
        accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset1] - l_d,    0, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset3] - l_d, -l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset4] - l_d, +l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset6] - l_d,    0, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b, difference_threshold);

        // solve
        long l_det =  l_A[0] * l_A[3] - l_A[1] * l_A[1];
        long l_ddx =  l_A[3] * l_b[0] - l_A[1] * l_b[1];
        long l_ddy = -l_A[1] * l_b[0] + l_A[0] * l_b[1];

        /// @todo Magic number 1150 is focal length? This is something like
        /// f in SXGA mode, but in VGA is more like 530.
        float l_nx = static_cast<float>(1150 * l_ddx);
        float l_ny = static_cast<float>(1150 * l_ddy);
        float l_nz = static_cast<float>(-l_det * l_d);

        float l_sqrt = sqrtf(l_nx * l_nx + l_ny * l_ny + l_nz * l_nz);

        if (l_sqrt > 0)
        {
          float l_norminv = 1.0f / (l_sqrt);

          l_nx *= l_norminv;
          l_ny *= l_norminv;
          l_nz *= l_norminv;

          //*lp_norm = fabs(l_nz)*255;

          int l_val1 = static_cast<int>(l_nx * l_offsetx + l_offsetx);
          int l_val2 = static_cast<int>(l_ny * l_offsety + l_offsety);
          int l_val3 = static_cast<int>(l_nz * GRANULARITY + GRANULARITY);

          *lp_norm = NORMAL_LUT[l_val3][l_val2][l_val1];
        }
        else
        {
          *lp_norm = 0; // Discard shadows from depth sensor
        }
      }
      else
      {
        *lp_norm = 0; //out of depth
      }
      ++lp_line;
      ++lp_norm;
    }
  }
  medianBlur(dst, dst, 5);
}

class DepthNormalPyramid : public QuantizedPyramid
{
public:
  DepthNormalPyramid(const cv::Mat& src, const cv::Mat& mask,
                     int distance_threshold, int difference_threshold, size_t num_features,
                     int extract_threshold);

  virtual void quantize(cv::Mat& dst) const;

  virtual bool extractTemplate(Template& templ) const;

  virtual void pyrDown();

protected:
  cv::Mat mask;

  int pyramid_level;
  cv::Mat normal;

  size_t num_features;
  int extract_threshold;
};

DepthNormalPyramid::DepthNormalPyramid(const cv::Mat& src, const cv::Mat& _mask,
                                       int distance_threshold, int difference_threshold, size_t _num_features,
                                       int _extract_threshold)
  : mask(_mask),
    pyramid_level(0),
    num_features(_num_features),
    extract_threshold(_extract_threshold)
{
  quantizedNormals(src, normal, distance_threshold, difference_threshold);
}

void DepthNormalPyramid::pyrDown()
{
  // Some parameters need to be adjusted
  num_features /= 2; /// @todo Why not 4?
  extract_threshold /= 2;
  ++pyramid_level;

  // In this case, NN-downsample the quantized image
  cv::Mat next_normal;
  cv::Size size(normal.cols / 2, normal.rows / 2);
  resize(normal, next_normal, size, 0.0, 0.0, cv::INTER_NEAREST);
  normal = next_normal;
  if (!mask.empty())
  {
    cv::Mat next_mask;
    resize(mask, next_mask, size, 0.0, 0.0, cv::INTER_NEAREST);
    mask = next_mask;
  }
}

void DepthNormalPyramid::quantize(cv::Mat& dst) const
{
  dst = cv::Mat::zeros(normal.size(), CV_8U);
  normal.copyTo(dst, mask);
}

bool DepthNormalPyramid::extractTemplate(Template& templ) const
{
  // Features right on the object border are unreliable
  cv::Mat local_mask;
  if (!mask.empty())
  {
    erode(mask, local_mask, cv::Mat(), cv::Point(-1,-1), 2, cv::BORDER_REPLICATE);
  }

  // Compute distance transform for each individual quantized orientation
  cv::Mat temp = cv::Mat::zeros(normal.size(), CV_8U);
  cv::Mat distances[8];
  for (int i = 0; i < 8; ++i)
  {
    temp.setTo(1 << i, local_mask);
    bitwise_and(temp, normal, temp);
    // temp is now non-zero at pixels in the mask with quantized orientation i
    distanceTransform(temp, distances[i], cv::DIST_C, 3);
  }

  // Count how many features taken for each label
  int label_counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  // Create sorted list of candidate features
  std::vector<Candidate> candidates;
  bool no_mask = local_mask.empty();
  for (int r = 0; r < normal.rows; ++r)
  {
    const uchar* normal_r = normal.ptr<uchar>(r);
    const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

    for (int c = 0; c < normal.cols; ++c)
    {
      if (no_mask || mask_r[c])
      {
        uchar quantized = normal_r[c];

        if (quantized != 0 && quantized != 255) // background and shadow
        {
          int label = getLabel(quantized);

          // Accept if distance to a pixel belonging to a different label is greater than
          // some threshold. IOW, ideal feature is in the center of a large homogeneous
          // region.
          float score = distances[label].at<float>(r, c);
          if (score >= extract_threshold)
          {
            candidates.push_back( Candidate(c, r, label, score) );
            ++label_counts[label];
          }
        }
      }
    }
  }
  // We require a certain number of features
  if (candidates.size() < num_features)
    return false;

  // Prefer large distances, but also want to collect features over all 8 labels.
  // So penalize labels with lots of candidates.
  for (size_t i = 0; i < candidates.size(); ++i)
  {
    Candidate& c = candidates[i];
    c.score /= (float)label_counts[c.f.label];
  }
  std::stable_sort(candidates.begin(), candidates.end());

  // Use heuristic based on object area for initial distance threshold
  float area = no_mask ? (float)normal.total() : (float)countNonZero(local_mask);
  float distance = sqrtf(area) / sqrtf((float)num_features) + 1.5f;
  selectScatteredFeatures(candidates, templ.features, num_features, distance);

  // cv::Size determined externally, needs to match templates for other modalities
  templ.width = -1;
  templ.height = -1;
  templ.pyramid_level = pyramid_level;

  return true;
}

DepthNormal::DepthNormal()
  : distance_threshold(2000),
    difference_threshold(50),
    num_features(63),
    extract_threshold(2)
{
}

DepthNormal::DepthNormal(int _distance_threshold, int _difference_threshold, size_t _num_features,
                         int _extract_threshold)
  : distance_threshold(_distance_threshold),
    difference_threshold(_difference_threshold),
    num_features(_num_features),
    extract_threshold(_extract_threshold)
{
}

static const char DN_NAME[] = "DepthNormal";

cv::String DepthNormal::name() const
{
  return DN_NAME;
}

cv::Ptr<QuantizedPyramid> DepthNormal::processImpl(const cv::Mat& src,
                                                   const cv::Mat& mask) const
{
  return cv::makePtr<DepthNormalPyramid>(src, mask, distance_threshold, difference_threshold,
                                     num_features, extract_threshold);
}

void DepthNormal::read(const cv::FileNode& fn)
{
  cv::String type = fn["type"];
  CV_Assert(type == DN_NAME);

  distance_threshold = fn["distance_threshold"];
  difference_threshold = fn["difference_threshold"];
  num_features = int(fn["num_features"]);
  extract_threshold = fn["extract_threshold"];
}

void DepthNormal::write(cv::FileStorage& fs) const
{
  fs << "type" << DN_NAME;
  fs << "distance_threshold" << distance_threshold;
  fs << "difference_threshold" << difference_threshold;
  fs << "num_features" << int(num_features);
  fs << "extract_threshold" << extract_threshold;
}

/****************************************************************************************\
*                                 Response maps                                          *
\****************************************************************************************/

//static void orUnaligned8u(const uchar * src, const int src_stride,
//                   uchar * dst, const int dst_stride,
//                   const int width, const int height)
//{
//#if CV_SSE2
//  volatile bool haveSSE2 = cv::checkHardwareSupport(CPU_SSE2);
//#if CV_SSE3
//  volatile bool haveSSE3 = cv::checkHardwareSupport(CPU_SSE3);
//#endif
//  bool src_aligned = reinterpret_cast<unsigned long long>(src) % 16 == 0;
//#endif
//
//  for (int r = 0; r < height; ++r)
//  {
//    int c = 0;
//
//#if CV_SSE2
//    // Use aligned loads if possible
//    if (haveSSE2 && src_aligned)
//    {
//      for ( ; c < width - 15; c += 16)
//      {
//        const __m128i* src_ptr = reinterpret_cast<const __m128i*>(src + c);
//        __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
//        *dst_ptr = _mm_or_si128(*dst_ptr, *src_ptr);
//      }
//    }
//#if CV_SSE3
//    // Use LDDQU for fast unaligned load
//    else if (haveSSE3)
//    {
//      for ( ; c < width - 15; c += 16)
//      {
//        __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(src + c));
//        __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
//        *dst_ptr = _mm_or_si128(*dst_ptr, val);
//      }
//    }
//#endif
//    // Fall back to MOVDQU
//    else if (haveSSE2)
//    {
//      for ( ; c < width - 15; c += 16)
//      {
//        __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + c));
//        __m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
//        *dst_ptr = _mm_or_si128(*dst_ptr, val);
//      }
//    }
//#endif
//    for ( ; c < width; ++c)
//      dst[c] |= src[c];
//
//    // Advance to next row
//    src += src_stride;
//    dst += dst_stride;
//  }
//}

static void orUnaligned8u(const uchar * src, const int src_stride,
	uchar * dst, const int dst_stride,
	const int width, const int height)
{
	for (int r = 0; r < height; ++r)
	{
		int c = 0;
		for (; c < width; ++c)
			dst[c] |= src[c];

		// Advance to next row
		src += src_stride;
		dst += dst_stride;
	}
}

/**
 * \brief Spread binary labels in a quantized image.
 *
 * Implements section 2.3 "Spreading the Orientations."
 *
 * \param[in]  src The source 8-bit quantized image.
 * \param[out] dst Destination 8-bit spread image.
 * \param      T   Sampling step. Spread labels T/2 pixels in each direction.
 */
static void spread(const cv::Mat& src, cv::Mat& dst, int T)
{
  // Allocate and zero-initialize spread (OR'ed) image
  dst = cv::Mat::zeros(src.size(), CV_8U);

  // Fill in spread gradient image (section 2.3)
  for (int r = 0; r < T; ++r)
  {
    int height = src.rows - r;
    for (int c = 0; c < T; ++c)
    {
      orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(), static_cast<const int>(dst.step1()), src.cols - c, height);
    }
  }
}

// Auto-generated by create_similarity_lut.py
CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};

/**
 * \brief Precompute response maps for a spread quantized image.
 *
 * Implements section 2.4 "Precomputing Response Maps."
 *
 * \param[in]  src           The source 8-bit spread quantized image.
 * \param[out] response_maps cv::Vector of 8 response maps, one for each bit label.
 */
static void computeResponseMaps(const cv::Mat& src, std::vector<cv::Mat>& response_maps)
{
  CV_Assert((src.rows * src.cols) % 16 == 0);

  // Allocate response maps
  response_maps.resize(8);
  for (int i = 0; i < 8; ++i)
    response_maps[i].create(src.size(), CV_8U);

  cv::Mat lsb4(src.size(), CV_8U);
  cv::Mat msb4(src.size(), CV_8U);

  for (int r = 0; r < src.rows; ++r)
  {
    const uchar* src_r = src.ptr(r);
    uchar* lsb4_r = lsb4.ptr(r);
    uchar* msb4_r = msb4.ptr(r);

    for (int c = 0; c < src.cols; ++c)
    {
      // Least significant 4 bits of spread image pixel
      lsb4_r[c] = src_r[c] & 15;
      // Most significant 4 bits, right-shifted to be in [0, 16)
      msb4_r[c] = (src_r[c] & 240) >> 4;
    }
  }

#if CV_SSSE3
  volatile bool haveSSSE3 = cv::checkHardwareSupport(CV_CPU_SSSE3);
  if (haveSSSE3)
  {
    const __m128i* lut = reinterpret_cast<const __m128i*>(SIMILARITY_LUT);
    for (int ori = 0; ori < 8; ++ori)
    {
      __m128i* map_data = response_maps[ori].ptr<__m128i>();
      __m128i* lsb4_data = lsb4.ptr<__m128i>();
      __m128i* msb4_data = msb4.ptr<__m128i>();

      // Precompute the 2D response map S_i (section 2.4)
      for (int i = 0; i < (src.rows * src.cols) / 16; ++i)
      {
        // Using SSE shuffle for table lookup on 4 orientations at a time
        // The most/least significant 4 bits are used as the LUT index
        __m128i res1 = _mm_shuffle_epi8(lut[2*ori + 0], lsb4_data[i]);
        __m128i res2 = _mm_shuffle_epi8(lut[2*ori + 1], msb4_data[i]);

        // Combine the results into a single similarity score
        map_data[i] = _mm_max_epu8(res1, res2);
      }
    }
  }
  else
#endif
  {
    // For each of the 8 quantized orientations...
    for (int ori = 0; ori < 8; ++ori)
    {
      uchar* map_data = response_maps[ori].ptr<uchar>();
      uchar* lsb4_data = lsb4.ptr<uchar>();
      uchar* msb4_data = msb4.ptr<uchar>();
      const uchar* lut_low = SIMILARITY_LUT + 32*ori;
      const uchar* lut_hi = lut_low + 16;

      for (int i = 0; i < src.rows * src.cols; ++i)
      {
        map_data[i] = std::max(lut_low[ lsb4_data[i] ], lut_hi[ msb4_data[i] ]);
      }
    }
  }
}

/**
 * \brief Convert a response map to fast linearized ordering.
 *
 * Implements section 2.5 "Linearizing the Memory for Parallelization."
 *
 * \param[in]  response_map The 2D response map, an 8-bit image.
 * \param[out] linearized   The response map in linearized order. It has T*T rows,
 *                          each of which is a linear memory of length (W/T)*(H/T).
 * \param      T            Sampling step.
 */
static void linearize(const cv::Mat& response_map, cv::Mat& linearized, int T)
{
  CV_Assert(response_map.rows % T == 0);
  CV_Assert(response_map.cols % T == 0);

  // linearized has T^2 rows, where each row is a linear memory
  int mem_width = response_map.cols / T;
  int mem_height = response_map.rows / T;
  linearized.create(T*T, mem_width * mem_height, CV_8U);

  // Outer two for loops iterate over top-left T^2 starting pixels
  int index = 0;
  for (int r_start = 0; r_start < T; ++r_start)
  {
    for (int c_start = 0; c_start < T; ++c_start)
    {
      uchar* memory = linearized.ptr(index);
      ++index;

      // Inner two loops copy every T-th pixel into the linear memory
      for (int r = r_start; r < response_map.rows; r += T)
      {
        const uchar* response_data = response_map.ptr(r);
        for (int c = c_start; c < response_map.cols; c += T)
          *memory++ = response_data[c];
      }
    }
  }
}

/****************************************************************************************\
*                               Linearized similarities                                  *
\****************************************************************************************/

static const unsigned char* accessLinearMemory(const std::vector<cv::Mat>& linear_memories,
          const Feature& f, int T, int W)
{
  // Retrieve the TxT grid of linear memories associated with the feature label
  const cv::Mat& memory_grid = linear_memories[f.label];
  CV_DbgAssert(memory_grid.rows == T*T);
  CV_DbgAssert(f.x >= 0);
  CV_DbgAssert(f.y >= 0);
  // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
  int grid_x = f.x % T;
  int grid_y = f.y % T;
  int grid_index = grid_y * T + grid_x;
  CV_DbgAssert(grid_index >= 0);
  CV_DbgAssert(grid_index < memory_grid.rows);
  const unsigned char* memory = memory_grid.ptr(grid_index);
  // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
  // input image width decimated by T.
  int lm_x = f.x / T;
  int lm_y = f.y / T;
  int lm_index = lm_y * W + lm_x;
  CV_DbgAssert(lm_index >= 0);
  CV_DbgAssert(lm_index < memory_grid.cols);
  return memory + lm_index;
}

/**
 * \brief Compute similarity measure for a given template at each sampled image location.
 *
 * Uses linear memories to compute the similarity measure as described in Fig. 7.
 *
 * \param[in]  linear_memories cv::Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image of size (W/T, H/T).
 * \param      size            cv::Size (W, H) of the original input image.
 * \param      T               Sampling step.
 */
static void similarity(const std::vector<cv::Mat>& linear_memories, const Template& templ,
                cv::Mat& dst, cv::Size size, int T)
{
  // 63 features or less is a special case because the max similarity per-feature is 4.
  // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
  // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
  // general function would use _mm_add_epi16.
	CV_Assert(templ.features.size() <= MAX_FEATURE_NUM);
  /// @todo Handle more than 255/MAX_RESPONSE features!!

  // Decimate input image size by factor of T
  int W = size.width / T;
  int H = size.height / T;

  // Feature dimensions, decimated by factor T and rounded up
  int wf = (templ.width - 1) / T + 1;
  int hf = (templ.height - 1) / T + 1;

  // Span is the range over which we can shift the template around the input image
  int span_x = W - wf;
  int span_y = H - hf;

  // Compute number of contiguous (in memory) pixels to check when sliding feature over
  // image. This allows template to wrap around left/right border incorrectly, so any
  // wrapped template matches must be filtered out!
  int template_positions = span_y * W + span_x + 1; // why add 1?
  //int template_positions = (span_y - 1) * W + span_x; // More correct?

  /// @todo In old code, dst is buffer of size m_U. Could make it something like
  /// (span_x)x(span_y) instead?
  dst = cv::Mat::zeros(H, W, CV_8U);
  uchar* dst_ptr = dst.ptr<uchar>();

#if CV_SSE2
  volatile bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
  volatile bool haveSSE3 = cv::checkHardwareSupport(CV_CPU_SSE3);
#endif
#endif

  // Compute the similarity measure for this template by accumulating the contribution of
  // each feature
  for (int i = 0; i < (int)templ.features.size(); ++i)
  {
    // Add the linear memory at the appropriate offset computed from the location of
    // the feature in the template
    Feature f = templ.features[i];
    // Discard feature if out of bounds
    /// @todo Shouldn't actually see x or y < 0 here?
    if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
      continue;
    const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

    // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
    int j = 0;
    // Process responses 16 at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
    if (haveSSE3)
    {
      // LDDQU may be more efficient than MOVDQU for unaligned load of next 16 responses
      for ( ; j < template_positions - 15; j += 16)
      {
        __m128i responses = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
        __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
        *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
      }
    }
    else
#endif
    if (haveSSE2)
    {
      // Fall back to MOVDQU
      for ( ; j < template_positions - 15; j += 16)
      {
        __m128i responses = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
        __m128i* dst_ptr_sse = reinterpret_cast<__m128i*>(dst_ptr + j);
        *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
      }
    }
#endif
    for ( ; j < template_positions; ++j)
      dst_ptr[j] = uchar(dst_ptr[j] + lm_ptr[j]);
  }
}

/**
 * \brief Compute similarity measure for a given template in a local region.
 *
 * \param[in]  linear_memories cv::Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image, 16x16.
 * \param      size            cv::Size (W, H) of the original input image.
 * \param      T               Sampling step.
 * \param      center          Center of the local region.
 */
static void similarityLocal(const std::vector<cv::Mat>& linear_memories, const Template& templ,
                     cv::Mat& dst, cv::Size size, int T, cv::Point center)
{
  // Similar to whole-image similarity() above. This version takes a position 'center'
  // and computes the energy in the 16x16 patch centered on it.
	CV_Assert(templ.features.size() <= MAX_FEATURE_NUM);

  // Compute the similarity map in a 16x16 patch around center
  int W = size.width / T;
  dst = cv::Mat::zeros(16, 16, CV_8U);

  // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
  // center to get the top-left corner of the 16x16 patch.
  // NOTE: We make the offsets multiples of T to agree with results of the original code.
  int offset_x = (center.x / T - 8) * T;
  int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
  volatile bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
  volatile bool haveSSE3 = cv::checkHardwareSupport(CV_CPU_SSE3);
#endif
  __m128i* dst_ptr_sse = dst.ptr<__m128i>();
#endif

  for (int i = 0; i < (int)templ.features.size(); ++i)
  {
    Feature f = templ.features[i];
    f.x += offset_x;
    f.y += offset_y;
    // Discard feature if out of bounds, possibly due to applying the offset
    if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
      continue;

    const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

    // Process whole row at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
    if (haveSSE3)
    {
      // LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
      for (int row = 0; row < 16; ++row)
      {
        __m128i aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
        dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
        lm_ptr += W; // Step to next row
      }
    }
    else
#endif
    if (haveSSE2)
    {
      // Fall back to MOVDQU
      for (int row = 0; row < 16; ++row)
      {
        __m128i aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
        dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
        lm_ptr += W; // Step to next row
      }
    }
    else
#endif
    {
      uchar* dst_ptr = dst.ptr<uchar>();
      for (int row = 0; row < 16; ++row)
      {
        for (int col = 0; col < 16; ++col)
          dst_ptr[col] = uchar(dst_ptr[col] + lm_ptr[col]);
        dst_ptr += 16;
        lm_ptr += W;
      }
    }
  }
}

static void addUnaligned8u16u(const uchar * src1, const uchar * src2, ushort * res, int length)
{
  const uchar * end = src1 + length;

  while (src1 != end)
  {
    *res = *src1 + *src2;

    ++src1;
    ++src2;
    ++res;
  }
}

/**
 * \brief Accumulate one or more 8-bit similarity images.
 *
 * \param[in]  similarities Source 8-bit similarity images.
 * \param[out] dst          Destination 16-bit similarity image.
 */
static void addSimilarities(const std::vector<cv::Mat>& similarities, cv::Mat& dst)
{
  if (similarities.size() == 1)
  {
    similarities[0].convertTo(dst, CV_16U);
  }
  else
  {
    // NOTE: add() seems to be rather slow in the 8U + 8U -> 16U case
    dst.create(similarities[0].size(), CV_16U);
    addUnaligned8u16u(similarities[0].ptr(), similarities[1].ptr(), dst.ptr<ushort>(), static_cast<int>(dst.total()));

    /// @todo Optimize 16u + 8u -> 16u when more than 2 modalities
    for (size_t i = 2; i < similarities.size(); ++i)
      add(dst, similarities[i], dst, cv::noArray(), CV_16U);
  }
}

/****************************************************************************************\
*                               High-level Detector API                                  *
\****************************************************************************************/

Detector::Detector()
{
}

Detector::Detector(const std::vector< cv::Ptr<Modality> >& _modalities,
                   const std::vector<int>& T_pyramid)
  : modalities(_modalities),
    pyramid_levels(static_cast<int>(T_pyramid.size())),
    T_at_level(T_pyramid)
{
}

static void clusterCoordinate(std::vector<Match>& vecSrc, std::vector<Match>& vecDst, double dDis)
{
	int numbb = vecSrc.size();
	std::vector<int> T;
	int c = 1;
	switch (numbb)
	{
	case 1:
		vecDst = vecSrc;
		return;
		break;
	default:
		T = std::vector<int>(numbb, 0);
		c = cv::partition(vecSrc, T, CmpCoordinate<double, Match>(dDis));
		break;
	}

	vecDst.resize(c);
	for (int i = 0; i < c; i++)
	{
		Match cand, maxCand;
		float score, maxScore = FLT_MIN;
		for (int j = 0; j < T.size(); j++)
		{
			if (T[j] == i)
			{
				cand = vecSrc[j];
				score = cand.similarity;
				if (score > maxScore)
				{
					maxScore = score;
					maxCand = cand;
				}
			}
		}
		vecDst[i] = maxCand;
	}
}

void Detector::match(const cv::Mat& source, float threshold, std::vector<Match>& matches,
                     const std::vector<cv::String>& class_ids, const cv::Mat& mask) const
{
  // Initialize each modality with our sources
  std::vector< cv::Ptr<QuantizedPyramid> > quantizers;
  CV_Assert(mask.empty() || mask.size() == source.size());
  quantizers.push_back(modalities[0]->process(source, mask));
  
  // pyramid level -> modality -> quantization
  LinearMemoryPyramid lm_pyramid(pyramid_levels,
                                 std::vector<LinearMemories>(modalities.size(), LinearMemories(8)));

  // For each pyramid level, precompute linear memories for each modality
  std::vector<cv::Size> sizes;
  for (int l = 0; l < pyramid_levels; ++l)
  {
    int T = T_at_level[l];
    std::vector<LinearMemories>& lm_level = lm_pyramid[l];

    if (l > 0)
    {
      for (int i = 0; i < (int)quantizers.size(); ++i)
        quantizers[i]->pyrDown();
    }

    cv::Mat quantized, spread_quantized;
    std::vector<cv::Mat> response_maps;
    for (int i = 0; i < (int)quantizers.size(); ++i)
    {
      quantizers[i]->quantize(quantized);
      spread(quantized, spread_quantized, T);
      computeResponseMaps(spread_quantized, response_maps);

      LinearMemories& memories = lm_level[i];
      for (int j = 0; j < 8; ++j)
        linearize(response_maps[j], memories[j], T);

    }

    sizes.push_back(quantized.size());
  }

  if (class_ids.empty())
  {
    // Match all templates
    TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
	matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
  }
  else
  {
    // Match only templates for the requested class IDs
    for (int i = 0; i < (int)class_ids.size(); ++i)
    {
      TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
      if (it != class_templates.end())
        matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
    }
  }

  // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
  std::sort(matches.begin(), matches.end());
  std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
  matches.erase(new_end, matches.end());

  //add by HuangLi 2019/08/06
  //Remove candidate targets for region duplication
  if (matches.size() > 0)
  {
	  std::vector<Match> selectMatchs;
	  clusterCoordinate(matches, selectMatchs, 10);
	  matches = selectMatchs;
  }
}

//add by HuangLi 2019/08/13
Scene_edge Detector::initScene(const cv::Mat image)
{
	// buffer
	Scene_edge scene = Scene_edge();
	std::vector<::Vec2f> pcd_buffer, normal_buffer;
	scene.init_Scene_edge_cpu(image, pcd_buffer, normal_buffer);
	return scene;
}


cv::Mat Detector::getIcpAffine(const cv::Mat image, Match& match, Scene_edge& scene, bool valid, cv::Rect& roi)
{
	std::vector<Template> templ = getTemplates(match.class_id, match.template_id);
	std::vector<::Vec2f> model_pcd(templ[0].features.size());
	for (int j = 0; j < templ[0].features.size(); j++)
	{
		auto& feat = templ[0].features[j];
		model_pcd[j] =
		{
			float(feat.x + match.x),
			float(feat.y + match.y)
		};
	}
	valid = true;
	cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene, valid);

	cv::Mat affineImg;
	cv::Mat templImg = templ[0].templImg.clone();
	int tl_x, tl_y;
	tl_x = templ[0].tl_x;
	tl_y = templ[0].tl_y;
	float x, y, new_x, new_y;
	int res_x, res_y;
	if (templImg.channels() == 1)
	{
		affineImg = cv::Mat::zeros(templImg.size(), CV_8UC1);
		for (int row = 0; row < templImg.rows; row++)
		{
			uchar* ptr = templImg.ptr<uchar>(row);
			for (int col = 0; col < templImg.cols; col++)
			{
				x = col + match.x - tl_x;
				y = row + match.y - tl_y;
				new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
				new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
				res_x = new_x - match.x + tl_x;
				res_y = new_y - match.y + tl_y;
				if (res_x > 0 && res_x < templImg.cols && res_y > 0 && res_y < templImg.rows)
				{
					affineImg.at<uchar>(res_y, res_x) = ptr[col];
				}
			}
		}
		//去除孔噪点（彷射变换造成）
		cv::Mat repairImg = affineImg.clone();
		int val;
		for (int row = 1; row < affineImg.rows - 1; row++)
		{
			uchar* top_ptr = repairImg.ptr<uchar>(row - 1);
			uchar* ptr = repairImg.ptr<uchar>(row);
			uchar* bottom_ptr = repairImg.ptr<uchar>(row + 1);
			uchar* dst_ptr = affineImg.ptr<uchar>(row);
			for (int col = 1; col < affineImg.cols - 1; col++)
			{
				if (ptr[col] == 0)
				{
					val = cvRound((top_ptr[col] + ptr[col - 1] + bottom_ptr[col] + ptr[col + 1]) / 4.0);
					dst_ptr[col] = val;
				}
			}
		}
	}
	else
	{
		affineImg = cv::Mat::zeros(templImg.size(), CV_8UC3);
		for (int row = 0; row < templImg.rows; row++)
		{
			cv::Vec3b* ptr = templImg.ptr<cv::Vec3b>(row);
			for (int col = 0; col < templImg.cols; col++)
			{
				x = col + match.x - tl_x;
				y = row + match.y - tl_y;
				new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
				new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
				res_x = cvRound(new_x - match.x + tl_x);
				res_y = cvRound(new_y - match.y + tl_y);
				if (res_x > 0 && res_x < templImg.cols && res_y > 0 && res_y < templImg.rows)
				{
					affineImg.at<cv::Vec3b>(res_y, res_x) = ptr[col];
				}
			}
		}

		//去除孔噪点（彷射变换造成）
		cv::Mat repairImg = affineImg.clone();
		cv::Vec3b val;
		for (int row = 1; row < affineImg.rows - 1; row++)
		{
			cv::Vec3b* top_ptr = repairImg.ptr<cv::Vec3b>(row - 1);
			cv::Vec3b* ptr = repairImg.ptr<cv::Vec3b>(row);
			cv::Vec3b* bottom_ptr = repairImg.ptr<cv::Vec3b>(row + 1);
			cv::Vec3b* dst_ptr = affineImg.ptr<cv::Vec3b>(row);
			for (int col = 1; col < affineImg.cols - 1; col++)
			{
				if (ptr[col] == cv::Vec3b(0, 0, 0))
				{
					val = cv::Vec3b((top_ptr[col] + ptr[col - 1] + bottom_ptr[col] + ptr[col + 1]));
					val = cv::Vec3b(val[0] / 4, val[1] / 4, val[2] / 4);
					dst_ptr[col] = val;
				}
			}
		}
	}

	//match point transform
	/*x = match.x;
	y = match.y;
	match.x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
	match.y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];*/
	roi = cv::Rect(match.x - tl_x, match.y - tl_y, templImg.cols, templImg.rows);

	return affineImg;
}

struct MatchPredicate
{
  MatchPredicate(float _threshold) : threshold(_threshold) {}
  bool operator() (const Match& m) { return m.similarity < threshold; }
  float threshold;
};

void Detector::matchClass(const LinearMemoryPyramid& lm_pyramid,
                          const std::vector<cv::Size>& sizes,
                          float threshold, std::vector<Match>& matches,
                          const cv::String& class_id,
                          const std::vector<TemplatePyramid>& template_pyramids) const
{
  // For each template...
  for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
  {
    const TemplatePyramid& tp = template_pyramids[template_id];

    // First match over the whole image at the lowest pyramid level
    /// @todo Factor this out into separate function
    const std::vector<LinearMemories>& lowest_lm = lm_pyramid.back();

    // Compute similarity maps for each modality at lowest pyramid level
    std::vector<cv::Mat> similarities(modalities.size());
    int lowest_start = static_cast<int>(tp.size() - modalities.size());
    int lowest_T = T_at_level.back();
    int num_features = 0;
    for (int i = 0; i < (int)modalities.size(); ++i)
    {
      const Template& templ = tp[lowest_start + i];
      num_features += static_cast<int>(templ.features.size());
      similarity(lowest_lm[i], templ, similarities[i], sizes.back(), lowest_T);
    }

    // Combine into overall similarity
    /// @todo Support weighting the modalities
    cv::Mat total_similarity;
    addSimilarities(similarities, total_similarity);

    // Convert user-friendly percentage to raw similarity threshold. The percentage
    // threshold scales from half the max response (what you would expect from applying
    // the template to a completely random image) to the max response.
    // NOTE: This assumes max per-feature response is 4, so we scale between [2*nf, 4*nf].
    int raw_threshold = static_cast<int>(2*num_features + (threshold / 100.f) * (2*num_features) + 0.5f);

    // Find initial matches
    std::vector<Match> candidates;
    for (int r = 0; r < total_similarity.rows; ++r)
    {
      ushort* row = total_similarity.ptr<ushort>(r);
      for (int c = 0; c < total_similarity.cols; ++c)
      {
        int raw_score = row[c];
        if (raw_score > raw_threshold)
        {
          int offset = lowest_T / 2 + (lowest_T % 2 - 1);
          int x = c * lowest_T + offset;
          int y = r * lowest_T + offset;
          float score =(raw_score * 100.f) / (4 * num_features) + 0.5f;
          candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
        }
      }
    }

    // Locally refine each match by marching up the pyramid
    for (int l = pyramid_levels - 2; l >= 0; --l)
    {
      const std::vector<LinearMemories>& lms = lm_pyramid[l];
      int T = T_at_level[l];
      int start = static_cast<int>(l * modalities.size());
      cv::Size size = sizes[l];
      int border = 8 * T;
      int offset = T / 2 + (T % 2 - 1);
      int max_x = size.width - tp[start].width - border;
      int max_y = size.height - tp[start].height - border;

      std::vector<cv::Mat> similarities2(modalities.size());
      cv::Mat total_similarity2;
      for (int m = 0; m < (int)candidates.size(); ++m)
      {
        Match& match2 = candidates[m];
        int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
        int y = match2.y * 2 + 1;

        // Require 8 (reduced) row/cols to the up/left
        x = std::max(x, border);
        y = std::max(y, border);

        // Require 8 (reduced) row/cols to the down/left, plus the template size
        x = std::min(x, max_x);
        y = std::min(y, max_y);

        // Compute local similarity maps for each modality
        int numFeatures = 0;
        for (int i = 0; i < (int)modalities.size(); ++i)
        {
          const Template& templ = tp[start + i];
          numFeatures += static_cast<int>(templ.features.size());
          similarityLocal(lms[i], templ, similarities2[i], size, T, cv::Point(x, y));
        }
        addSimilarities(similarities2, total_similarity2);

        // Find best local adjustment
        int best_score = 0;
        int best_r = -1, best_c = -1;
        for (int r = 0; r < total_similarity2.rows; ++r)
        {
          ushort* row = total_similarity2.ptr<ushort>(r);
          for (int c = 0; c < total_similarity2.cols; ++c)
          {
            int score = row[c];
            if (score > best_score)
            {
              best_score = score;
              best_r = r;
              best_c = c;
            }
          }
        }
        // Update current match
        match2.x = (x / T - 8 + best_c) * T + offset;
        match2.y = (y / T - 8 + best_r) * T + offset;
        match2.similarity = (best_score * 100.f) / (4 * numFeatures);
      }

      // Filter out any matches that drop below the similarity threshold
      std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
                                                            MatchPredicate(threshold));
      candidates.erase(new_end, candidates.end());
    }

    matches.insert(matches.end(), candidates.begin(), candidates.end());
  }
}

int Detector::addTemplate(const std::vector<cv::Mat>& sources, const cv::String& class_id,
                          const cv::Mat& object_mask, cv::Rect* bounding_box)
{
  int num_modalities = static_cast<int>(modalities.size());
  std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
  int template_id = static_cast<int>(template_pyramids.size());

  TemplatePyramid tp;
  tp.resize(num_modalities * pyramid_levels);

  // For each modality...
  for (int i = 0; i < num_modalities; ++i)
  {
    // Extract a template at each pyramid level
    cv::Ptr<QuantizedPyramid> qp = modalities[i]->process(sources[i], object_mask);
    for (int l = 0; l < pyramid_levels; ++l)
    {
      /// @todo Could do mask subsampling here instead of in pyrDown()
      if (l > 0)
        qp->pyrDown();

      bool success = qp->extractTemplate(tp[l*num_modalities + i]);
      if (!success)
        return -1;
    }
  }

  cv::Rect bb = cropTemplates(tp);
  if (bounding_box)
    *bounding_box = bb;

  /// @todo Can probably avoid a copy of tp here with swap
  template_pyramids.push_back(tp);
  return template_id;
}

//add by HuangLi 2019/08/06
int Detector::addTemplate(const cv::Mat& sources, const cv::String& class_id,
	const cv::Mat& object_mask, cv::Rect* bounding_box, float angle, float scale)
{
	//int num_modalities = static_cast<int>(modalities.size());
	int num_modalities = 1;
	std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
	int template_id = static_cast<int>(template_pyramids.size());

	TemplatePyramid tp;
	tp.resize(num_modalities * pyramid_levels);

	// For each modality...
	for (int i = 0; i < num_modalities; ++i)
	{
		// Extract a template at each pyramid level
		cv::Ptr<QuantizedPyramid> qp = modalities[i]->process(sources, object_mask);
		for (int l = 0; l < pyramid_levels; ++l)
		{
			/// @todo Could do mask subsampling here instead of in pyrDown()
			if (l > 0)
				qp->pyrDown();
			bool success = qp->extractTemplate(tp[l*num_modalities + i]);
			//add by HuangLi 2019/08/08
			tp[l*num_modalities + i].angle = angle;
			tp[l*num_modalities + i].scale = scale;
			if (!success)
				return -1;
		}
	}

	cv::Rect bb = cropTemplates(tp);
	if (bounding_box)
		*bounding_box = bb;

	//add by HuangLi 2019/08/17
	for (int i = 0; i < num_modalities; ++i)
	{
		for (int l = 0; l < pyramid_levels; ++l)
		{
			tp[l*num_modalities + i].angle = angle;
			tp[l*num_modalities + i].scale = scale;
			tp[l*num_modalities + i].templImg = sources.clone();
		}
	}

	/// @todo Can probably avoid a copy of tp here with swap
	template_pyramids.push_back(tp);
	return template_id;
}

int Detector::addSyntheticTemplate(const std::vector<Template>& templates, const cv::String& class_id)
{
  std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
  int template_id = static_cast<int>(template_pyramids.size());
  template_pyramids.push_back(templates);
  return template_id;
}

const std::vector<Template>& Detector::getTemplates(const cv::String& class_id, int template_id) const
{
  TemplatesMap::const_iterator i = class_templates.find(class_id);
  CV_Assert(i != class_templates.end());
  CV_Assert(i->second.size() > size_t(template_id));
  return i->second[template_id];
}

int Detector::numTemplates() const
{
  int ret = 0;
  TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
  for ( ; i != iend; ++i)
    ret += static_cast<int>(i->second.size());
  return ret;
}

int Detector::numTemplates(const cv::String& class_id) const
{
  TemplatesMap::const_iterator i = class_templates.find(class_id);
  if (i == class_templates.end())
    return 0;
  return static_cast<int>(i->second.size());
}

std::vector<cv::String> Detector::classIds() const
{
  std::vector<cv::String> ids;
  TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
  for ( ; i != iend; ++i)
  {
    ids.push_back(i->first);
  }

  return ids;
}

void Detector::read(const cv::FileNode& fn)
{
  class_templates.clear();
  pyramid_levels = fn["pyramid_levels"];
  fn["T"] >> T_at_level;

  modalities.clear();
  cv::FileNode modalities_fn = fn["modalities"];
  cv::FileNodeIterator it = modalities_fn.begin(), it_end = modalities_fn.end();
  for ( ; it != it_end; ++it)
  {
    modalities.push_back(Modality::create(*it));
  }
}

void Detector::write(cv::FileStorage& fs) const
{
  fs << "pyramid_levels" << pyramid_levels;
  fs << "T" << T_at_level;

  fs << "modalities" << "[";
  for (int i = 0; i < (int)modalities.size(); ++i)
  {
    fs << "{";
    modalities[i]->write(fs);
    fs << "}";
  }
  fs << "]"; // modalities
}

  cv::String Detector::readClass(const cv::FileNode& fn, const cv::String &class_id_override)
  {
  // Verify compatible with Detector settings
  cv::FileNode mod_fn = fn["modalities"];
  CV_Assert(mod_fn.size() == modalities.size());
  cv::FileNodeIterator mod_it = mod_fn.begin(), mod_it_end = mod_fn.end();
  int i = 0;
  for ( ; mod_it != mod_it_end; ++mod_it, ++i)
    CV_Assert(modalities[i]->name() == (cv::String)(*mod_it));
  CV_Assert((int)fn["pyramid_levels"] == pyramid_levels);

  // Detector should not already have this class
    cv::String class_id;
    if (class_id_override.empty())
    {
      cv::String class_id_tmp = fn["class_id"];
      CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
      class_id = class_id_tmp;
    }
    else
    {
      class_id = class_id_override;
    }

  TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
  std::vector<TemplatePyramid>& tps = v.second;
  int expected_id = 0;

  cv::FileNode tps_fn = fn["template_pyramids"];
  tps.resize(tps_fn.size());
  cv::FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
  for ( ; tps_it != tps_it_end; ++tps_it, ++expected_id)
  {
    int template_id = (*tps_it)["template_id"];
    CV_Assert(template_id == expected_id);
    cv::FileNode templates_fn = (*tps_it)["templates"];
    tps[template_id].resize(templates_fn.size());

    cv::FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
    int idx = 0;
    for ( ; templ_it != templ_it_end; ++templ_it)
    {
      tps[template_id][idx++].read(*templ_it);
    }
  }

  class_templates.insert(v);
  return class_id;
}

void Detector::writeClass(const cv::String& class_id, cv::FileStorage& fs) const
{
  TemplatesMap::const_iterator it = class_templates.find(class_id);
  CV_Assert(it != class_templates.end());
  const std::vector<TemplatePyramid>& tps = it->second;

  fs << "class_id" << it->first;
  fs << "modalities" << "[:";
  for (size_t i = 0; i < modalities.size(); ++i)
    fs << modalities[i]->name();
  fs << "]"; // modalities
  fs << "pyramid_levels" << pyramid_levels;
  fs << "template_pyramids" << "[";
  for (size_t i = 0; i < tps.size(); ++i)
  {
    const TemplatePyramid& tp = tps[i];
    fs << "{";
    fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
    fs << "templates" << "[";
    for (size_t j = 0; j < tp.size(); ++j)
    {
      fs << "{";
      tp[j].write(fs);
      fs << "}"; // current template
    }
    fs << "]"; // templates
    fs << "}"; // current pyramid
  }
  fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<cv::String>& class_ids,
                           const cv::String& format)
{
  for (size_t i = 0; i < class_ids.size(); ++i)
  {
    const cv::String& class_id = class_ids[i];
    cv::String filename = cv::format(format.c_str(), class_id.c_str());
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    readClass(fs.root());
  }
}

void Detector::writeClasses(const cv::String& format) const
{
  TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
  for ( ; it != it_end; ++it)
  {
    const cv::String& class_id = it->first;
    cv::String filename = cv::format(format.c_str(), class_id.c_str());
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    writeClass(class_id, fs);
  }
}

//void Detector::drawResponse(const std::vector<matchHL::linemod::Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset, int T)
//{
//	static const cv::Scalar COLORS[5] = { CV_RGB(0, 255, 255),
//		CV_RGB(0, 255, 0),
//		CV_RGB(255, 255, 0),
//		CV_RGB(255, 140, 0),
//		CV_RGB(255, 0, 0) };
//
//	for (int m = 0; m < num_modalities; ++m)
//	{
//		// NOTE: Original demo recalculated max response for each feature in the TxT 
//		// box around it and chose the display color based on that response. Here 
//		// the display color just depends on the modality. 
//		cv::Scalar color = COLORS[m];
//
//		for (int i = 0; i < (int)templates[m].features.size(); ++i)
//		{
//			matchHL::linemod::Feature f = templates[m].features[i];
//			cv::Point pt(f.x + offset.x, f.y + offset.y);
//			cv::circle(dst, pt, T / 2, color);
//		}
//	}
//}

void Detector::drawResponse(const std::vector<Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset)
{
	static const cv::Scalar COLORS[5] = { CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 140, 0),
		CV_RGB(255, 0, 0) };

	for (int m = 0; m < num_modalities; ++m)
	{
		// NOTE: Original demo recalculated max response for each feature in the TxT 
		// box around it and chose the display color based on that response. Here 
		// the display color just depends on the modality. 
		cv::Scalar color = COLORS[m];

		for (int i = 0; i < (int)templates[m].features.size(); ++i)
		{
			auto feat = templates[m].features[i];
			float x = feat.x + offset.x;
			float y = feat.y + offset.y;
			cv::circle(dst, { int(x + 0.5f), int(y + 0.5f) }, 2, color, 1);
		}
	}
}

void Detector::drawResponse(const std::vector<Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset, cuda_icp::RegistrationResult& icpAffine)
{
	static const cv::Scalar COLORS[5] = { CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 140, 0),
		CV_RGB(255, 0, 0) };

	for (int m = 0; m < num_modalities; ++m)
	{
		// NOTE: Original demo recalculated max response for each feature in the TxT 
		// box around it and chose the display color based on that response. Here 
		// the display color just depends on the modality. 
		cv::Scalar color = COLORS[m];

		for (int i = 0; i < (int)templates[m].features.size(); ++i)
		{
			auto feat = templates[m].features[i];
			float x = feat.x + offset.x;
			float y = feat.y + offset.y;
			float new_x = icpAffine.transformation_[0][0] * x + icpAffine.transformation_[0][1] * y + icpAffine.transformation_[0][2];
			float new_y = icpAffine.transformation_[1][0] * x + icpAffine.transformation_[1][1] * y + icpAffine.transformation_[1][2];
			cv::circle(dst, { int(new_x + 0.5f), int(new_y + 0.5f) }, 2, color, 1);
		}
	}
}

//add by HuangLi 2019/08/14
void Detector::createModel(const cv::Mat& src, const cv::Mat& mask, std::string modelName)
{
	matchHL::shape_based_matching::shapeInfo shapes(src, mask);
	shapes.angle_range = { startAngle, endAngle };
	shapes.angle_step = stepAngle;
	shapes.produce_infos();
	std::vector<matchHL::shape_based_matching::shapeInfo::shape_and_info> infos_have_templ;
	std::string class_id;//对于单目标
	int template_id;
	cv::Rect bb;
	int num = 0;
	std::vector<std::string> vecClassId;
	std::vector<int> vecTemplateId;
	std::vector<cv::Rect> vecRect;
	//此处为单目标
	int index = 0;
	double time = 0.0;
	class_id = modelName;
	for (auto& info : shapes.infos)
	{
		cv::imshow("train", info.src);
		cv::waitKey(1);
		std::cout << "\ninfo.angle: " << info.angle << std::endl;
		time = cv::getTickCount();
		template_id = addTemplate(info.src, class_id, info.mask, &bb, info.angle, info.scale);
		time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
		std::cout << "Add Template cost: " << time << std::endl;
		std::cout << "template_id: " << template_id << std::endl;
		if (template_id != -1)
		{
			infos_have_templ.push_back(info);
			printf("*** Added template (id %d) for new object class %d***\n", template_id, num);
			printf("Extracted at (%d, %d) size %dx%d\n", bb.x, bb.y, bb.width, bb.height);
			vecClassId.push_back(class_id);
			vecTemplateId.push_back(template_id);
			vecRect.push_back(bb);
		}
		else
		{
			printf("Try adding template but failed.\n");
		}
	}
	if (vecRect.size() == 0)
	{
		printf("Create Template Error!!!\n");
		return;
	}

	/***********************************保存模板***************************************/
	writeClasses("%s.yaml");

	int num_modalities = (int)getModalities().size();
	printf("num_modalities = %d \n", num_modalities);
	//显示模板点
	cv::namedWindow("view template", cv::WINDOW_AUTOSIZE);
	float angle = startAngle;
	cv::Mat affineMat = cv::getRotationMatrix2D(cv::Point2f(src.cols / 2, src.rows / 2), angle, 1.0);
	cv::Mat display;
	cv::warpAffine(src, display, affineMat, src.size());
	template_id = 0;
	cv::Rect box = vecRect[0];
	const std::vector<matchHL::linemod::Template>& templates = getTemplates(class_id, template_id);
	drawResponse(templates, num_modalities, display, cv::Point(box.x, box.y));

	cv::imshow("view template", display);
	cv::waitKey(0);

	
}

//add by HuangLi 2019/08/14
void Detector::loadModel()
{
	/***********************************加载模板***************************************/
	int num = 0;
	std::vector<cv::String> ids;
	cv::String class_id = cv::format("class%d", num); //单目标，class_id固定，如class0
	ids.push_back(class_id);
	readClasses(ids, "%s.yaml");
}

//add by HuangLi 2019/08/16
cv::Mat Detector::getRoiImg(cv::Mat& src, cv::Rect roi)
{
	cv::Rect tempRoi;
	cv::Mat roiImg;
	int top, bottom, left, right;
	top = bottom = left = right = 0;
	if (roi.x < 0 || roi.x + roi.width >= src.cols || roi.y < 0 || roi.y + roi.height >= src.rows)
	{
		if (roi.x < 0)
		{
			left = -roi.x;
		}
		if (roi.x + roi.width > src.cols)
		{
			right = roi.x + roi.width - src.cols;
		}
		if (roi.y < 0)
		{
			top = -roi.y;
		}
		if (roi.y + roi.height > src.rows)
		{
			bottom = roi.y + roi.height - src.rows;
		}
		tempRoi = roi & cv::Rect(0, 0, src.cols, src.rows);
		roiImg = src(tempRoi).clone();
		cv::copyMakeBorder(roiImg, roiImg, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
	}
	else
	{
		roiImg = src(roi).clone();
	}

	return roiImg;
}

//add by HuangLi 2019/08/14
void Detector::detect(cv::Mat &src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, int amount, cv::Mat &mask)
{
	double time = cv::getTickCount();
	/************************************检测*****************************************/
	std::vector<matchHL::linemod::Match> matches;
	std::vector<cv::String> class_ids;
	std::vector<cv::Mat> quantized_images;
	match(src, matchThreshold, matches, class_ids, mask);
	matches.resize(amount);
	time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
	printf("查找模板耗时: %lf毫秒\n", time);

	time = cv::getTickCount();
	int classes_visited = 0;
	std::set<std::string> visited;
	if (src.channels() == 1)
	{
		cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
	}
	else
	{
		dst = src.clone();
	}
	
	Scene_edge scene;
	scene = initScene(src);
	cv::Rect roi, tempRoi;
	findObjects.resize(matches.size());
	int num_modalities = (int)getModalities().size();
#pragma omp parallel for
	for (int i = 0; i < (int)matches.size(); ++i)
	{
		auto match = matches[i];
		std::vector<matchHL::linemod::Template> templates = getTemplates(match.class_id, match.template_id);
		cv::Mat affineImg;
		bool valid = true;;
		affineImg = getIcpAffine(src, match, scene, valid, roi);
		drawResponse(templates, num_modalities, dst, cv::Point(match.x, match.y));
		findObjects[i] = { roi, templates[0].angle, match.similarity, affineImg };
	}
	time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
	printf("转换耗时: %lf毫秒\n", time);

}

//static const int T_DEFAULTS[] = {5, 8};
static const int T_DEFAULTS[] = { 4, 8 };

cv::Ptr<Detector> getDefaultLINE()
{
  std::vector< cv::Ptr<Modality> > modalities;
  modalities.push_back(cv::makePtr<ColorGradient>());
  return cv::makePtr<Detector>(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
}

cv::Ptr<Detector> getDefaultLINE(float _weak_threshold, size_t _num_features, float _strong_threshold)
{
	std::vector< cv::Ptr<Modality> > modalities;
	modalities.push_back(cv::makePtr<ColorGradient>(_weak_threshold, _num_features, _strong_threshold));
	return cv::makePtr<Detector>(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
}

cv::Ptr<Detector> getDefaultLINEMOD()
{
  std::vector< cv::Ptr<Modality> > modalities;
  modalities.push_back(cv::makePtr<ColorGradient>());
  modalities.push_back(cv::makePtr<DepthNormal>());
  return cv::makePtr<Detector>(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
}

} // namespace linemod
} // namespace cv
