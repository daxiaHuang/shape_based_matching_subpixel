#include "line2Dup.hpp"
#include <iostream>

using namespace std;
using namespace cv;

#include <chrono>
#include <direct.h>

static bool bDrawContour = false;
static bool bKeyPt = true;
static bool bMaxSelect = false;

bool Similarity(line2Dup::FindObject& b1, line2Dup::FindObject& b2)
{
	return b1.score > b2.score;
};

bool SimilarityMatch(line2Dup::Match& b1, line2Dup::Match& b2)
{
	return b1.similarity > b2.similarity;
};

struct Cmp
{
	Cmp(int p)
	{
		this->p = p;
	}
	bool operator()(cv::Rect b1, cv::Rect b2)
	{
		int exp = p * 0.5;
		cv::Rect rect, rect1, rect2;
		rect1 = b1;
		rect2 = b2;
		rect1 = cv::Rect(rect1.x - exp, rect1.y - exp, rect1.width + p, rect1.height + p);
		rect2 = cv::Rect(rect2.x - exp, rect2.y - exp, rect2.width + p, rect2.height + p);
		rect = rect1 & rect2;
		if (rect.area() > 0)
		{
			return true;
		}
		return false;
	}
	int p;
};

static void clusterRoi(const std::vector<cv::Rect>& vecSrc, std::vector<cv::Rect>& vecDst, double dDis)
{
	if (vecSrc.size() == 0)
	{
		vecDst.clear();
		return;
	}
	int numbb = vecSrc.size();
	std::vector<int> vecIndex;
	int c = 1;
	switch (numbb)
	{
	case 1:
		vecDst = std::vector<Rect>(1);
		vecDst = vecSrc;
		return;
		break;
	default:
		vecIndex = std::vector<int>(numbb, 0);
		c = cv::partition(vecSrc, vecIndex, Cmp(dDis));
		break;
	}

	cv::Rect rect;
	vecDst = std::vector<Rect>(c);
	for (int i = 0; i < c; i++)
	{
		std::vector<cv::Rect> pts;
		int x1, x2, y1, y2;
		x1 = y1 = INT_MAX;
		x2 = y2 = INT_MIN;
		for (int j = 0; j < vecIndex.size(); j++)
		{
			if (vecIndex[j] == i)
			{
				rect = vecSrc[j];
				x1 = std::min(x1, rect.x);
				y1 = std::min(y1, rect.y);
				x2 = std::max(x2, rect.x + rect.width);
				y2 = std::max(y2, rect.y + rect.height);
			}
		}
		rect = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
		vecDst[i] = rect;
	}
}


class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s\n" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

namespace line2Dup
{
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


	/**
	 * \brief Get the label [0,8) of the single bit set in quantized.
	 */
	static inline int getLabel(int quantized)
	{
		switch (quantized)
		{
		case 1:
			return 0;
		case 2:
			return 1;
		case 4:
			return 2;
		case 8:
			return 3;
		case 16:
			return 4;
		case 32:
			return 5;
		case 64:
			return 6;
		case 128:
			return 7;
		default:
			CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
			return -1; //avoid warning
		}
	}

	void Feature::read(const FileNode &fn)
	{
		FileNodeIterator fni = fn.begin();
		fni >> x >> y >> label;
	}

	void Feature::write(FileStorage &fs) const
	{
		fs << "[:" << x << y << label << "]";
	}

	void Template::read(const FileNode &fn)
	{
		width = fn["width"];
		height = fn["height"];
		tl_x = fn["tl_x"];
		tl_y = fn["tl_y"];
		pyramid_level = fn["pyramid_level"];

		fn["width"] >> width;
		fn["height"] >> height;
		fn["tl_x"] >> tl_x;
		fn["tl_y"] >> tl_y;
		fn["pyramid_level"] >> pyramid_level;

		fn["scale"] >> scale;
		fn["angle"] >> angle;
		//fn["templImg"] >> templImg;
		//fn["templMask"] >> templMask;
		fn["pt0"] >> pt0;
		fn["pt1"] >> pt1;
		fn["pt2"] >> pt2;
		fn["pt3"] >> pt3;
	
		FileNode features_fn = fn["features"];
		features.resize(features_fn.size());
		FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
		for (int i = 0; it != it_end; ++it, ++i)
		{
			features[i].read(*it);
		}
	}

	void Template::write(FileStorage &fs) const
	{
		fs << "width" << width;
		fs << "height" << height;
		fs << "tl_x" << tl_x;
		fs << "tl_y" << tl_y;
		fs << "pyramid_level" << pyramid_level;

		fs << "scale" << scale;
		fs << "angle" << angle;
		//fs << "templImg" << templImg;
		//fs << "templMask" << templMask;
		fs << "pt0" << pt0;
		fs << "pt1" << pt1;
		fs << "pt2" << pt2;
		fs << "pt3" << pt3;
	
		fs << "features"
		   << "[";
		for (int i = 0; i < (int)features.size(); ++i)
		{
			features[i].write(fs);
		}
		fs << "]"; // features
	}

	static Rect cropTemplates(std::vector<Template> &templates)
	{
		int min_x = std::numeric_limits<int>::max();
		int min_y = std::numeric_limits<int>::max();
		int max_x = std::numeric_limits<int>::min();
		int max_y = std::numeric_limits<int>::min();

		// First pass: find min/max feature x,y over all pyramid levels and modalities
		for (int i = 0; i < (int)templates.size(); ++i)
		{
			Template &templ = templates[i];

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
		if (min_x % 2 == 1)
			--min_x;
		if (min_y % 2 == 1)
			--min_y;

		// Second pass: set width/height and shift all feature positions
		for (int i = 0; i < (int)templates.size(); ++i)
		{
			Template &templ = templates[i];
			templ.width = (max_x - min_x) >> templ.pyramid_level;
			templ.height = (max_y - min_y) >> templ.pyramid_level;
			templ.tl_x = min_x >> templ.pyramid_level;
			templ.tl_y = min_y  >> templ.pyramid_level;

			for (int j = 0; j < (int)templ.features.size(); ++j)
			{
				templ.features[j].x -= templ.tl_x;
				templ.features[j].y -= templ.tl_y;
			}
		}

		return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	}

	bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates, std::vector<Feature> &features,
																	size_t num_features, float distance, cv::Mat& edgeMask)
	{
		features.clear();
		float distance_sq = distance * distance;
		int i = 0;

		bool first_select = true;

		while (true)
		{
			Candidate c = candidates[i];
			
			// Add if sufficient distance away from any previously chosen feature
			bool keep = true;
			for (int j = 0; (j < (int)features.size()) && keep; ++j)
			{
				Feature f = features[j];
				keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
			}
			//增加去除边界点功能（roi会截断产品，图像边界不一定是产品边界）
			/*if (keep)
			{
				features.push_back(c.f);
			}*/
			if (keep && edgeMask.at<uchar>(c.f.y, c.f.x) < 255)
			{
				features.push_back(c.f);
			}
				
			if (++i == (int)candidates.size()) 
			{
				bool num_ok = features.size() >= num_features;

				if (first_select)
				{
					if (num_ok)
					{
						features.clear(); // we don't want too many first time
						i = 0;
						distance += 1.0f;
						distance_sq = distance * distance;
						continue;
					}
					else 
					{
						first_select = false;
					}
				}

				// Start back at beginning, and relax required distance
				i = 0;
				distance -= 1.0f;
				distance_sq = distance * distance;
				if (num_ok || distance < 3) 
				{
					break;
				}
			}
		}
		if (features.size() >= num_features)
		{
			return true;
		}
		else
		{
			std::cout << "this templ has no enough features, but we let it go" << std::endl;
			return true;
		}

	}

	/****************************************************************************************\
	*                                                         Color gradient ColorGradient                                                                        *
	\****************************************************************************************/

	void hysteresisGradient(Mat &magnitude, Mat &quantized_angle,
							Mat &angle, float threshold)
	{
		// Quantize 360 degree range of orientations into 16 buckets
		// Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
		// for stability of horizontal and vertical features.
		Mat_<unsigned char> quantized_unfiltered;
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
			uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
			for (int c = 1; c < angle.cols - 1; ++c)
			{
				quant_r[c] &= 7;
			}
		}

		// Filter the raw quantized image. Only accept pixels where the magnitude is above some
		// threshold, and there is local agreement on the quantization.
		quantized_angle = Mat::zeros(angle.size(), CV_8U);
		for (int r = 1; r < angle.rows - 1; ++r)
		{
			float *mag_r = magnitude.ptr<float>(r);

			for (int c = 1; c < angle.cols - 1; ++c)
			{
				if (mag_r[c] > threshold)
				{
					// Compute histogram of quantized bins in 3x3 patch around pixel
					int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

					uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
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

	static void quantizedOrientations(const Mat &src, Mat &magnitude,
									  Mat &angle, Mat& angle_ori, float threshold)
	{
		cv::Mat smoothed;
		// Compute horizontal and vertical image derivatives on all color channels separately
		static const int KERNEL_SIZE = 7;
		// For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
		GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

		if(src.channels() == 1)
		{
			cv::Mat sobel_dx, sobel_dy, sobel_ag;
			Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
			Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
			magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
			phase(sobel_dx, sobel_dy, sobel_ag, true);
			hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
			angle_ori = sobel_ag;
		}
		else
		{

			magnitude.create(src.size(), CV_32F);

			// Allocate temporary buffers
			cv::Size size = src.size();
			cv::Mat sobel_3dx;              // per-channel horizontal derivative
			cv::Mat sobel_3dy;              // per-channel vertical derivative
			cv::Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
			cv::Mat sobel_dy(size, CV_32F); // maximum vertical derivative
			cv::Mat sobel_ag;               // final gradient orientation (unquantized)

			cv::Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
			cv::Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

			short *ptrx = (short *)sobel_3dx.data;
			short *ptry = (short *)sobel_3dy.data;
			float *ptr0x = (float *)sobel_dx.data;
			float *ptr0y = (float *)sobel_dy.data;
			float *ptrmg = (float *)magnitude.data;

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
			angle_ori = sobel_ag;
		}


	}

	ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
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
		quantizedOrientations(src, magnitude, angle, angle_ori, weak_threshold);
	}

	void ColorGradientPyramid::pyrDown()
	{
		// Some parameters need to be adjusted
		num_features /= 2; /// @todo Why not 4?
		++pyramid_level;

		// Downsample the current inputs
		Size size(src.cols / 2, src.rows / 2);
		Mat next_src;
		cv::pyrDown(src, next_src, size);
		src = next_src;

		if (!mask.empty())
		{
			Mat next_mask;
			resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
			mask = next_mask;
		}

		update();
	}

	void ColorGradientPyramid::quantize(Mat &dst) const
	{
		dst = Mat::zeros(angle.size(), CV_8U);
		angle.copyTo(dst, mask);
	}

	bool ColorGradientPyramid::extractTemplate(Template &templ, std::vector<cv::Point2f>& pts) const
	{
		//edgeMask用来排除边界点（有效区域可能产品截断，但并非是产品的边界）
		cv::Mat edgeMask = cv::Mat::zeros(mask.size(), CV_8UC1);
	
		if (1)
		{
			// Want features on the border to distinguish from background
			Mat local_mask;
			if (!mask.empty())
			{
				if (0)
				{
					erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
					subtract(mask, local_mask, local_mask);
				}
				else
				{
					local_mask = mask.clone();
				}
			}
			// Create sorted list of all pixels with magnitude greater than a threshold
			std::vector<Candidate> candidates;
			bool no_mask = local_mask.empty();
			float threshold_sq = strong_threshold * strong_threshold;
			for (int r = 0; r < magnitude.rows; ++r)
			{
				const uchar* angle_r = angle.ptr<uchar>(r);
				const float* magnitude_r = magnitude.ptr<float>(r);
				const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);
				//const uchar* ptr_mask = edgeMask.ptr<uchar>(r);
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
								candidates.back().f.theta = angle_ori.at<float>(r, c);
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

			// Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
			float distance = static_cast<float>(candidates.size() / num_features + 1);
			//add by HuangLi 2019/12/09
			//edgeMask用来排除边界点（有效区域可能产品截断，但并非是产品的边界）
			cv::Mat edgeMask = cv::Mat::zeros(mask.size(), CV_8UC1);
			int num = pts.size();
			if (num >= 2)
			{
				for (int i = 0; i < num - 1; ++i)
				{
					cv::line(edgeMask, pts[i], pts[i + 1], cv::Scalar(255), 3);
				}
				cv::line(edgeMask, pts[0], pts[num - 1], cv::Scalar(255), 3);
			}
			if (!selectScatteredFeatures(candidates, templ.features, num_features, distance, edgeMask))
			{
				return false;
			}

			// cv::Size determined externally, needs to match templates for other modalities
			templ.width = -1;
			templ.height = -1;
			templ.pyramid_level = pyramid_level;

		}
		else
		{
			// Want features on the border to distinguish from background
			Mat local_mask;
			if (!mask.empty())
			{
				//erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
				//        subtract(mask, local_mask, local_mask);
				local_mask = mask.clone();
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
						if (magnitude_valid.at<uchar>(r, c) > 0)
						{
							score = magnitude.at<float>(r, c);
							bool is_max = true;
							for (int r_offset = -nms_kernel_size / 2; r_offset <= nms_kernel_size / 2; r_offset++)
							{
								for (int c_offset = -nms_kernel_size / 2; c_offset <= nms_kernel_size / 2; c_offset++) 
								{
									if (r_offset == 0 && c_offset == 0) continue;

									if (score < magnitude.at<float>(r + r_offset, c + c_offset))
									{
										score = 0;
										is_max = false;
										break;
									}
								}
							}

							if (is_max) {
								for (int r_offset = -nms_kernel_size / 2; r_offset <= nms_kernel_size / 2; r_offset++) {
									for (int c_offset = -nms_kernel_size / 2; c_offset <= nms_kernel_size / 2; c_offset++) {
										if (r_offset == 0 && c_offset == 0) continue;
										magnitude_valid.at<uchar>(r + r_offset, c + c_offset) = 0;
									}
								}
							}
						}

						if (score > threshold_sq && angle.at<uchar>(r, c) > 0)
						{
							candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
							candidates.back().f.theta = angle_ori.at<float>(r, c);
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
			//add by HuangLi 2019/12/09
			//edgeMask用来排除边界点（有效区域可能产品截断，但并非是产品的边界）
			cv::Mat edgeMask = cv::Mat::zeros(mask.size(), CV_8UC1);
			int num = pts.size();
			if (num >= 2)
			{
				for (int i = 0; i < num - 1; ++i)
				{
					cv::line(edgeMask, pts[i], pts[i + 1], cv::Scalar(255), 3);
				}
				cv::line(edgeMask, pts[0], pts[num - 1], cv::Scalar(255), 3);
			}
			if (!selectScatteredFeatures(candidates, templ.features, num_features, distance, edgeMask))
			{
				return false;
			}

			// Size determined externally, needs to match templates for other modalities
			templ.width = -1;
			templ.height = -1;
			templ.pyramid_level = pyramid_level;
		}
   
		return true;
	}

	ColorGradient::ColorGradient()
		: weak_threshold(30.0f),
		  num_features(63),
		  strong_threshold(60.0f)
	{
	}

	ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
		: weak_threshold(_weak_threshold),
		  num_features(_num_features),
		  strong_threshold(_strong_threshold)
	{
	}

	static const char CG_NAME[] = "ColorGradient";

	std::string ColorGradient::name() const
	{
		return CG_NAME;
	}

	void ColorGradient::read(const FileNode &fn)
	{
		String type = fn["type"];
		CV_Assert(type == CG_NAME);

		weak_threshold = fn["weak_threshold"];
		num_features = int(fn["num_features"]);
		strong_threshold = fn["strong_threshold"];
	}

	void ColorGradient::write(FileStorage &fs) const
	{
		fs << "type" << CG_NAME;
		fs << "weak_threshold" << weak_threshold;
		fs << "num_features" << int(num_features);
		fs << "strong_threshold" << strong_threshold;
	}
	/****************************************************************************************\
	*                                                                 Response maps                                                                                    *
	\****************************************************************************************/

	static void orUnaligned8u(const uchar *src, const int src_stride,
							  uchar *dst, const int dst_stride,
							  const int width, const int height)
	{
		for (int r = 0; r < height; ++r)
		{
			int c = 0;

			// not aligned, which will happen because we move 1 bytes a time for spreading
			while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0) {
				dst[c] |= src[c];
				c++;
			}

			// avoid out of bound when can't divid
			// note: can't use c<width !!!
			for (; c <= width-mipp::N<uint8_t>(); c+=mipp::N<uint8_t>()){
				mipp::Reg<uint8_t> src_v((uint8_t*)src + c);
				mipp::Reg<uint8_t> dst_v((uint8_t*)dst + c);

				mipp::Reg<uint8_t> res_v = mipp::orb(src_v, dst_v);
				res_v.store((uint8_t*)dst + c);
			}

			for(; c<width; c++)
				dst[c] |= src[c];

			// Advance to next row
			src += src_stride;
			dst += dst_stride;
		}
	}

	static void spread(const Mat &src, Mat &dst, int T)
	{
		// Allocate and zero-initialize spread (OR'ed) image
		dst = Mat::zeros(src.size(), CV_8U);

		// Fill in spread gradient image (section 2.3)
		for (int r = 0; r < T; ++r)
		{
			for (int c = 0; c < T; ++c)
			{
				orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(),
							  static_cast<const int>(dst.step1()), src.cols - c, src.rows - r);
			}
		}
	}

	// 1,2-->0 3-->1
	CV_DECL_ALIGNED(16) static const unsigned char SIMILARITY_LUT[256] = {0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4};

	static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
	{
		CV_Assert((src.rows * src.cols) % 16 == 0);

		// Allocate response maps
		response_maps.resize(8);
		for (int i = 0; i < 8; ++i)
			response_maps[i].create(src.size(), CV_8U);

		Mat lsb4(src.size(), CV_8U);
		Mat msb4(src.size(), CV_8U);

		for (int r = 0; r < src.rows; ++r)
		{
			const uchar *src_r = src.ptr(r);
			uchar *lsb4_r = lsb4.ptr(r);
			uchar *msb4_r = msb4.ptr(r);

			for (int c = 0; c < src.cols; ++c)
			{
				// Least significant 4 bits of spread image pixel
				lsb4_r[c] = src_r[c] & 15;
				// Most significant 4 bits, right-shifted to be in [0, 16)
				msb4_r[c] = (src_r[c] & 240) >> 4;
			}
		}

		{
			uchar *lsb4_data = lsb4.ptr<uchar>();
			uchar *msb4_data = msb4.ptr<uchar>();

			bool no_max = true;
			bool no_shuff = true;

	#ifdef has_max_int8_t
			no_max = false;
	#endif

	#ifdef has_shuff_int8_t
			no_shuff = false;
	#endif
			// LUT is designed for 128 bits SIMD, so quite triky for others

			// For each of the 8 quantized orientations...
			for (int ori = 0; ori < 8; ++ori){
				uchar *map_data = response_maps[ori].ptr<uchar>();
				const uchar *lut_low = SIMILARITY_LUT + 32 * ori;

				if(mipp::N<uint8_t>() == 1 || no_max || no_shuff){ // no SIMD
					for (int i = 0; i < src.rows * src.cols; ++i)
						map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
				}
				else if(mipp::N<uint8_t>() == 16){ // 128 SIMD, no add base

					const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
					mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);
					mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);

					for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
						mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);
						mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);

						mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
						mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

						mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
						result.store((uint8_t*)map_data + i);
					}
				}
				else if(mipp::N<uint8_t>() == 16 || mipp::N<uint8_t>() == 32
						|| mipp::N<uint8_t>() == 64){ //128 256 512 SIMD
					CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0);

					uint8_t lut_temp[mipp::N<uint8_t>()] = {0};

					for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
						std::copy_n(lut_low, 16, lut_temp+slice*16);
					}
					mipp::Reg<uint8_t> lut_low_v(lut_temp);

					uint8_t base_add_array[mipp::N<uint8_t>()] = {0};
					for(uint8_t slice=0; slice<mipp::N<uint8_t>(); slice+=16){
						std::copy_n(lut_low+16, 16, lut_temp+slice);
						std::fill_n(base_add_array+slice, 16, slice);
					}
					mipp::Reg<uint8_t> base_add(base_add_array);
					mipp::Reg<uint8_t> lut_high_v(lut_temp);

					for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()){
						mipp::Reg<uint8_t> mask_low_v((uint8_t*)lsb4_data+i);
						mipp::Reg<uint8_t> mask_high_v((uint8_t*)msb4_data+i);

						mask_low_v += base_add;
						mask_high_v += base_add;

						mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
						mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);

						mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
						result.store((uint8_t*)map_data + i);
					}
				}
				else{
					for (int i = 0; i < src.rows * src.cols; ++i)
						map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
				}
			}


		}
	}

	static void linearize(const Mat &response_map, Mat &linearized, int T)
	{
		CV_Assert(response_map.rows % T == 0);
		CV_Assert(response_map.cols % T == 0);

		// linearized has T^2 rows, where each row is a linear memory
		int mem_width = response_map.cols / T;
		int mem_height = response_map.rows / T;
		linearized.create(T * T, mem_width * mem_height, CV_8U);

		// Outer two for loops iterate over top-left T^2 starting pixels
		int index = 0;
		for (int r_start = 0; r_start < T; ++r_start)
		{
			for (int c_start = 0; c_start < T; ++c_start)
			{
				uchar *memory = linearized.ptr(index);
				++index;

				// Inner two loops copy every T-th pixel into the linear memory
				for (int r = r_start; r < response_map.rows; r += T)
				{
					const uchar *response_data = response_map.ptr(r);
					for (int c = c_start; c < response_map.cols; c += T)
						*memory++ = response_data[c];
				}
			}
		}
	}
	/****************************************************************************************\
	*                                                             Linearized similarities                                                                    *
	\****************************************************************************************/

	static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,
												   const Feature &f, int T, int W)
	{
		// Retrieve the TxT grid of linear memories associated with the feature label
		const Mat &memory_grid = linear_memories[f.label];
		CV_DbgAssert(memory_grid.rows == T * T);
		CV_DbgAssert(f.x >= 0);
		CV_DbgAssert(f.y >= 0);
		// The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
		int grid_x = f.x % T;
		int grid_y = f.y % T;
		int grid_index = grid_y * T + grid_x;
		CV_DbgAssert(grid_index >= 0);
		CV_DbgAssert(grid_index < memory_grid.rows);
		const unsigned char *memory = memory_grid.ptr(grid_index);
		// Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
		// input image width decimated by T.
		int lm_x = f.x / T;
		int lm_y = f.y / T;
		int lm_index = lm_y * W + lm_x;
		CV_DbgAssert(lm_index >= 0);
		CV_DbgAssert(lm_index < memory_grid.cols);
		return memory + lm_index;
	}

	static void similarity(const std::vector<Mat> &linear_memories, const Template &templ,
						   Mat &dst, Size size, int T)
	{
		// we only have one modality, so 8192*2, due to mipp, back to 8192
		CV_Assert(templ.features.size() < 8192);

		// Decimate input image size by factor of T
		int W = size.width / T;
		int H = size.height / T;

		// Feature dimensions, decimated by factor T and rounded up
		int wf = (templ.width - 1) / T + 1;
		int hf = (templ.height - 1) / T + 1;

		// Span is the range over which we can shift the template around the input image
		int span_x = W - wf;
		int span_y = H - hf;

		int template_positions = span_y * W + span_x + 1; // why add 1?

		dst = Mat::zeros(H, W, CV_16U);
		short *dst_ptr = dst.ptr<short>();
		mipp::Reg<uint8_t> zero_v(uint8_t(0));

		for (int i = 0; i < (int)templ.features.size(); ++i)
		{

			Feature f = templ.features[i];

			if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
				continue;
			const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

			int j = 0;

			// *2 to avoid int8 read out of range
			for(; j <= template_positions -mipp::N<int16_t>()*2; j+=mipp::N<int16_t>()){
				mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);

				// uchar to short, once for N bytes
				mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

				mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);

				mipp::Reg<int16_t> res_v = src16_v + dst_v;
				res_v.store((int16_t*)dst_ptr + j);
			}

			for(; j<template_positions; j++)
				dst_ptr[j] += short(lm_ptr[j]);
		}
	}

	static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
								Mat &dst, Size size, int T, Point center)
	{
		CV_Assert(templ.features.size() < 8192);

		int W = size.width / T;
		dst = Mat::zeros(16, 16, CV_16U);

		int offset_x = (center.x / T - 8) * T;
		int offset_y = (center.y / T - 8) * T;
		mipp::Reg<uint8_t> zero_v = uint8_t(0);

		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			Feature f = templ.features[i];
			f.x += offset_x;
			f.y += offset_y;
			// Discard feature if out of bounds, possibly due to applying the offset
			if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
				continue;

			const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
			{
				short *dst_ptr = dst.ptr<short>();

				if(mipp::N<uint8_t>() > 32){ //512 bits SIMD
					for (int row = 0; row < 16; row += mipp::N<int16_t>()/16){
						mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + row*16);

						// load lm_ptr, 16 bytes once, for half
						uint8_t local_v[mipp::N<uint8_t>()] = {0};
						for(int slice=0; slice<mipp::N<uint8_t>()/16/2; slice++){
							std::copy_n(lm_ptr, 16, &local_v[16*slice]);
							lm_ptr += W;
						}
						mipp::Reg<uint8_t> src8_v(local_v);
						// uchar to short, once for N bytes
						mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

						mipp::Reg<int16_t> res_v = src16_v + dst_v;
						res_v.store((int16_t*)dst_ptr);

						dst_ptr += mipp::N<int16_t>();
					}
				}else{ // 256 128 or no SIMD
					for (int row = 0; row < 16; ++row){
						for(int col=0; col<16; col+=mipp::N<int16_t>()){
							mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + col);

							// uchar to short, once for N bytes
							mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

							mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + col);
							mipp::Reg<int16_t> res_v = src16_v + dst_v;
							res_v.store((int16_t*)dst_ptr + col);
						}
						dst_ptr += 16;
						lm_ptr += W;
					}
				}
			}
		}
	}

	static void similarity_64(const std::vector<Mat> &linear_memories, const Template &templ,
							  Mat &dst, Size size, int T)
	{
		// 63 features or less is a special case because the max similarity per-feature is 4.
		// 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
		// about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
		// general function would use _mm_add_epi16.
		CV_Assert(templ.features.size() < 64);
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
		dst = Mat::zeros(H, W, CV_8U);
		uchar *dst_ptr = dst.ptr<uchar>();

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
			const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

			// Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
			int j = 0;

			for(; j <= template_positions -mipp::N<uint8_t>(); j+=mipp::N<uint8_t>()){
				mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + j);
				mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + j);

				mipp::Reg<uint8_t> res_v = src_v + dst_v;
				res_v.store((uint8_t*)dst_ptr + j);
			}

			for(; j<template_positions; j++)
				dst_ptr[j] += lm_ptr[j];
		}
	}

	static void similarityLocal_64(const std::vector<Mat> &linear_memories, const Template &templ,
								   Mat &dst, Size size, int T, Point center)
	{
		// Similar to whole-image similarity() above. This version takes a position 'center'
		// and computes the energy in the 16x16 patch centered on it.
		CV_Assert(templ.features.size() < 64);

		// Compute the similarity map in a 16x16 patch around center
		int W = size.width / T;
		dst = Mat::zeros(16, 16, CV_8U);

		// Offset each feature point by the requested center. Further adjust to (-8,-8) from the
		// center to get the top-left corner of the 16x16 patch.
		// NOTE: We make the offsets multiples of T to agree with results of the original code.
		int offset_x = (center.x / T - 8) * T;
		int offset_y = (center.y / T - 8) * T;

		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			Feature f = templ.features[i];
			f.x += offset_x;
			f.y += offset_y;
			// Discard feature if out of bounds, possibly due to applying the offset
			if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
				continue;

			const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

			{
				uchar *dst_ptr = dst.ptr<uchar>();

				if(mipp::N<uint8_t>() > 16){ // 256 or 512 bits SIMD
					for (int row = 0; row < 16; row += mipp::N<uint8_t>()/16){
						mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr);

						// load lm_ptr, 16 bytes once
						uint8_t local_v[mipp::N<uint8_t>()];
						for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
							std::copy_n(lm_ptr, 16, &local_v[16*slice]);
							lm_ptr += W;
						}
						mipp::Reg<uint8_t> src_v(local_v);

						mipp::Reg<uint8_t> res_v = src_v + dst_v;
						res_v.store((uint8_t*)dst_ptr);

						dst_ptr += mipp::N<uint8_t>();
					}
				}else{ // 128 or no SIMD
					for (int row = 0; row < 16; ++row){
						for(int col=0; col<16; col+=mipp::N<uint8_t>()){
							mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + col);
							mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + col);
							mipp::Reg<uint8_t> res_v = src_v + dst_v;
							res_v.store((uint8_t*)dst_ptr + col);
						}
						dst_ptr += 16;
						lm_ptr += W;
					}
				}
			}
		}
	}

	/****************************************************************************************\
	*                                                             High-level Detector API                                                                    *
	\****************************************************************************************/

	Detector::Detector()
	{
		this->modality = makePtr<ColorGradient>();
		pyramid_levels = 2;
		T_at_level.push_back(4);
		T_at_level.push_back(8);
		bIcp = false;
		bExitModel = false;

		m_callbackFunc = nullptr;
	}

	Detector::Detector(std::vector<int> T)
	{
		this->modality = makePtr<ColorGradient>();
		pyramid_levels = T.size();
		T_at_level = T;
		bIcp = false;
	}

	Detector::Detector(int num_features, std::vector<int> T, float weak_thresh, float strong_threash)
	{
		this->modality = makePtr<ColorGradient>(weak_thresh, num_features, strong_threash);
		pyramid_levels = T.size();
		T_at_level = T;
		bIcp = false;
	}

	/*计算欧式距离*/
	static float calcuDistance(int* ptr, int* ptrCen, int cols) 
	{
		float x1 = ptr[0];
		float y1 = ptr[1];
		float x2 = ptrCen[0];
		float y2 = ptrCen[1];

		//cout << x1 <<" " << y1 << " " << x2 << " " << y2 << " " << endl;

		float d = 0.0;
		for (size_t j = 0; j < cols; j++)
		{
			d += (double)(ptr[j] - ptrCen[j])*(ptr[j] - ptrCen[j]);
		}
		d = sqrt(d);
		return d;
	}

	/** @brief   最大最小距离算法
	 @param data  输入样本数据，每一行为一个样本，每个样本可以存在多个特征数据
	 @param Theta 阈值，阈值越小聚类中心越多
	 @return 返回每个样本的类别，类别从1开始，0表示未分类或者分类失败
	*/
	static std::vector<int> maxMinDisCluster(cv::Mat data, float Theta) 
	{
		double maxDistance = 0;
		int start = 0;    //初始选一个中心点
		int index = start; //相当于指针指示新中心点的位置
		int k = 0;        //中心点计数，也即是类别
		int dataNum = data.rows; //输入的样本数
		vector<int>	centerIndex;//保存中心点
		cv::Mat distance = cv::Mat::zeros(cv::Size(1, dataNum), CV_32FC1); //表示所有样本到当前聚类中心的距离
		cv::Mat minDistance = cv::Mat::zeros(cv::Size(1, dataNum), CV_32FC1); //取较小距离
		//cv::Mat classes = cv::Mat::zeros(cv::Size(1, dataNum), CV_32SC1);     //表示类别
		std::vector<int> classes(dataNum);
		centerIndex.push_back(index); //保存第一个聚类中心

		for (size_t i = 0; i < dataNum; i++)
		{
			int* ptr1 = data.ptr<int>(i);
			int* ptrCen = data.ptr<int>(centerIndex.at(0));
			float d = calcuDistance(ptr1, ptrCen, data.cols);
			distance.at<float>(i, 0) = d;
			classes[i] = k;
			if (maxDistance < d)
			{
				maxDistance = d;
				index = i; //与第一个聚类中心距离最大的样本
			}
		}
		minDistance = distance.clone();
		double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
		maxVal = maxDistance;
		//while (maxVal > maxDistance*0.5)
		while (maxVal > Theta)
		{
			k = k + 1;
			centerIndex.push_back(index); //新的聚类中心
			for (size_t i = 0; i < dataNum; i++)
			{
				int* ptr1 = data.ptr<int>(i);
				int* ptrCen = data.ptr<int>(centerIndex.at(k));
				float d = calcuDistance(ptr1, ptrCen, data.cols);
				distance.at<float>(i, 0) = d;
				//按照当前最近临方式分类，哪个近就分哪个类别
				if (minDistance.at<float>(i, 0) > distance.at<float>(i, 0))
				{
					minDistance.at<float>(i, 0) = distance.at<float>(i, 0);
					classes[i] = k;
				}
			}
			//查找minDistance中最大值
			cv::minMaxLoc(minDistance, &minVal, &maxVal, &minLoc, &maxLoc);
			index = maxLoc.y;
		}
		return classes;
	}

	//Spend too much time of this clustering method, using following method
	//static void clusterCoordinate(std::vector<Match>& vecSrc, std::vector<Match>& vecDst, double dDis)
	//{
	//	int numbb = vecSrc.size();
	//	std::vector<int> T;
	//	int c = 1;
	//	switch (numbb)
	//	{
	//	case 1:
	//		vecDst = vecSrc;
	//		return;
	//		break;
	//	default:
	//		T = std::vector<int>(numbb, 0);
	//		c = cv::partition(vecSrc, T, CmpCoordinate<double, Match>(dDis));
	//		break;
	//	}
	//
	//	vecDst.resize(c);
	//	for (int i = 0; i < c; i++)
	//	{
	//		Match cand, maxCand;
	//		float score, maxScore = FLT_MIN;
	//		for (int j = 0; j < T.size(); j++)
	//		{
	//			if (T[j] == i)
	//			{
	//				cand = vecSrc[j];
	//				score = cand.similarity;
	//				if (score > maxScore)
	//				{
	//					maxScore = score;
	//					maxCand = cand;
	//				}
	//			}
	//		}
	//		vecDst[i] = maxCand;
	//	}
	//}

	//最大最小距离算法 https://blog.csdn.net/guyuealian/article/details/80255524
	static void clusterCoordinate(std::vector<Match>& vecSrc, std::vector<Match>& vecDst, double dDis)
	{
		//cluster to reduce matching categories
		cv::Mat data = cv::Mat(vecSrc.size(), 2, CV_32SC1);
		Match match;
		for (int i = 0; i < vecSrc.size(); i++)
		{
			match = vecSrc[i];
			int* ptr = data.ptr<int>(i);
			ptr[0] = match.x;
			ptr[1] = match.y;
		}
		std::vector<int> classes = maxMinDisCluster(data, 15);
		if (classes.size() == 0)
		{
			return;
		}
		//Calculate the number of categories
		int categories = INT_MIN;
		int val;
		for (int i = 0; i < classes.size(); i++)
		{
			val = classes[i];
			if (categories < val)
			{
				categories = val;
			}
		}
		categories += 1;//categories start from zero

		std::vector<int> similaritys(categories, 0);
		std::vector<int> idxs(categories, 0);
	#pragma omp parallel for
		for (int i = 0; i < categories; i++)
		{
			double similarity;
			for (int j = 0; j < classes.size(); j++)
			{
				if (classes[j] == i)
				{
					similarity = vecSrc[j].similarity;
					if (similarity > similaritys[i])
					{
						similaritys[i] = similarity;
						idxs[i] = j;
					}
				}
			}
		}

		vecDst.resize(idxs.size());
		for (int i = 0; i < idxs.size(); i++)
		{
			vecDst[i] = vecSrc[idxs[i]];
		}
	}

	//最大最小距离算法 https://blog.csdn.net/guyuealian/article/details/80255524
	static void clusterCoordinate(std::vector<Match>& vecSrc, std::vector<std::vector<Match> >& vecDst, double dDis)
	{
		//cluster to reduce matching categories
		cv::Mat data = cv::Mat(vecSrc.size(), 2, CV_32SC1);
		Match match;
		for (int i = 0; i < vecSrc.size(); i++)
		{
			match = vecSrc[i];
			int* ptr = data.ptr<int>(i);
			ptr[0] = match.x;
			ptr[1] = match.y;
		}
		std::vector<int> classes = maxMinDisCluster(data, 15);
		if (classes.size() == 0)
		{
			return;
		}
		//Calculate the number of categories
		int categories = INT_MIN;
		int val;
		for (int i = 0; i < classes.size(); i++)
		{
			val = classes[i];
			if (categories < val)
			{
				categories = val;
			}
		}
		categories += 1;//categories start from zero

		std::vector<int> similaritys(categories, 0);
		std::vector<int> idxs(categories, 0);
		vecDst.resize(categories);
#pragma omp parallel for
		for (int i = 0; i < categories; i++)
		{
			double similarity;
			std::vector<Match> vecTemp;
			for (int j = 0; j < classes.size(); j++)
			{
				if (classes[j] == i)
				{
					vecTemp.push_back(vecSrc[j]);
				}
				
			}
			vecDst[i] = vecTemp;
		}
	}

	std::map<std::string, std::vector<Match> > Detector::match(cv::Mat source, float threshold, const std::vector<std::string> &class_ids, const cv::Mat mask, const float startAngle, const float endAngle)
	{
		// Initialize each ColorGradient with our sources
		std::vector<Ptr<ColorGradientPyramid>> quantizers;
		CV_Assert(mask.empty() || mask.size() == source.size());
		quantizers.push_back(modality->process(source, mask));

		// pyramid level -> ColorGradient -> quantization
		LinearMemoryPyramid lm_pyramid(pyramid_levels,
									   std::vector<LinearMemories>(1, LinearMemories(8)));

		// For each pyramid level, precompute linear memories for each ColorGradient
		std::vector<Size> sizes;
		for (int l = 0; l < pyramid_levels; ++l)
		{
			int T = T_at_level[l];
			std::vector<LinearMemories> &lm_level = lm_pyramid[l];

			if (l > 0)
			{
				for (int i = 0; i < (int)quantizers.size(); ++i)
					quantizers[i]->pyrDown();
			}

			Mat quantized, spread_quantized;
			std::vector<Mat> response_maps;
			for (int i = 0; i < (int)quantizers.size(); ++i)
			{
				quantizers[i]->quantize(quantized);
				spread(quantized, spread_quantized, T);
				computeResponseMaps(spread_quantized, response_maps);

				LinearMemories &memories = lm_level[i];
				for (int j = 0; j < 8; ++j)
					linearize(response_maps[j], memories[j], T);
			}

			sizes.push_back(quantized.size());
		}

		std::map<string, std::vector<Match> > mapMatches;
		float minThreshold = 40;
		if (class_ids.empty())
		{
			// Match all templates
			TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
			for (; it != itend; ++it)
			{
				std::vector<Match> tmpMatch;
				//matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
				matchClass(lm_pyramid, sizes, minThreshold, tmpMatch, it->first, it->second, startAngle, endAngle);
				mapMatches.insert(make_pair(it->first, tmpMatch));
			}
		}
		else
		{
			// Match only templates for the requested class IDs
			for (int i = 0; i < (int)class_ids.size(); ++i)
			{
				TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
				if (it != class_templates.end())
				{
					std::vector<Match> tmpMatch;
					//matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
					matchClass(lm_pyramid, sizes, minThreshold, tmpMatch, it->first, it->second, startAngle, endAngle);
					mapMatches.insert(make_pair(it->first, tmpMatch));
				}
			}
		}

		Match match;
		std::vector<Match> matches, tmpMatches;
		std::map<std::string, std::vector<Match> > mapResultMatches;
		std::map<string, std::vector<Match> >::const_iterator it = mapMatches.begin(), itend = mapMatches.end();
		//step1 根据分数进行筛选
		for (; it != itend; ++it)
		{
			matches = it->second;
			tmpMatches.clear();
			std::vector<Match>::iterator it1 = matches.begin();
			for (; it1 != matches.end(); )
			{
				if (it1->similarity < threshold)
				{
					it1 = matches.erase(it1);
				}
				else
				{
					it1++;
				}
			}
			mapMatches[it->first] = matches;
		}
		
		//step2 根据类别先分类
		it = mapMatches.begin(), itend = mapMatches.end();
		for (; it != itend; ++it)
		{
			matches = it->second;
			//Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
			std::sort(matches.begin(), matches.end());
			std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
			matches.erase(new_end, matches.end());

			//cluster to reduce matching categories
			double time = cv::getTickCount();
			std::vector<Match> selectMatchs;
			std::vector<std::vector<Match> > resultMatches;
			if (bMaxSelect)//采用取最大值
			{
				if (matches.size() > 0)
				{
					clusterCoordinate(matches, selectMatchs, 20);
					resultMatches.push_back(selectMatchs);
				}
			}
			else//采用取N最大值
			{
				if (matches.size() > 0)
				{
					int th = 3;
					clusterCoordinate(matches, resultMatches, 20);
					std::vector<Match> vecTemp;
					matches.resize(resultMatches.size());
					for (int i = 0; i < resultMatches.size(); ++i)
					{
						vecTemp = resultMatches[i];
						//排序
						std::sort(vecTemp.begin(), vecTemp.end(), SimilarityMatch);
						//去除后面若干个
						double maxSimilarity = vecTemp[0].similarity;
						Match matchTemp;
						float angle = 0.0;
						int num = 0;
						for (int j = 0; j < vecTemp.size(); ++j)
						{
							matchTemp = vecTemp[j];
							if (maxSimilarity - matchTemp.similarity < th)
							{
								angle += matchTemp.angle;
								num++;
							}
							else
							{
								break;
							}
						}
						angle /= (num + 1e-9);
						vecTemp[0].angle = angle;
						selectMatchs.push_back(vecTemp[0]);
					}
				}
			}
			mapResultMatches.insert(make_pair(it->first, selectMatchs));
		}

		//Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
		/*
		std::sort(matches.begin(), matches.end());
		std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
		matches.erase(new_end, matches.end());

		//cluster to reduce matching categories
		if (bMaxSelect)//采用取最大值
		{
			if (matches.size() > 0)
			{
				std::vector<Match> selectMatchs;
				clusterCoordinate(matches, selectMatchs, 20);
				resultMatches.push_back(selectMatchs);
			}
		}
		else//采用取N最大值
		{
			if (matches.size() > 0)
			{
				int th = 4;
				clusterCoordinate(matches, resultMatches, 20);
				std::vector<Match> vecTemp;
				matches.resize(resultMatches.size());
				for (int i = 0; i < resultMatches.size(); ++i)
				{
					vecTemp = resultMatches[i];
					//排序
					std::sort(vecTemp.begin(), vecTemp.end(), SimilarityMatch);
					//去除后面若干个
					double maxSimilarity = vecTemp[0].similarity;
					int idx = 0;
					Match matchTemp;
					for (int j = 0; j < vecTemp.size(); ++j)
					{
						matchTemp = vecTemp[j];
						if (maxSimilarity - matchTemp.similarity > th)
						{
							idx = j;
							break;
						}
					}
					resultMatches[i].resize(idx+1);
				}
			}
		}
		*/
	
		return mapResultMatches;
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

	cv::Mat Detector::getIcpAffine(Match& match, Scene_edge& scene, cuda_icp::RegistrationResult& result, bool& valid, cv::Rect& roi)
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
		double time = cv::getTickCount();
		result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene, valid);
		time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
		std::cout << "ICP--Result耗时：" << time << std::endl;

		return cv::Mat();
	}

	cv::Mat Detector::getIcpAffineDefect(const cv::Mat image, Match& match, Scene_edge& scene, cuda_icp::RegistrationResult& result, bool& valid, cv::Rect& roi)
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
		result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene, valid);

		cv::Mat affineImg;
		cv::Mat templImg = templ[0].templImg.clone();
		if (valid)
		{
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
		}
		else
		{
			affineImg = templImg.clone();
		}

		return affineImg;
	}

	// Used to filter out weak matches
	struct MatchPredicate
	{
		MatchPredicate(float _threshold) : threshold(_threshold) {}
		bool operator()(const Match &m) { return m.similarity < threshold; }
		float threshold;
	};

	void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid, const std::vector<Size> &sizes, float threshold, std::vector<Match> &matches, const std::string &class_id,
							  const std::vector<TemplatePyramid> &template_pyramids, const float startAngle, const float endAngle) const
	{
	//#pragma omp declare reduction \
	   //(omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
	//#pragma omp parallel for reduction(omp_insert:matches)
		for (int template_id = 0; template_id < template_pyramids.size(); ++template_id)
		{
			const TemplatePyramid &tp = template_pyramids[template_id];
			// First match over the whole image at the lowest pyramid level
			/// @todo Factor this out into separate function
			const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back();

			std::vector<Match> candidates;
			// Compute similarity maps for each ColorGradient at lowest pyramid level
			Mat similarities;
			int lowest_start = static_cast<int>(tp.size() - 1);
			int lowest_T = T_at_level.back();
			int num_features = 0;
			const Template &templ = tp[lowest_start];
			float angle = templ.angle;
			if (angle > startAngle && angle < endAngle)//只检测一定范围内的，加速检测
			{
				num_features += static_cast<int>(templ.features.size());
				if (templ.features.size() < 64)
				{
					similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
					similarities.convertTo(similarities, CV_16U);
				}
				else if (templ.features.size() < 8192)
				{
					similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
				}
				else
				{
					CV_Error(Error::StsBadArg, "feature size too large");
				}

				// Find initial matches
				for (int r = 0; r < similarities.rows; ++r)
				{
					ushort *row = similarities.ptr<ushort>(r);
					for (int c = 0; c < similarities.cols; ++c)
					{
						int raw_score = row[c];
						float score = (raw_score * 100.f) / (4 * num_features);

						if (score > threshold)
						{
							float offset = lowest_T / 2 + (lowest_T % 2 - 1);
							int x = cvRound(c * lowest_T + offset);
							int y = cvRound(r * lowest_T + offset);
							candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id), templ.angle));
						}
					}
				}

				// Locally refine each match by marching up the pyramid
				for (int l = pyramid_levels - 2; l >= 0; --l)
				{
					const std::vector<LinearMemories> &lms = lm_pyramid[l];
					int T = T_at_level[l];
					int start = static_cast<int>(l);
					Size size = sizes[l];
					int border = 8 * T;
					float offset = T / 2 + (T % 2 - 1);
					int max_x = size.width - tp[start].width - border;
					int max_y = size.height - tp[start].height - border;

					Mat similarities2;
					for (int m = 0; m < (int)candidates.size(); ++m)
					{
						Match &match2 = candidates[m];
						int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
						int y = match2.y * 2 + 1;

						// Require 8 (reduced) row/cols to the up/left
						x = std::max(x, border);
						y = std::max(y, border);

						// Require 8 (reduced) row/cols to the down/left, plus the template size
						x = std::min(x, max_x);
						y = std::min(y, max_y);

						// Compute local similarity maps for each ColorGradient
						int numFeatures = 0;

						{
							const Template &templ = tp[start];
							numFeatures += static_cast<int>(templ.features.size());

							if (templ.features.size() < 64) {
								similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
								similarities2.convertTo(similarities2, CV_16U);
							}
							else if (templ.features.size() < 8192) {
								similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
							}
							else {
								CV_Error(Error::StsBadArg, "feature size too large");
							}
						}

						// Find best local adjustment
						float best_score = 0;
						int best_r = -1, best_c = -1;
						for (int r = 0; r < similarities2.rows; ++r)
						{
							ushort *row = similarities2.ptr<ushort>(r);
							for (int c = 0; c < similarities2.cols; ++c)
							{
								int score_int = row[c];
								float score = (score_int * 100.f) / (4 * numFeatures);

								if (score > best_score)
								{
									best_score = score;
									best_r = r;
									best_c = c;
								}
							}
						}
						// Update current match
						match2.similarity = best_score;
						match2.x = cvRound((x / T - 8 + best_c) * T + offset);
						match2.y = cvRound((y / T - 8 + best_r) * T + offset);
					}

					// Filter out any matches that drop below the similarity threshold
					std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
						MatchPredicate(threshold));
					candidates.erase(new_end, candidates.end());
				}

				matches.insert(matches.end(), candidates.begin(), candidates.end());
			}
		}
	}

	int Detector::addTemplate(const Mat source, const std::string &class_id,
							  const Mat &object_mask, const Mat &edge_mask, int num_features, cv::Rect* bounding_box, float angle, float scale, std::vector<cv::Point2f>& pts)
	{
		std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
		int template_id = static_cast<int>(template_pyramids.size());

		TemplatePyramid tp;
		tp.resize(pyramid_levels);
	
		// Extract a template at each pyramid level
		Ptr<ColorGradientPyramid> qp = modality->process(source, edge_mask);
		if (num_features > 0)
		{
			qp->num_features = num_features;
		}

		for (int l = 0; l < pyramid_levels; ++l)
		{
			/// @todo Could do mask subsampling here instead of in pyrDown()
			if (l > 0)
				qp->pyrDown();
			bool success = qp->extractTemplate(tp[l], pts);
			if (!success)
			{
				return -1;
			}
               
		}

		Rect bb = cropTemplates(tp);
		if (bounding_box)
		{
			*bounding_box = bb;
		}
		//add by HuangLi 2019/08/17
		if (pts.size() >= 4)
		{
			tp[0].pt0 = pts[0];
			tp[0].pt1 = pts[1];
			tp[0].pt2 = pts[2];
			tp[0].pt3 = pts[3];
		}
		for (int l = 0; l < pyramid_levels; ++l)
		{
			tp[l].angle = angle;
		}
		tp[0].scale = scale;
		tp[0].templImg = source.clone();
		tp[0].templMask = object_mask;
	
		/// @todo Can probably avoid a copy of tp here with swap
		template_pyramids.push_back(tp);
		return template_id;
	}

	static cv::Point2f rotate2d(const cv::Point2f inPoint, const double angRad)
	{
		cv::Point2f outPoint;
		//CW rotation
		outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
		outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
		return outPoint;
	}

	static cv::Point2f rotatePoint(const cv::Point2f inPoint, const cv::Point2f center, const double angRad)
	{
		return rotate2d(inPoint - center, angRad) + center;
	}

	int Detector::addTemplate_rotate(const cv::Mat& source, const cv::Mat &object_mask, const Mat &edge_mask, const string &class_id, int zero_id, float theta, cv::Point2f center,
																cv::Rect* bounding_box, float angle, float scale,  std::vector<cv::Point2f>& pts)
	{
		std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
		int template_id = static_cast<int>(template_pyramids.size());

		const auto& to_rotate_tp = template_pyramids[zero_id];

		TemplatePyramid tp;
		tp.resize(pyramid_levels);

		for (int l = 0; l < pyramid_levels; ++l)
		{
			if (l > 0) center /= 2;

			for (auto& f : to_rotate_tp[l].features)
			{
				Point2f p;
				p.x = f.x + to_rotate_tp[l].tl_x;
				p.y = f.y + to_rotate_tp[l].tl_y;
				Point2f p_rot = rotatePoint(p, center, -theta / 180 * CV_PI);

				Feature f_new;
				f_new.x = int(p_rot.x + 0.5f);
				f_new.y = int(p_rot.y + 0.5f);

				//f_new.theta = f.theta - theta;//error of author
				f_new.theta = f.theta + theta;
				while (f_new.theta > 360) f_new.theta -= 360;
				while (f_new.theta < 0) f_new.theta += 360;

				f_new.label = int(f_new.theta * 16 / 360 + 0.5f);
				f_new.label &= 7;


				tp[l].features.push_back(f_new);
			}
			tp[l].pyramid_level = l;
		}

		Rect bb = cropTemplates(tp);
		if (bounding_box)
		{
			*bounding_box = bb;
		}

		//add by HuangLi 2019/08/24
		if (pts.size() >= 4)
		{
			tp[0].pt0 = pts[0];
			tp[0].pt1 = pts[1];
			tp[0].pt2 = pts[2];
			tp[0].pt3 = pts[3];
		}
		tp[0].angle = angle;
		tp[0].scale = scale;
		tp[0].templImg = source.clone();
		tp[0].templMask = object_mask;

		template_pyramids.push_back(tp);
		return template_id;
	}

	const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
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
		for (; i != iend; ++i)
			ret += static_cast<int>(i->second.size());
		return ret;
	}

	int Detector::numTemplates(const std::string &class_id) const
	{
		TemplatesMap::const_iterator i = class_templates.find(class_id);
		if (i == class_templates.end())
			return 0;
		return static_cast<int>(i->second.size());
	}

	std::vector<std::string> Detector::classIds() const
	{
		std::vector<std::string> ids;
		TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
		for (; i != iend; ++i)
		{
			ids.push_back(i->first);
		}

		return ids;
	}

	void Detector::read(const FileNode &fn)
	{
		class_templates.clear();
		pyramid_levels = fn["pyramid_levels"];
		fn["T"] >> T_at_level;

		modality = makePtr<ColorGradient>();
	}

	void Detector::write(FileStorage &fs)
	{
		fs << "pyramid_levels" << pyramid_levels;
		fs << "T" << T_at_level;

		modality->write(fs);
	}

	std::string Detector::readClass(const FileNode &fn, const std::string &class_id_override)
	{
		// Detector should not already have this class
		String class_id;
		if (class_id_override.empty())
		{
			String class_id_tmp = fn["class_id"];
			CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
			class_id = class_id_tmp;
		}
		else
		{
			class_id = class_id_override;
		}

		TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
		std::vector<TemplatePyramid> &tps = v.second;
		int expected_id = 0;

		FileNode tps_fn = fn["template_pyramids"];
		tps.resize(tps_fn.size());
		FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
		int num = tps_fn.size();
		for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
		{
			int template_id = (*tps_it)["template_id"];
			CV_Assert(template_id == expected_id);
			FileNode templates_fn = (*tps_it)["templates"];
			tps[template_id].resize(templates_fn.size());

			FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
			int idx = 0;
			for (; templ_it != templ_it_end; ++templ_it)
			{
				tps[template_id][idx++].read(*templ_it);
			}
		}
		class_templates.insert(v);

		//读取图像
		fn["templateImg"] >> m_templateImg;
		fn["templMask"] >> m_templMask;
		return class_id;
	}

	void Detector::readClass(FileStorage &fs)
	{
		std::string class_id;
		int template_num = fs["template_num"];
		for (int i = 0; i < template_num; ++i)
		{
			class_id = cv::format("classID_%d", i);
			FileNode classID_fn = fs[class_id];
			FileNode tps_fn = classID_fn["template_pyramids"];

			TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
			std::vector<TemplatePyramid> &tps = v.second;
			int expected_id = 0;
			tps.resize(tps_fn.size());
			FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
			for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
			{
				int template_id = (*tps_it)["template_id"];
				CV_Assert(template_id == expected_id);
				FileNode templates_fn = (*tps_it)["templates"];
				tps[template_id].resize(templates_fn.size());

				FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
				int idx = 0;
				for (; templ_it != templ_it_end; ++templ_it)
				{
					tps[template_id][idx++].read(*templ_it);
				}
			}
			class_templates.insert(v);

		}
	}

	void Detector::setCallbackFunc(void(*func) (cv::Mat& src, void* psend, void* precv))
	{
		m_callbackFunc = func;
	}

	void Detector::writeClass(const std::string &class_id, FileStorage &fs)
	{
		std::string str;
		TemplatesMap::const_iterator it = class_templates.find(class_id);
		CV_Assert(it != class_templates.end());
		const std::vector<TemplatePyramid> &tps = it->second;
		str = cv::format("%s", it->first);
		fs << str << "{"; // pyramids
		fs << "pyramid_levels" << pyramid_levels;
		fs << "template_pyramids"
			<< "[";
		for (size_t i = 0; i < tps.size(); ++i)
		{
			const TemplatePyramid &tp = tps[i];
			fs << "{";
			fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
			fs << "templates"
				<< "[";
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

		//保存图像
		fs << "templateImg" << m_templateImg;
		fs << "templMask" << m_templMask;
		fs << "}"; // pyramids

	}

	int Detector::readClasses(const std::string str, std::vector<std::string> &class_ids)
	{
		cv::FileStorage fs;
		bool flag = fs.open(str, FileStorage::READ);
		//readClass(fs.root());
		//readClass(fs);
		std::string class_id;
		int template_num = fs["template_num"];
		for (int i = 0; i < template_num; ++i)
		{
			class_id = cv::format("classID_%d", i);
			FileNode classID_fn = fs[class_id];
			FileNode tps_fn = classID_fn["template_pyramids"];

			TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
			std::vector<TemplatePyramid> &tps = v.second;
			int expected_id = 0;
			tps.resize(tps_fn.size());
			FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
			for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
			{
				int template_id = (*tps_it)["template_id"];
				CV_Assert(template_id == expected_id);
				FileNode templates_fn = (*tps_it)["templates"];
				tps[template_id].resize(templates_fn.size());

				FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
				int idx = 0;
				for (; templ_it != templ_it_end; ++templ_it)
				{
					tps[template_id][idx++].read(*templ_it);
				}
			}
			class_templates.insert(v);

		}

		return template_num;
	}

	void Detector::writeClasses(const std::string str)
	{
		FileStorage fs(str, FileStorage::WRITE);
		fs << "template_num" << (int)class_templates.size();
		TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
		for (; it != it_end; ++it)
		{
			const String &class_id = it->first;
			writeClass(class_id, fs);
		}
	}

	//add by HuangLi 2019/08/14
	void Detector::createModel(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask, cv::Mat& invalidMask, std::string modelName, cv::Rect roi)
	{
		if (src.empty())
		{
			printf("Create Model Fail!!!\n");
			return;
		}

		if (roi.width == 0 || roi.height == 0)
		{
			roi = cv::Rect(0, 0, src.cols, src.rows);
		}
		cv::Mat roiImg, roiMask, roiEdgeMask;
		roiImg = src(roi).clone();
		roiMask = mask(roi).clone();
		roiEdgeMask = invalidMask(roi).clone();

		// padding to avoid rotating out
		int length = sqrt(pow(double(roi.width), 2.0) + pow(double(roi.height), 2.0));
		int top, bottom, left, right;
		top = bottom = (length - roi.height) / 2.0;
		left = right = (length - roi.width) / 2.0;
		cv::copyMakeBorder(roiImg, roiImg, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		cv::copyMakeBorder(roiMask, roiMask, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		cv::copyMakeBorder(roiEdgeMask, roiEdgeMask, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		m_templateImg = roiImg.clone();
		m_templMask = roiMask.clone();

		shape_based_matching::shapeInfo_producer shapes(roiImg, roiMask, roiEdgeMask);
		shapes.angle_range = { startAngle, endAngle };
		shapes.angle_step = stepAngle;
		shapes.eps = stepAngle * 0.5;
		shapes.produce_infos();
		std::string class_id = modelName;

		cv::Point2f center = cv::Point2f(roiImg.cols / 2, roiImg.rows / 2);
		std::vector<int> vecTemplateId(shapes.infos.size());
		std::vector<cv::Rect> vecRect(shapes.infos.size());

		double time = cv::getTickCount();
	
		bool use_rot = true;
		int first_id = 0;
		float first_angle = 0;
		bool is_fisrt = true;
		cv::Mat to_show, to_show_mask;
		cv::Mat to_show_first, to_show_mask_first;
		float angle;
		for (int i = 0; i < shapes.infos.size(); ++i)
		{
			auto info = shapes.infos[i];
			float radian;
			cv::Rect bb;
			cv::Point2f pt, pt0, pt1, pt2, pt3;
			pt0 = cv::Point2f(left, top);
			pt1 = cv::Point2f(left, top + roi.height);
			pt2 = cv::Point2f(left + roi.width, top + roi.height);
			pt3 = cv::Point2f(left + roi.width, top);
			radian = CV_PI * info.angle / 180.0;
			pt.x = pt0.x * cos(radian) + pt0.y * sin(radian) + (1 - cos(radian)) * center.x - sin(radian) * center.y;
			pt.y = -pt0.x * sin(radian) + pt0.y * cos(radian) + sin(radian) * center.x + (1 - cos(radian)) * center.y;
			pt0 = pt;
			pt.x = pt1.x * cos(radian) + pt1.y * sin(radian) + (1 - cos(radian)) * center.x - sin(radian) * center.y;
			pt.y = -pt1.x * sin(radian) + pt1.y * cos(radian) + sin(radian) * center.x + (1 - cos(radian)) * center.y;
			pt1 = pt;
			pt.x = pt2.x * cos(radian) + pt2.y * sin(radian) + (1 - cos(radian)) * center.x - sin(radian) * center.y;
			pt.y = -pt2.x * sin(radian) + pt2.y * cos(radian) + sin(radian) * center.x + (1 - cos(radian)) * center.y;
			pt2 = pt;
			pt.x = pt3.x * cos(radian) + pt3.y * sin(radian) + (1 - cos(radian)) * center.x - sin(radian) * center.y;
			pt.y = -pt3.x * sin(radian) + pt3.y * cos(radian) + sin(radian) * center.x + (1 - cos(radian)) * center.y;
			pt3 = pt;
			std::vector<cv::Point2f> pts(4);
			pts[0] = pt0;
			pts[1] = pt1;
			pts[2] = pt2;
			pts[3] = pt3;
			to_show = shapes.src_of(info);
			to_show_mask = cv::Mat::zeros(to_show.size(), CV_8UC1);
			
			int templ_id;
			templ_id = addTemplate(to_show, class_id, shapes.mask_of(info), shapes.edgeMask_of(info), 0, &bb, info.angle, info.scale, pts);
			/*if (is_fisrt)
			{
				templ_id = addTemplate(to_show, class_id, shapes.mask_of(info), shapes.edgeMask_of(info), 0, &bb, info.angle, info.scale, pts);
				if (templ_id != -1)
				{
					first_id = templ_id;
					first_angle = info.angle;
					if (use_rot)
					{
						is_fisrt = false;
					}
				}
			}
			else
			{
				templ_id = addTemplate_rotate(to_show, shapes.mask_of(info), shapes.edgeMask_of(info), class_id, first_id, info.angle - first_angle,
					cv::Point2f(shapes.src.cols / 2.0f, shapes.src.rows / 2.0f), &bb, info.angle, info.scale, pts);
			}*/
			if (templ_id != -1)
			{
				vecRect[i] = bb;
				vecTemplateId[i] = templ_id;
			}
			printf("templ_id: %d\n", templ_id);

			if (templ_id != -1)
			{
				auto templ = getTemplates(class_id, templ_id);
				for (int j = 0; j < templ[0].features.size(); j++)
				{
					auto feat = templ[0].features[j];
						cv::circle(to_show_mask, cv::Point(feat.x + templ[0].tl_x, feat.y + templ[0].tl_y), 1, cv::Scalar(255), -1);
				}
			}
			//add qt
			if (m_callbackFunc)
			{
				//m_callbackFunc(to_show, NULL, NULL);
			}

			//cv::Mat tq = to_show + to_show_mask;
		
			if (i == 0)
			{
				to_show_first = to_show.clone();
				to_show_mask_first = to_show_mask.clone();
				angle = info.angle;
			}
		}
		time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
		printf("多尺寸角度建模耗时: %lf毫秒\n", time);

		//Judge whether NULL
		bool bExist = false;
		int value, index = -1;
		for (int i = 0; i < vecTemplateId.size(); ++i)
		{
			value = vecTemplateId[i];
			if (value > -1 && index < 0)
			{
				bExist = true;
				index = i;
			}
		}

		if (!bExist)
		{
			dst = src.clone();
			printf("Create Template Error!!!\n");
			return;
		}

		/***********************************保存模板***************************************/
		//判断是否存在template文件夹
		/*_mkdir("./template");
		writeClasses("./template/%s.yaml");*/
		bExitModel = true;
		/***********************************显示模板点***************************************/
		if (to_show.channels() == 3)
		{
			dst = to_show_first.clone();
			dst.setTo(cv::Scalar(0, 255, 0), to_show_mask_first);
		}
		else
		{
			cv::cvtColor(to_show_first, dst, cv::COLOR_GRAY2BGR);
			dst.setTo(cv::Scalar(0, 255, 0), to_show_mask_first);
		}
		cv::Point2f pt, pt0, pt1, pt2, pt3;
		pt0 = cv::Point2f(left, top);
		pt1 = cv::Point2f(left, top + roi.height);
		pt2 = cv::Point2f(left + roi.width, top + roi.height);
		pt3 = cv::Point2f(left + roi.width, top);

		float radian;
		radian = CV_PI * angle / 180.0;
		pt.x = pt0.x * cos(radian) + pt0.y * sin(radian) + (1 - cos(radian)) * center.x - sin(radian) * center.y;
		pt.y = -pt0.x * sin(radian) + pt0.y * cos(radian) + sin(radian) * center.x + (1 - cos(radian)) * center.y;
		pt0 = pt;
		pt.x = pt1.x * cos(radian) + pt1.y * sin(radian) + (1 - cos(radian)) * center.x - sin(radian) * center.y;
		pt.y = -pt1.x * sin(radian) + pt1.y * cos(radian) + sin(radian) * center.x + (1 - cos(radian)) * center.y;
		pt1 = pt;
		pt.x = pt2.x * cos(radian) + pt2.y * sin(radian) + (1 - cos(radian)) * center.x - sin(radian) * center.y;
		pt.y = -pt2.x * sin(radian) + pt2.y * cos(radian) + sin(radian) * center.x + (1 - cos(radian)) * center.y;
		pt2 = pt;
		pt.x = pt3.x * cos(radian) + pt3.y * sin(radian) + (1 - cos(radian)) * center.x - sin(radian) * center.y;
		pt.y = -pt3.x * sin(radian) + pt3.y * cos(radian) + sin(radian) * center.x + (1 - cos(radian)) * center.y;
		pt3 = pt;
		cv::line(dst, pt0, pt1, cv::Scalar(255, 0, 0), 1);
		cv::line(dst, pt1, pt2, cv::Scalar(255, 0, 0), 1);
		cv::line(dst, pt2, pt3, cv::Scalar(255, 0, 0), 1);
		cv::line(dst, pt3, pt0, cv::Scalar(255, 0, 0), 1);
	}

	int Detector::saveModel(string path)
	{
		//判断是否已经产生模板
		if (!bExitModel)
		{
			return 0;
		}
		/***********************************保存模板***************************************/
		writeClasses(path);
	
		return 1;
	}

	//add by HuangLi 2019/08/14
	void Detector::detect(cv::Mat &src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, int amount, cv::Mat &mask)
	{
		/*
		double time;
		if (src.channels() == 1)
		{
			cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
		}
		else
		{
			dst = src.clone();
		}

		time = cv::getTickCount();
		std::vector<std::string> ids;
		std::map<std::string, std::vector<Match> > mapMatches;
		mapMatches = match(src, matchThreshold, ids);
		time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
		cout << "加速版--匹配耗时：" << time << endl;

		int num;
		std::vector<Match> matches;
		std::map<string, std::vector<Match> >::const_iterator it = mapMatches.begin(), itend = mapMatches.end();
		for (; it != itend; ++it)
		{
			num = mapMatches[it->first].size();
			if (num > amount)
			{
				mapMatches[it->first].resize(amount);
			}
		}

		Scene_edge scene;
		scene = initScene(src);

		findObjects.resize(matches.size());
		time = cv::getTickCount();
	#pragma omp parallel for
		for (int i = 0; i < (int)matches.size(); ++i)
		{
			auto match = matches[i];
			cv::Mat affineImg;
			cuda_icp::RegistrationResult result;
			bool valid = false;
			cv::Rect roi;
			affineImg = getIcpAffineDefect(src, match, scene, result, valid, roi);
			auto templates = getTemplates(match.class_id, match.template_id);
			if (valid)
			{
				drawResponse(templates, 1, dst, cv::Point(match.x, match.y), result);
				//cout << "-----------匹配序号: "<< i << " ICP有效" << endl;
			}
			else
			{
				drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
				//cout << ">>>>>>>>>>>匹配序号: " << i << " ICP无效" << endl;
			}
			//findObjects[i] = { roi, templates[0].angle, match.similarity, affineImg };
		}
		time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
		cout << "加速版-ICP及点绘制耗时: " << time << endl;
		//add
		{
			cv::Mat edge;
			cv::Mat edge1 = dst.clone();
			if (src.channels() == 1)
			{
				cv::cvtColor(src, edge, cv::COLOR_GRAY2BGR);
			}
			else
			{
				edge = src.clone();
			}
			Scene_edge scene;
			vector<::Vec2f> pcd_buffer, normal_buffer;
			scene.init_Scene_edge_cpu(src, pcd_buffer, normal_buffer);
			for (int i = 0; i < (int)matches.size(); ++i)
			{
				auto match = matches[i];
				auto templ = getTemplates(match.class_id, match.template_id);
				vector<::Vec2f> model_pcd(templ[0].features.size());
				for (int i = 0; i < templ[0].features.size(); i++) {
					auto& feat = templ[0].features[i];
					model_pcd[i] = {
						float(feat.x + match.x),
						float(feat.y + match.y)
					};
				}
				bool valid = true;;
				cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene, valid);
				cv::Vec3b randColor;
				randColor[0] = 0;
				randColor[1] = 255;
				randColor[2] = 0;
				for (int i = 0; i < templ[0].features.size(); i++) {
					auto feat = templ[0].features[i];
					float x = feat.x + match.x;
					float y = feat.y + match.y;
					float new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
					float new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];

					cv::circle(edge, { int(new_x + 0.5f), int(new_y + 0.5f) }, 2, randColor, 1);
				}

			}
		}
		*/
	}

	void Detector::getFeatures(const std::vector<Template>& templates, int num_modalities, std::vector<cv::Point>& pts, cv::Point offset)
	{
		pts.clear();
		for (int m = 0; m < num_modalities; ++m)
		{
			for (int i = 0; i < (int)templates[m].features.size(); ++i)
			{
				auto feat = templates[m].features[i];
				float x = feat.x + offset.x;
				float y = feat.y + offset.y;
				pts.push_back(cv::Point(int(x + 0.5f), int(y + 0.5f)));
			}
		}
	}

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
				cv::circle(dst, { int(x + 0.5f), int(y + 0.5f) }, 1, color, -1);
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
				cv::circle(dst, { int(new_x + 0.5f), int(new_y + 0.5f) }, 1, color, -1);
			}
		}
	}

	void Detector::getCurrentTemplImg(Template& templ)
	{
		float angle = templ.angle;
		float scale = templ.scale;
		cv::Point2f center(m_templateImg.cols / 2.0f, m_templateImg.rows / 2.0f);
		cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
		cv::warpAffine(m_templateImg, templ.templImg, rot_mat, m_templateImg.size());
		cv::warpAffine(m_templMask, templ.templMask, rot_mat, m_templMask.size());
	}

	static float coarsePosition(cv::Mat& src, cv::Mat& templateImg, cv::Rect& roi)
	{
		int result_cols = src.cols - templateImg.cols + 1;
		int result_rows = src.rows - templateImg.rows + 1;
		if (result_cols < 0 || result_rows < 0)
		{
			roi = cv::Rect();
			return 0;
		}
		cv::Mat dst(result_rows, result_cols, CV_32FC1);
		//matchTemplate(grayImg, templGrayImg, dst, cv::TM_CCOEFF_NORMED);
		matchTemplate(src, templateImg, dst, cv::TM_CCOEFF_NORMED);

		double minVal, maxVal;
		cv::Point minLoc, maxLoc, matchLoc;
		minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
		matchLoc = maxLoc;
		//rectangle(src, matchLoc, cv::Point(matchLoc.x + templateImg.cols, matchLoc.y + templateImg.rows), cv::Scalar::all(255), 2, 8, 0);
		roi = cv::Rect(matchLoc.x, matchLoc.y, templateImg.cols, templateImg.rows);

		return maxVal;
	}

	void Detector::detectPose(cv::Mat& src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, int amount, cv::Mat& mask)
	{
		/*
		double time;
		int val = T_at_level[0] * T_at_level[1] * 2;//代码内限制  CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0); mipp::N<uint8_t>()=32
		int width = (src.cols % val) == 0 ? src.cols : ((src.cols / val + 1) * val);
		int height = (src.rows % val) == 0 ? src.rows : ((src.rows / val + 1) * val);
		//图像不是一定要2的n次方，要的是设置的4*1 8*2的倍数，是因为后面内存重排的需要。比如设成5 8就需要5*1 8*2的倍数，也就是80的倍数。

		cv::Mat borderImage, borderMask;
		cv::copyMakeBorder(src, borderImage, 0, height - src.rows, 0, width - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		if (!mask.empty())
		{
			cv::copyMakeBorder(mask, borderMask, 0, height - src.rows, 0, width - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		}
		
		if (src.channels() == 1)
		{
			cv::cvtColor(borderImage, dst, cv::COLOR_GRAY2BGR);
		}
		else
		{
			dst = borderImage.clone();
		}

		std::vector<std::string> ids;
		std::map<std::string, std::vector<Match> > mapMatches;
		if (!mask.empty())
		{
			mapMatches = match(borderImage, matchThreshold, ids, borderMask);
		}
		else
		{
			mapMatches = match(borderImage, matchThreshold, ids);
		}

		int num;
		std::map<string, std::vector<Match> >::const_iterator it = mapMatches.begin(), itend = mapMatches.end();
		for (; it != itend; ++it)
		{
			num = mapMatches[it->first].size();
			if (num > amount)
			{
				mapMatches[it->first].resize(amount);
			}
		}

		//ICP
		Scene_edge scene;
		scene = initScene(borderImage);
		bool bIcp = getIcp();
		bIcp = true;

		std::string classID;
		std::vector<Match> matches;
		it = mapMatches.begin(), itend = mapMatches.end();
		for (; it != itend; ++it)
		{
			classID = it->first;
			matches = it->second;
			std::vector<FindObject> tempFindObject;
			if (bIcp)
			{
#pragma omp parallel for
				for (int i = 0; i < (int)matches.size(); ++i)
				{
					Match match = matches[i];
					std::vector<Template> templates = getTemplates(match.class_id, match.template_id);
					cuda_icp::RegistrationResult result;
					bool valid = false;
					double time1 = cv::getTickCount();
					getIcpAffine(match, scene, result, valid);
					time1 = (cv::getTickCount() - time1) * 1000.0 / cv::getTickFrequency();
					//ICP angle and scale no use now
					//add by HuangLi 2019/08/23
					double icp_angle = std::atan(result.transformation_[1][0] / result.transformation_[0][0]) / CV_PI * 180;
					double icp_scale = sqrt((result.transformation_[0][0])*(result.transformation_[0][0]) + (result.transformation_[1][0]) * (result.transformation_[1][0]));
					std::vector<cv::Point2f> pts(4);
					Template templ;
					float x, y, new_x, new_y;
					if (valid)
					{
						drawResponse(templates, 1, dst, cv::Point(match.x, match.y), result);
						templ = templates[0];
						cv::Point2f pt;
						pt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
						pts[0] = cv::Point2f(templ.pt0) + pt;
						pts[1] = cv::Point2f(templ.pt1) + pt;
						pts[2] = cv::Point2f(templ.pt2) + pt;
						pts[3] = cv::Point2f(templ.pt3) + pt;
						//ICP
						for (int j = 0; j < pts.size(); ++j)
						{
							pt = pts[j];
							x = pt.x;
							y = pt.y;
							new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
							new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
							pts[j] = cv::Point2f(new_x, new_y);
						}
						cv::line(dst, pts[0], pts[1], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[1], pts[2], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[2], pts[3], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[3], pts[0], cv::Scalar(0, 0, 255));

						//draw contours
						if (bDrawContour)
						{
							cv::Mat edgeImg;
							cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
							for (int row = 0; row < edgeImg.rows; row++)
							{
								uchar* ptr = edgeImg.ptr<uchar>(row);
								for (int col = 0; col < edgeImg.cols; col++)
								{
									if (ptr[col])
									{
										x = col - templ.tl_x + match.x;
										y = row - templ.tl_y + match.y;
										new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
										new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
										dst.at<Vec3b>(new_y, new_x) = Vec3b(255, 0, 0);
									}
								}
							}
						}
					}
					else
					{
						drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
						templ = templates[0];
						cv::Point2f pt;
						pt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
						pts[0] = cv::Point2f(templ.pt0) + pt;
						pts[1] = cv::Point2f(templ.pt1) + pt;
						pts[2] = cv::Point2f(templ.pt2) + pt;
						pts[3] = cv::Point2f(templ.pt3) + pt;
						cv::line(dst, pts[0], pts[1], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[1], pts[2], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[2], pts[3], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[3], pts[0], cv::Scalar(0, 0, 255));

						//draw contours
						if (bDrawContour)
						{
							cv::Mat edgeImg;
							cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
							for (int row = 0; row < edgeImg.rows; row++)
							{
								uchar* ptr = edgeImg.ptr<uchar>(row);
								for (int col = 0; col < edgeImg.cols; col++)
								{
									if (ptr[col])
									{
										x = col - templ.tl_x + match.x;
										y = row - templ.tl_y + match.y;
										dst.at<Vec3b>(y, x) = Vec3b(255, 0, 0);
									}
								}
							}
						}

					}

					findObjects[i] = { pts, templ.angle, match.similarity };
				}
			}
			else
			{
#pragma omp parallel for
				for (int i = 0; i < (int)matches.size(); ++i)
				{
					Match match = matches[i];
					std::vector<Template> templates = getTemplates(match.class_id, match.template_id);
					Template templ = templates[0];
					drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
					cv::Point2f pt;
					pt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
					std::vector<cv::Point2f> pts(4);
					pts[0] = cv::Point2f(templ.pt0) + pt;
					pts[1] = cv::Point2f(templ.pt1) + pt;
					pts[2] = cv::Point2f(templ.pt2) + pt;
					pts[3] = cv::Point2f(templ.pt3) + pt;
					cv::line(dst, pts[0], pts[1], cv::Scalar(0, 0, 255));
					cv::line(dst, pts[1], pts[2], cv::Scalar(0, 0, 255));
					cv::line(dst, pts[2], pts[3], cv::Scalar(0, 0, 255));
					cv::line(dst, pts[3], pts[0], cv::Scalar(0, 0, 255));

					//draw contours
					if (bDrawContour)
					{
						float x, y;
						cv::Mat edgeImg;
						cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
						for (int row = 0; row < edgeImg.rows; row++)
						{
							uchar* ptr = edgeImg.ptr<uchar>(row);
							for (int col = 0; col < edgeImg.cols; col++)
							{
								if (ptr[col])
								{
									x = col - templ.tl_x + match.x;
									y = row - templ.tl_y + match.y;
									dst.at<Vec3b>(y, x) = Vec3b(255, 0, 0);
								}
							}
						}
					}
					findObjects[i] = { pts, templ.angle, match.similarity };
				}

			}
		}

		dst = dst(cv::Rect(0, 0, src.cols, src.rows));
		*/
	}

	void Detector::detectPoseAndDefect(cv::Mat& src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, int amount, cv::Mat &mask)
	{
		/*
		double time;
		int val = T_at_level[0] * T_at_level[1] * 2;//代码内限制  CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0); mipp::N<uint8_t>()=32
		int width = (src.cols % val) == 0 ? src.cols : ((src.cols / val + 1) * val);
		int height = (src.rows % val) == 0 ? src.rows : ((src.rows / val + 1) * val);
		//图像不是一定要2的n次方，要的是设置的4 * 1 8 * 2的倍数，是因为后面内存重排的需要。比如设成5 8就需要5 * 1 8 * 2的倍数，也就是80的倍数。

		cv::Mat borderImage;
		cv::copyMakeBorder(src, borderImage, 0, height - src.rows, 0, width - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		if (src.channels() == 1)
		{
			cv::cvtColor(borderImage, dst, cv::COLOR_GRAY2BGR);
		}
		else
		{
			dst = borderImage.clone();
		}

		time = cv::getTickCount();
		std::vector<std::string> ids;
		std::vector<std::vector<Match> > candMatches;
		std::map<std::string, std::vector<Match> > mapMatches;
		if (!mask.empty())
		{
			mapMatches = match(borderImage, matchThreshold, ids);
		}
		else
		{
			mapMatches = match(borderImage, matchThreshold, ids);
		}
		time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
		cout << "加速版--匹配耗时：" << time << endl;

		if (candMatches.size() == 0)
		{
			dst = dst(cv::Rect(0, 0, src.cols, src.rows));
			return;
		}
		if (amount < candMatches.size())
		{
			candMatches.resize(amount);
		}
		std::vector<Match> matches;
		for (int i = 0; i < candMatches.size(); ++i)
		{
			matches.push_back(candMatches[i][0]);
		}

		Scene_edge scene;
		scene = initScene(borderImage);

		findObjects.resize(matches.size());
		bool bIcp = getIcp();
		bIcp = true;
		time = cv::getTickCount();
		cv::Mat affineImg = cv::Mat::zeros(borderImage.size(), CV_8UC1);
		cv::Mat affineMask = cv::Mat::zeros(borderImage.size(), CV_8UC1);
		if (bIcp)
		{
	#pragma omp parallel for
			for (int i = 0; i < (int)matches.size(); ++i)
			{
				Match match = matches[i];
				std::vector<Template> templates = getTemplates(match.class_id, match.template_id);
				cuda_icp::RegistrationResult result;
				bool valid = false;
				double time1 = cv::getTickCount();
				getIcpAffine(match, scene, result, valid);
				time1 = (cv::getTickCount() - time1) * 1000.0 / cv::getTickFrequency();
				std::cout << "加速版-ICP耗时: " << time1 << endl;
				//ICP angle and scale no use now
				//add by HuangLi 2019/08/23
				double icp_angle = std::atan(result.transformation_[1][0] / result.transformation_[0][0]) / CV_PI * 180;
				double icp_scale = sqrt((result.transformation_[0][0])*(result.transformation_[0][0]) + (result.transformation_[1][0]) * (result.transformation_[1][0]));
				std::vector<cv::Point2f> pts(4);
				Template templ;
				float x, y, new_x, new_y;
				cv::Point2f referPt;
				if (0)//采用模板变换到原图
				{
					if (valid)
					{
						if (bDrawContour)
						{
							drawResponse(templates, 1, dst, cv::Point(match.x, match.y), result);
						}
						templ = templates[0];
						cv::Point2f pt;
						referPt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
						pts[0] = cv::Point2f(templ.pt0) + referPt;
						pts[1] = cv::Point2f(templ.pt1) + referPt;
						pts[2] = cv::Point2f(templ.pt2) + referPt;
						pts[3] = cv::Point2f(templ.pt3) + referPt;
						//ICP
						for (int j = 0; j < pts.size(); ++j)
						{
							pt = pts[j];
							x = pt.x;
							y = pt.y;
							new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
							new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
							pts[j] = cv::Point2f(new_x, new_y);
						}
						cv::line(dst, pts[0], pts[1], cv::Scalar(255, 0, 0));
						cv::line(dst, pts[1], pts[2], cv::Scalar(255, 0, 0));
						cv::line(dst, pts[2], pts[3], cv::Scalar(255, 0, 0));
						cv::line(dst, pts[3], pts[0], cv::Scalar(255, 0, 0));

						//draw contours
						if (bDrawContour)
						{
							cv::Mat edgeImg;
							cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
							for (int row = 0; row < edgeImg.rows; row++)
							{
								uchar* ptr = edgeImg.ptr<uchar>(row);
								for (int col = 0; col < edgeImg.cols; col++)
								{
									if (ptr[col])
									{
										x = col - templ.tl_x + match.x;
										y = row - templ.tl_y + match.y;
										new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
										new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
										dst.at<Vec3b>(new_y, new_x) = Vec3b(255, 0, 0);
									}
								}
							}
						}

						//模板变换后的图像,采用投射标变换
						cv::Mat templImg = templ.templImg;
						cv::Mat templMask = templ.templMask;
						cv::Mat affineTempl;
						if (0)//原始写法
						{
							for (int row = 0; row < templImg.rows; row++)
							{
								uchar* ptr = templImg.ptr<uchar>(row);
								uchar* ptrMask = templMask.ptr<uchar>(row);
								for (int col = 0; col < templImg.cols; col++)
								{
									if (ptr[col])
									{
										x = col - templ.tl_x + match.x;
										y = row - templ.tl_y + match.y;
										new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
										new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
										affineImg.at<uchar>(new_y, new_x) = ptr[col];
										affineMask.at<uchar>(new_y, new_x) = ptrMask[col];
									}
								}
							}

							//去除彷设变换噪点
							int an = 1;
							cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(an * 2 + 1, an * 2 + 1), cv::Point(an, an));
							cv::morphologyEx(affineMask, affineMask, cv::MORPH_CLOSE, element);
						}
						else if (1)
						{
							//透射变换
							float A[3][3] = { {result.transformation_[0][0], result.transformation_[0][1], result.transformation_[0][2] + match.x - templ.tl_x},
											  {result.transformation_[1][0], result.transformation_[1][1], result.transformation_[1][2] + match.y - templ.tl_y},
											  {result.transformation_[2][0], result.transformation_[2][1], result.transformation_[2][2]} };
							cv::Mat warpMatrix = cv::Mat(3, 3, CV_32FC1, A);
							warpPerspective(templImg, affineImg, warpMatrix, borderImage.size(), INTER_LINEAR, BORDER_CONSTANT);
							warpPerspective(templMask, affineMask, warpMatrix, borderImage.size(), INTER_LINEAR, BORDER_CONSTANT);
						}
					}
					else
					{
						drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
						templ = templates[0];
						cv::Point2f pt;
						pt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
						pts[0] = cv::Point2f(templ.pt0) + pt;
						pts[1] = cv::Point2f(templ.pt1) + pt;
						pts[2] = cv::Point2f(templ.pt2) + pt;
						pts[3] = cv::Point2f(templ.pt3) + pt;
						cv::line(dst, pts[0], pts[1], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[1], pts[2], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[2], pts[3], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[3], pts[0], cv::Scalar(0, 0, 255));

						//draw contours
						if (bDrawContour)
						{
							cv::Mat edgeImg;
							cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
							for (int row = 0; row < edgeImg.rows; row++)
							{
								uchar* ptr = edgeImg.ptr<uchar>(row);
								for (int col = 0; col < edgeImg.cols; col++)
								{
									if (ptr[col])
									{
										x = col - templ.tl_x + match.x;
										y = row - templ.tl_y + match.y;
										dst.at<Vec3b>(y, x) = Vec3b(255, 0, 0);
									}
								}
							}
						}

						//模板变换后的图像
						cv::Mat templImg = templ.templImg;
						cv::Mat templMask = templ.templMask;
						for (int row = 0; row < templImg.rows; row++)
						{
							uchar* ptr = templImg.ptr<uchar>(row);
							uchar* ptrMask = templMask.ptr<uchar>(row);
							for (int col = 0; col < templImg.cols; col++)
							{
								if (ptr[col])
								{
									x = col - templ.tl_x + match.x;
									y = row - templ.tl_y + match.y;
									affineImg.at<uchar>(y, x) = ptr[col];
									affineMask.at<uchar>(y, x) = ptrMask[col];
								}
							}
						}

					}
					affineImg = affineImg(cv::Rect(0, 0, src.cols, src.rows));
					findObjects[i] = { pts, templates[0].angle, match.similarity, affineImg };

					//计算缺陷
					cv::Mat surfaceImg, gapImg, breakImg, defectImg;
					std::vector<cv::Rect> defectRois;
					defectImg = cv::Mat::zeros(borderImage.size(), CV_8UC1);
					cv::Rect roi = cv::boundingRect(pts);
					cv::Mat roiAffineImg, roiImg, roiMask, binary;
					roiAffineImg = affineImg(roi);
					roiImg = src(roi);
					roiMask = affineMask(roi);
					//表面缺陷
					calSurface(roiImg, roiAffineImg, roiMask, surfaceImg, defectRois, cv::Point(roi.x, roi.y));
					if (surfaceImg.size() == roi.size())
					{
						defectImg(roi) += surfaceImg;
					}
					//裂缝
					calGap(roiImg, roiMask, gapImg, defectRois, cv::Point(roi.x, roi.y));
					defectImg(roi) += gapImg;

					//显示缺陷
					for (int i = 0; i < defectRois.size(); ++i)
					{
						cv::rectangle(dst, defectRois[i], cv::Scalar(0, 0, 255), 2);
					}
					dst.setTo(cv::Scalar(0, 255, 255), defectImg);
				}
				else//采用原图变换到模板
				{
					drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
					templ = templates[0];
					cv::Point2f referPt;
					referPt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
					pts[0] = cv::Point2f(templ.pt0) + referPt;
					pts[1] = cv::Point2f(templ.pt1) + referPt;
					pts[2] = cv::Point2f(templ.pt2) + referPt;
					pts[3] = cv::Point2f(templ.pt3) + referPt;
					cv::line(dst, pts[0], pts[1], cv::Scalar(0, 0, 255));
					cv::line(dst, pts[1], pts[2], cv::Scalar(0, 0, 255));
					cv::line(dst, pts[2], pts[3], cv::Scalar(0, 0, 255));
					cv::line(dst, pts[3], pts[0], cv::Scalar(0, 0, 255));
					//draw contours
					if (bDrawContour)
					{
						cv::Mat edgeImg;
						cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
						for (int row = 0; row < edgeImg.rows; row++)
						{
							uchar* ptr = edgeImg.ptr<uchar>(row);
							for (int col = 0; col < edgeImg.cols; col++)
							{
								if (ptr[col])
								{
									x = col - templ.tl_x + match.x;
									y = row - templ.tl_y + match.y;
									dst.at<Vec3b>(y, x) = Vec3b(255, 0, 0);
								}
							}
						}
					}
					getCurrentTemplImg(templ);
					cv::Mat templImg = templ.templImg;
					cv::Mat templMask = templ.templMask;
					cv::Mat roiImg, roiMask, roiTempl;
					cv::Rect roi = cv::boundingRect(pts);
					cv::Rect drawRoi;
					if (valid)
					{
						pts[0] = cv::Point2f(templ.pt0);
						pts[1] = cv::Point2f(templ.pt1);
						pts[2] = cv::Point2f(templ.pt2);
						pts[3] = cv::Point2f(templ.pt3);

						//模板变换后的图像,采用投射标变换（逆变换）
						vector<Point2f> src_coners;
						vector<Point2f> dst_coners;
						cv::Point2f pt1, pt2, pt3, pt4;
						cv::Point2f pt1_, pt2_, pt3_, pt4_;
						pt1 = cv::Point2f(0, 0);
						x = pt1.x - templ.tl_x + match.x;
						y = pt1.y - templ.tl_y + match.y;
						new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
						new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
						pt1_ = cv::Point2f(new_x, new_y);

						pt2 = cv::Point2f(0, templImg.rows);
						x = pt2.x - templ.tl_x + match.x;
						y = pt2.y - templ.tl_y + match.y;
						new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
						new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
						pt2_ = cv::Point2f(new_x, new_y);

						pt3 = cv::Point2f(templImg.cols, 0);
						x = pt3.x - templ.tl_x + match.x;
						y = pt3.y - templ.tl_y + match.y;
						new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
						new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
						pt3_ = cv::Point2f(new_x, new_y);

						pt4 = cv::Point2f(templImg.cols, templImg.rows);
						x = pt4.x - templ.tl_x + match.x;
						y = pt4.y - templ.tl_y + match.y;
						new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
						new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
						pt4_ = cv::Point2f(new_x, new_y);

						src_coners.push_back(pt1);
						src_coners.push_back(pt2);
						src_coners.push_back(pt3);
						src_coners.push_back(pt4);

						dst_coners.push_back(pt1_);
						dst_coners.push_back(pt2_);
						dst_coners.push_back(pt3_);
						dst_coners.push_back(pt4_);
						cv::Mat warpMatrix = getPerspectiveTransform(dst_coners, src_coners);
						warpPerspective(borderImage, affineImg, warpMatrix, templImg.size(), INTER_LINEAR, BORDER_CONSTANT);

						cv::Rect tempRect;
						tempRect = roi - cv::Point(referPt.x, referPt.y);
						roiImg = affineImg(tempRect);
						roiMask = templ.templMask(tempRect);
						roiTempl = templImg(tempRect);
						drawRoi = tempRect + cv::Point(referPt.x, referPt.y);
					}
					else
					{
						//模板变换后的图像
						if (roi.x < 0 || roi.x + roi.width >= borderImage.cols || roi.y < 0 || roi.y + roi.height >= borderImage.rows)
						{
							continue;
						}
						affineImg = borderImage(roi).clone();
						cv::Rect tempRect;
						tempRect = roi - cv::Point(referPt.x, referPt.y);
						roiImg = affineImg;
						roiMask = templMask(tempRect);
						roiTempl = templImg(tempRect);
						drawRoi = roi;
					}
					findObjects[i] = { pts, templates[0].angle, match.similarity, affineImg };

					//计算缺陷
					cv::Mat surfaceImg, gapImg, breakImg, defectImg;
					std::vector<cv::Rect> defectRois;
					defectImg = cv::Mat::zeros(roiImg.size(), CV_8UC1);
					//表面缺陷
					calSurface(roiImg, roiTempl, roiMask, surfaceImg, defectRois, cv::Point(0, 0));
					if (surfaceImg.size() == roi.size())
					{
						defectImg += surfaceImg;
					}

					//显示缺陷
					for (int i = 0; i < defectRois.size(); ++i)
					{
						cv::rectangle(dst(drawRoi), defectRois[i], cv::Scalar(0, 0, 255), 2);
					}
					dst(drawRoi).setTo(cv::Scalar(0, 255, 255), defectImg);
				}
			}
		}
		else
		{
#pragma omp parallel for
			for (int i = 0; i < (int)matches.size(); ++i)
			{
				Match match = matches[i];
				std::vector<Template> templates = getTemplates(match.class_id, match.template_id);
				cuda_icp::RegistrationResult result;
				bool valid = false;
				double time1 = cv::getTickCount();
				getIcpAffine(match, scene, result, valid);
				time1 = (cv::getTickCount() - time1) * 1000.0 / cv::getTickFrequency();
				std::cout << "加速版-ICP耗时: " << time1 << endl;
				//ICP angle and scale no use now
				//add by HuangLi 2019/08/23
				double icp_angle = std::atan(result.transformation_[1][0] / result.transformation_[0][0]) / CV_PI * 180;
				double icp_scale = sqrt((result.transformation_[0][0])*(result.transformation_[0][0]) + (result.transformation_[1][0]) * (result.transformation_[1][0]));
				std::vector<cv::Point2f> pts(4);
				Template templ;
				float x, y, new_x, new_y;
				cv::Point2f referPt;
				drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
				templ = templates[0];
				cv::Point2f pt;
				pt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
				pts[0] = cv::Point2f(templ.pt0) + pt;
				pts[1] = cv::Point2f(templ.pt1) + pt;
				pts[2] = cv::Point2f(templ.pt2) + pt;
				pts[3] = cv::Point2f(templ.pt3) + pt;
				cv::line(dst, pts[0], pts[1], cv::Scalar(0, 0, 255));
				cv::line(dst, pts[1], pts[2], cv::Scalar(0, 0, 255));
				cv::line(dst, pts[2], pts[3], cv::Scalar(0, 0, 255));
				cv::line(dst, pts[3], pts[0], cv::Scalar(0, 0, 255));

				//draw contours
				if (bDrawContour)
				{
					cv::Mat edgeImg;
					cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
					for (int row = 0; row < edgeImg.rows; row++)
					{
						uchar* ptr = edgeImg.ptr<uchar>(row);
						for (int col = 0; col < edgeImg.cols; col++)
						{
							if (ptr[col])
							{
								x = col - templ.tl_x + match.x;
								y = row - templ.tl_y + match.y;
								if (x >= 0 && x < dst.cols && y >= 0 && y < dst.rows)
								{
									dst.at<Vec3b>(y, x) = Vec3b(255, 0, 0);
								}
								
							}
						}
					}
				}

				//模板变换后的图像
				cv::Mat templImg = templ.templImg;
				cv::Mat templMask = templ.templMask;
				cv::Rect roi = cv::boundingRect(pts);
				if (roi.x < 0 || roi.x + roi.width >= borderImage.cols || roi.y < 0 || roi.y + roi.height >= borderImage.rows)
				{
					continue;
				}
				affineImg = borderImage(roi);
				findObjects[i] = { pts, templates[0].angle, match.similarity, affineImg };

				//计算缺陷
				cv::Mat surfaceImg, gapImg, breakImg, defectImg;
				std::vector<cv::Rect> defectRois;
				cv::Mat roiAffineImg, roiImg, roiMask, roiTempl, binary;
				roiImg = affineImg;
				roiTempl = templ.templImg(roi - cv::Point(pt.x, pt.y));
				roiMask = templ.templMask(roi - cv::Point(pt.x, pt.y));

				defectImg = cv::Mat::zeros(roiImg.size(), CV_8UC1);
				//表面缺陷
				calSurface(roiImg, roiTempl, roiMask, surfaceImg, defectRois, cv::Point(0, 0));
				if (surfaceImg.size() == roi.size())
				{
					defectImg += surfaceImg;
				}

				//显示缺陷
				for (int i = 0; i < defectRois.size(); ++i)
				{
					cv::rectangle(dst(roi), defectRois[i], cv::Scalar(0, 0, 255), 2);
				}
				dst(roi).setTo(cv::Scalar(0, 255, 255), defectImg);

				//int tq = 100;
			}
		}
		dst = dst(cv::Rect(0, 0, src.cols, src.rows));
		time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
		std::cout << "加速版-ICP及点绘制耗时: " << time << std::endl;
		*/
	}

	int Detector::detectObject(cv::Mat& src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, std::vector<std::string>& classIds, 
																									float startAngle, float endAngle, int amount, cv::Mat& mask)
	{
		cv::Mat grayImg;
		if (src.channels() == 3)
		{
			cv::cvtColor(src, grayImg, cv::COLOR_RGB2GRAY);
		}
		else
		{
			grayImg = src;
		}

		double time;
		int val = T_at_level[0] * T_at_level[1] * 2;//代码内限制  CV_Assert((grayImg.rows * grayImg.cols) % mipp::N<uint8_t>() == 0); mipp::N<uint8_t>()=32
		int width = (grayImg.cols % val) == 0 ? grayImg.cols : ((grayImg.cols / val + 1) * val);
		int height = (grayImg.rows % val) == 0 ? grayImg.rows : ((grayImg.rows / val + 1) * val);
		//图像不是一定要2的n次方，要的是设置的4*1 8*2的倍数，是因为后面内存重排的需要。比如设成5 8就需要5*1 8*2的倍数，也就是80的倍数。

		cv::Mat borderImage, borderMask;
		cv::copyMakeBorder(grayImg, borderImage, 0, height - grayImg.rows, 0, width - grayImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		if (!mask.empty())
		{
			cv::copyMakeBorder(mask, borderMask, 0, height - grayImg.rows, 0, width - grayImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		}

		if (grayImg.channels() == 1)
		{
			cv::cvtColor(borderImage, dst, cv::COLOR_GRAY2BGR);
		}
		else
		{
			dst = borderImage.clone();
		}

		time = cv::getTickCount();
		std::map<string, std::vector<Match> > mapMatches;
		if (!borderMask.empty())
		{
			mapMatches = match(borderImage, matchThreshold, classIds, borderMask, startAngle, endAngle);
		}
		else
		{
			mapMatches = match(borderImage, matchThreshold, classIds, cv::Mat(), startAngle, endAngle);
		}
		//判断检测是否成功

		//对于每一个类别，求前N大的值
		std::map<string, std::vector<Match> >::const_iterator it = mapMatches.begin(), itend = mapMatches.end();
		std::vector<Match> tmplMatches;
		int num;
		bool bFlag = false;
		for (; it != itend; ++it)
		{
			num = mapMatches[it->first].size();
			if (num > amount)
			{
				mapMatches[it->first].resize(amount);
			}
			else if (num == 0)
			{
				bFlag = true;
			}
		}
		if (bFlag)
		{
			return 0;
		}

		//IPC
		Scene_edge scene;
		scene = initScene(borderImage);
		bool bIcp = getIcp();
		bIcp = true;
		time = cv::getTickCount();

		std::string classID;
		std::vector<Match>matches;
		it = mapMatches.begin(), itend = mapMatches.end();
		for (; it != itend; ++it)
		{
			classID = it->first;
			matches = it->second;
			std::vector<FindObject> findTmpObjects(matches.size());
			if (bIcp)
			{
#pragma omp parallel for
				for (int idx = 0; idx < (int)matches.size(); ++idx)
				{
					Match match = matches[idx];
					std::vector<Template> templates = getTemplates(match.class_id, match.template_id);
					Template templ = templates[0];
					cuda_icp::RegistrationResult result;
					bool valid = false;
					getIcpAffine(match, scene, result, valid);
					double icp_angle = std::atan(result.transformation_[1][0] / result.transformation_[0][0]) / CV_PI * 180;
					double icp_scale = sqrt((result.transformation_[0][0])*(result.transformation_[0][0]) + (result.transformation_[1][0]) * (result.transformation_[1][0]));
					templ.angle -= icp_angle;
					std::vector<cv::Point2f> pts(4);
					if (valid)
					{
						Template templ;
						float x, y, new_x, new_y;
						templ = templates[0];
						cv::Point2f pt;
						pt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
						pts[0] = cv::Point2f(templ.pt0) + pt;
						pts[1] = cv::Point2f(templ.pt1) + pt;
						pts[2] = cv::Point2f(templ.pt2) + pt;
						pts[3] = cv::Point2f(templ.pt3) + pt;
						//ICP
						for (int j = 0; j < pts.size(); ++j)
						{
							pt = pts[j];
							x = pt.x;
							y = pt.y;
							new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
							new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
							pts[j] = cv::Point2f(new_x, new_y);
						}

						if (bKeyPt)
						{
							drawResponse(templates, 1, dst, cv::Point(match.x, match.y), result);
						}
						cv::line(dst, pts[0], pts[1], cv::Scalar(0, 255, 0), 3);
						cv::line(dst, pts[1], pts[2], cv::Scalar(0, 255, 0), 3);
						cv::line(dst, pts[2], pts[3], cv::Scalar(0, 255, 0), 3);
						cv::line(dst, pts[3], pts[0], cv::Scalar(0, 255, 0), 3);

						//draw contours
						if (bDrawContour)
						{
							cv::Mat edgeImg;
							cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
							for (int row = 0; row < edgeImg.rows; row++)
							{
								uchar* ptr = edgeImg.ptr<uchar>(row);
								for (int col = 0; col < edgeImg.cols; col++)
								{
									if (ptr[col])
									{
										x = col - templ.tl_x + match.x;
										y = row - templ.tl_y + match.y;
										new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
										new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
										dst.at<Vec3b>(new_y, new_x) = Vec3b(255, 0, 0);
									}
								}
							}
						}
					}
					else
					{
						templ = templates[0];
						cv::Point2f pt;
						pt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
						pts[0] = cv::Point2f(templ.pt0) + pt;
						pts[1] = cv::Point2f(templ.pt1) + pt;
						pts[2] = cv::Point2f(templ.pt2) + pt;
						pts[3] = cv::Point2f(templ.pt3) + pt;

						if (bKeyPt)
						{
							drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
						}
						cv::line(dst, pts[0], pts[1], cv::Scalar(0, 255, 0), 3);
						cv::line(dst, pts[1], pts[2], cv::Scalar(0, 255, 0), 3);
						cv::line(dst, pts[2], pts[3], cv::Scalar(0, 255, 0), 3);
						cv::line(dst, pts[3], pts[0], cv::Scalar(0, 255, 0), 3);

						//draw contours
						if (bDrawContour)
						{
							cv::Mat edgeImg;
							cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
							for (int row = 0; row < edgeImg.rows; row++)
							{
								uchar* ptr = edgeImg.ptr<uchar>(row);
								for (int col = 0; col < edgeImg.cols; col++)
								{
									if (ptr[col])
									{
										int x = col - templ.tl_x + match.x;
										int y = row - templ.tl_y + match.y;
										dst.at<Vec3b>(y, x) = Vec3b(255, 0, 0);
									}
								}
							}
						}
					}
					findTmpObjects[idx] = { pts, match.angle, match.similarity, classID };
				}
				findObjects.insert(findObjects.end(), findTmpObjects.begin(), findTmpObjects.end());
			}
			else
			{
#pragma omp parallel for
				for (int idx = 0; idx < (int)matches.size(); ++idx)
				{
					Match match = matches[idx];
					std::vector<Template> templates = getTemplates(match.class_id, match.template_id);
					Template templ = templates[0];
					cv::Point2f pt;
					pt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
					std::vector<cv::Point2f> pts(4);
					pts[0] = cv::Point2f(templ.pt0) + pt;
					pts[1] = cv::Point2f(templ.pt1) + pt;
					pts[2] = cv::Point2f(templ.pt2) + pt;
					pts[3] = cv::Point2f(templ.pt3) + pt;

					if (bKeyPt)
					{
						drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
					}
					cv::line(dst, pts[0], pts[1], cv::Scalar(0, 255, 0), 3);
					cv::line(dst, pts[1], pts[2], cv::Scalar(0, 255, 0), 3);
					cv::line(dst, pts[2], pts[3], cv::Scalar(0, 255, 0), 3);
					cv::line(dst, pts[3], pts[0], cv::Scalar(0, 255, 0), 3);

					//draw contours
					if (bDrawContour)
					{
						float x, y;
						cv::Mat edgeImg;
						cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
						for (int row = 0; row < edgeImg.rows; row++)
						{
							uchar* ptr = edgeImg.ptr<uchar>(row);
							for (int col = 0; col < edgeImg.cols; col++)
							{
								if (ptr[col])
								{
									x = col - templ.tl_x + match.x;
									y = row - templ.tl_y + match.y;
									dst.at<Vec3b>(y, x) = Vec3b(255, 0, 0);
								}
							}
						}
					}
					findTmpObjects[idx] = { pts, match.angle, match.similarity, classID };
				}
				findObjects.insert(findObjects.end(), findTmpObjects.begin(), findTmpObjects.end());

			}
		
		}

		dst = dst(cv::Rect(0, 0, grayImg.cols, grayImg.rows));
		time = (cv::getTickCount() - time) * 1000.0 / cv::getTickFrequency();
		cout << "加速版-ICP及点绘制耗时: " << time << endl;

		return 1;
	}

	int Detector::detectObjectDefect(cv::Mat& src, cv::Mat& dst, line2Dup::Template& templateInfo, float matchThreshold, int amount, cv::Mat& mask, bool flag)
	{
		cv::Mat grayImg;
		if (src.channels() == 3)
		{
			cv::cvtColor(src, grayImg, cv::COLOR_RGB2GRAY);
		}
		else
		{
			grayImg = src;
		}

		int val = T_at_level[0] * T_at_level[1] * 2;//代码内限制  CV_Assert((grayImg.rows * grayImg.cols) % mipp::N<uint8_t>() == 0); mipp::N<uint8_t>()=32
		int width = (grayImg.cols % val) == 0 ? grayImg.cols : ((grayImg.cols / val + 1) * val);
		int height = (grayImg.rows % val) == 0 ? grayImg.rows : ((grayImg.rows / val + 1) * val);
		//图像不是一定要2的n次方，要的是设置的4*1 8*2的倍数，是因为后面内存重排的需要。比如设成5 8就需要5*1 8*2的倍数，也就是80的倍数。

		cv::Mat borderImage, borderMask;
		cv::copyMakeBorder(grayImg, borderImage, 0, height - grayImg.rows, 0, width - grayImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		if (!mask.empty())
		{
			cv::copyMakeBorder(mask, borderMask, 0, height - grayImg.rows, 0, width - grayImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		}

		if (grayImg.channels() == 1)
		{
			cv::cvtColor(borderImage, dst, cv::COLOR_GRAY2BGR);
		}
		else
		{
			dst = borderImage.clone();
		}


		std::vector<std::string> ids;
		std::map<string, std::vector<Match> > mapMatches;
		if (!borderMask.empty())
		{
			mapMatches = match(borderImage, matchThreshold, ids, borderMask);
		}
		else
		{
			mapMatches = match(borderImage, matchThreshold, ids);
		}

		//对于每一个类别，求前N大的值
		std::map<string, std::vector<Match> >::const_iterator it = mapMatches.begin(), itend = mapMatches.end();
		std::vector<Match> tmplMatches;
		int num;
		for (; it != itend; ++it)
		{
			num = mapMatches[it->first].size();
			if (num > amount)
			{
				mapMatches[it->first].resize(amount);
			}
		}

		//IPC
		Scene_edge scene;
		scene = initScene(borderImage);
		bool bIcp = getIcp();
		bIcp = true;

		cv::Mat affineImg = cv::Mat::zeros(borderImage.size(), CV_8UC1);
		cv::Mat affineMask = cv::Mat::zeros(borderImage.size(), CV_8UC1);
		std::string classID;
		std::vector<Match>matches;
		it = mapMatches.begin(), itend = mapMatches.end();
		for (; it != itend; ++it)
		{
			classID = it->first;
			matches = it->second;
			std::vector<FindObject> findTmpObjects(matches.size());
			if (bIcp)
			{
#pragma omp parallel for
				for (int i = 0; i < (int)matches.size(); ++i)
				{
					Match match = matches[i];
					std::vector<Template> templates = getTemplates(match.class_id, match.template_id);
					cuda_icp::RegistrationResult result;
					bool valid = false;
					getIcpAffine(match, scene, result, valid);
					//ICP angle and scale no use now
					//add by HuangLi 2019/08/23
					double icp_angle = std::atan(result.transformation_[1][0] / result.transformation_[0][0]) / CV_PI * 180;
					double icp_scale = sqrt((result.transformation_[0][0])*(result.transformation_[0][0]) + (result.transformation_[1][0]) * (result.transformation_[1][0]));
					std::vector<cv::Point2f> pts(4);
					Template templ;
					float x, y, new_x, new_y;
					cv::Point2f referPt;
					if (1)//采用原图变换到模板(正在用)
					{
						templ = templates[0];
						cv::Point2f referPt;
						referPt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
						pts[0] = cv::Point2f(templ.pt0) + referPt;
						pts[1] = cv::Point2f(templ.pt1) + referPt;
						pts[2] = cv::Point2f(templ.pt2) + referPt;
						pts[3] = cv::Point2f(templ.pt3) + referPt;

						if (bKeyPt)
						{
							drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
						}
						cv::line(dst, pts[0], pts[1], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[1], pts[2], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[2], pts[3], cv::Scalar(0, 0, 255));
						cv::line(dst, pts[3], pts[0], cv::Scalar(0, 0, 255));

						//draw contours
						if (bDrawContour)
						{
							cv::Mat edgeImg;
							cv::Canny(templ.templImg, edgeImg, 20, 40, 3);
							for (int row = 0; row < edgeImg.rows; row++)
							{
								uchar* ptr = edgeImg.ptr<uchar>(row);
								for (int col = 0; col < edgeImg.cols; col++)
								{
									if (ptr[col])
									{
										x = col - templ.tl_x + match.x;
										y = row - templ.tl_y + match.y;
										dst.at<Vec3b>(y, x) = Vec3b(255, 0, 0);
									}
								}
							}
						}
						cv::Rect tempRect;
						if (valid)
						{
							getCurrentTemplImg(templ);
							cv::Mat templImg = templ.templImg;
							cv::Mat templMask = templ.templMask;
							cv::Mat roiImg, roiMask, roiTempl;
							cv::Rect roi = cv::boundingRect(pts);
							pts[0] = cv::Point2f(templ.pt0);
							pts[1] = cv::Point2f(templ.pt1);
							pts[2] = cv::Point2f(templ.pt2);
							pts[3] = cv::Point2f(templ.pt3);

							//模板变换后的图像,采用投射标变换（逆变换）
							vector<Point2f> src_coners;
							vector<Point2f> dst_coners;
							cv::Point2f pt1, pt2, pt3, pt4;
							cv::Point2f pt1_, pt2_, pt3_, pt4_;
							pt1 = cv::Point2f(0, 0);
							x = pt1.x - templ.tl_x + match.x;
							y = pt1.y - templ.tl_y + match.y;
							new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
							new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
							pt1_ = cv::Point2f(new_x, new_y);

							pt2 = cv::Point2f(0, templImg.rows);
							x = pt2.x - templ.tl_x + match.x;
							y = pt2.y - templ.tl_y + match.y;
							new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
							new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
							pt2_ = cv::Point2f(new_x, new_y);

							pt3 = cv::Point2f(templImg.cols, 0);
							x = pt3.x - templ.tl_x + match.x;
							y = pt3.y - templ.tl_y + match.y;
							new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
							new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
							pt3_ = cv::Point2f(new_x, new_y);

							pt4 = cv::Point2f(templImg.cols, templImg.rows);
							x = pt4.x - templ.tl_x + match.x;
							y = pt4.y - templ.tl_y + match.y;
							new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
							new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
							pt4_ = cv::Point2f(new_x, new_y);

							src_coners.push_back(pt1);
							src_coners.push_back(pt2);
							src_coners.push_back(pt3);
							src_coners.push_back(pt4);

							dst_coners.push_back(pt1_);
							dst_coners.push_back(pt2_);
							dst_coners.push_back(pt3_);
							dst_coners.push_back(pt4_);
							cv::Mat warpMatrix = getPerspectiveTransform(dst_coners, src_coners);
							warpPerspective(borderImage, borderImage, warpMatrix, templImg.size(), INTER_LINEAR, BORDER_CONSTANT);

							if (flag)
							{
								//跟模板第一个对齐
								std::vector<Template> alignTempls = getTemplates(match.class_id, 0);
								Template alignTempl;
								alignTempl = alignTempls[0];
								float disAngle = alignTempl.angle - templ.angle;
								cv::Point2f center(borderImage.cols / 2.0f, borderImage.rows / 2.0f);
								cv::Mat rot_mat = cv::getRotationMatrix2D(center, disAngle, 1.0);
								cv::warpAffine(borderImage, borderImage, rot_mat, borderImage.size());

								//有效区域
								pts[0] = cv::Point2f(alignTempl.pt0);
								pts[1] = cv::Point2f(alignTempl.pt1);
								pts[2] = cv::Point2f(alignTempl.pt2);
								pts[3] = cv::Point2f(alignTempl.pt3);
								tempRect = cv::boundingRect(pts);

								//XY方向模板那匹配
								getCurrentTemplImg(alignTempl);
								cv::Mat templImg = alignTempl.templImg(tempRect);
								coarsePosition(borderImage, templImg, tempRect);
								src = borderImage(tempRect);
								templateInfo.templImg = templImg;
								templateInfo.templMask = alignTempl.templMask(tempRect);
								templateInfo.angle = templ.angle;
							}
							else
							{
								cv::Rect tempRect;
								tempRect = roi - cv::Point(referPt.x, referPt.y);
								src = affineImg(tempRect);
								templateInfo.templImg = templImg(tempRect);
								templateInfo.templMask = templ.templMask(tempRect);
							}
						}
						else
						{
							cv::Rect tempRect = cv::Rect(match.x - templ.tl_x, match.y - templ.tl_y, m_templateImg.cols, m_templateImg.rows);
							int top, bottom, left, right;
							top = bottom = left = right = 0;
							if (tempRect.x < 0)
							{
								left = -tempRect.x;
							}
							if (tempRect.y < 0)
							{
								top = -tempRect.y;
							}
							if (tempRect.x + tempRect.width > borderImage.cols)
							{
								right = tempRect.x + tempRect.width - borderImage.cols;
							}
							if (tempRect.y + tempRect.height > borderImage.rows)
							{
								bottom = tempRect.y + tempRect.height - borderImage.rows;
							}
							cv::copyMakeBorder(borderImage, borderImage, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
							tempRect += cv::Point(left, top);
							borderImage = borderImage(tempRect);

							if (flag)
							{
								//跟模板第一个对齐
								std::vector<Template> alignTempls = getTemplates(match.class_id, 0);
								Template alignTempl;
								alignTempl = alignTempls[0];
								float disAngle = alignTempl.angle - templ.angle;
								cv::Point2f center(borderImage.cols / 2.0f, borderImage.rows / 2.0f);
								cv::Mat rot_mat = cv::getRotationMatrix2D(center, disAngle, 1.0);
								cv::warpAffine(borderImage, borderImage, rot_mat, borderImage.size());

								//有效区域
								pts[0] = cv::Point2f(alignTempl.pt0);
								pts[1] = cv::Point2f(alignTempl.pt1);
								pts[2] = cv::Point2f(alignTempl.pt2);
								pts[3] = cv::Point2f(alignTempl.pt3);
								tempRect = cv::boundingRect(pts);

								//XY方向模板那匹配
								getCurrentTemplImg(alignTempl);
								cv::Mat templImg = alignTempl.templImg(tempRect);
								coarsePosition(borderImage, templImg, tempRect);
								src = borderImage(tempRect);
							}
							else
							{
								getCurrentTemplImg(templ);
								cv::Rect roi = cv::boundingRect(pts);
								cv::Rect tempRect = roi - cv::Point(referPt.x, referPt.y);
								src = affineImg(tempRect);
								templateInfo.templImg = templ.templImg(tempRect);
								templateInfo.templMask = templ.templMask(tempRect);
							}
						}
					}
				}
			}
			else
			{
#pragma omp parallel for
				for (int i = 0; i < (int)matches.size(); ++i)
				{
					Match match = matches[i];
					std::vector<Template> templates = getTemplates(match.class_id, match.template_id);
					Template templ = templates[0];
					cv::Point2f referPt;
					referPt = cv::Point2f(match.x - templ.tl_x, match.y - templ.tl_y);
					std::vector<cv::Point2f> pts(4);
					pts[0] = cv::Point2f(templ.pt0) + referPt;
					pts[1] = cv::Point2f(templ.pt1) + referPt;
					pts[2] = cv::Point2f(templ.pt2) + referPt;
					pts[3] = cv::Point2f(templ.pt3) + referPt;

					if (bKeyPt)
					{
						drawResponse(templates, 1, dst, cv::Point(match.x, match.y));
					}
					cv::line(dst, pts[0], pts[1], cv::Scalar(0, 0, 255));
					cv::line(dst, pts[1], pts[2], cv::Scalar(0, 0, 255));
					cv::line(dst, pts[2], pts[3], cv::Scalar(0, 0, 255));
					cv::line(dst, pts[3], pts[0], cv::Scalar(0, 0, 255));

					cv::Rect tempRect = cv::Rect(match.x - templ.tl_x, match.y - templ.tl_y, m_templateImg.cols, m_templateImg.rows);
					int top, bottom, left, right;
					top = bottom = left = right = 0;
					if (tempRect.x < 0)
					{
						left = -tempRect.x;
					}
					if (tempRect.y < 0)
					{
						top = -tempRect.y;
					}
					if (tempRect.x + tempRect.width > borderImage.cols)
					{
						right = tempRect.x + tempRect.width - borderImage.cols;
					}
					if (tempRect.y + tempRect.height > borderImage.rows)
					{
						bottom = tempRect.y + tempRect.height - borderImage.rows;
					}
					cv::copyMakeBorder(borderImage, borderImage, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
					tempRect += cv::Point(left, top);
					borderImage = borderImage(tempRect);

					if (flag)
					{
						//跟模板第一个对齐
						std::vector<Template> alignTempls = getTemplates(match.class_id, 0);
						Template alignTempl;
						alignTempl = alignTempls[0];
						float disAngle = alignTempl.angle - templ.angle;
						cv::Point2f center(borderImage.cols / 2.0f, borderImage.rows / 2.0f);
						cv::Mat rot_mat = cv::getRotationMatrix2D(center, disAngle, 1.0);
						cv::warpAffine(borderImage, borderImage, rot_mat, borderImage.size());

						//有效区域
						pts[0] = cv::Point2f(alignTempl.pt0);
						pts[1] = cv::Point2f(alignTempl.pt1);
						pts[2] = cv::Point2f(alignTempl.pt2);
						pts[3] = cv::Point2f(alignTempl.pt3);
						tempRect = cv::boundingRect(pts);

						//XY方向模板那匹配
						getCurrentTemplImg(alignTempl);
						cv::Mat templImg = alignTempl.templImg(tempRect);
						coarsePosition(borderImage, templImg, tempRect);
						src = borderImage(tempRect);
						templateInfo.templImg = alignTempl.templImg(tempRect);
						templateInfo.templMask = alignTempl.templMask(tempRect);
					}
					else
					{
						getCurrentTemplImg(templ);
						cv::Rect roi = cv::boundingRect(pts);
						cv::Rect tempRect = roi - cv::Point(referPt.x, referPt.y);
						src = affineImg(tempRect);
						templateInfo.templImg = templ.templImg(tempRect);
						templateInfo.templMask = templ.templMask(tempRect);
					}
				}
			}

		}

		return 1;
	}

	void Detector::calSurface(cv::Mat& src, cv::Mat& templImg, cv::Mat& mask, cv::Mat& dst, std::vector<cv::Rect>& defectRois, cv::Point pt)
	{
		//读取参数
		int flag = parms["Surface"]["Flag"];
		int bigTh = parms["Surface"]["BigTh"];
		int smallTh = parms["Surface"]["SmallTh"];
		int winSize = parms["Surface"]["WinSize"];
		int width = parms["Surface"]["Width"];
		int height = parms["Surface"]["Height"];
		int size = parms["Surface"]["Size"];
		int erode = parms["Surface"]["Erode"];
		int breakTh = parms["Surface"]["BreakTh"];
		int breakWidth = parms["Surface"]["BreakWidth"];
		int breakHeight = parms["Surface"]["BreakHeight"];
		int candWidth = parms["Surface"]["CandWidth"];
		int candHeight = parms["Surface"]["CandHeight"];

		if (flag == 0)
		{
			return;
		}

		//自身计算mask
		cv::Mat validMask;
		/*validMask = src < 220;
		cv::bitwise_and(mask, validMask, validMask);*/
		cv::erode(mask, validMask, cv::Mat(), cv::Point(-1, -1), erode);

		//区分不同区域
		cv::Mat bigMask, smallMask, ironMask;
		cv::inRange(validMask, 2, 50, bigMask);
		cv::inRange(validMask, 50, 100, smallMask);
		cv::inRange(validMask, 100, 150, ironMask);

		/*************************滤波处理*************************/
		cv::Mat filterImg;
		//retinexPde(src, filterImg, 3);
		filterImg = src.clone();

		/*************************计算裂纹*************************/
		cv::Mat crackImg;
		cv::Mat smoothImg, diffImg, binary;
		bilateralFilter(filterImg, smoothImg, winSize, winSize * 2, winSize / 2);
		diffImg = smoothImg - filterImg;
		crackImg = cv::Mat::zeros(src.size(), CV_8UC1);
		//正常区域
		binary = diffImg > bigTh;
		cv::bitwise_and(binary, bigMask, binary);
		crackImg += binary;
		//非正常区域（两侧）
		binary = diffImg > smallTh;
		cv::bitwise_and(binary, smallMask, binary);
		crackImg += binary;
		
		std::vector<std::vector<cv::Point> > contours;
		cv::Rect tempRect, exactRect;
		cv::findContours(crackImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		crackImg = cv::Mat::zeros(crackImg.size(), CV_8UC1);
		for (int i = 0; i < contours.size(); i++)
		{
			tempRect = cv::boundingRect(contours[i]);
			if (tempRect.width > 5 && tempRect.height > 5)
			{
				cv::drawContours(crackImg, contours, i, cv::Scalar(255), -1);
			}
		}
		int an = 2;
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * an + 1, 2 * an + 1), cv::Point(an, an));
		cv::morphologyEx(crackImg, crackImg, cv::MORPH_CLOSE, element);
		cv::findContours(crackImg.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		float ratio, area;
		for (int i = 0; i < contours.size(); i++)
		{
			tempRect = cv::boundingRect(contours[i]);
			if (tempRect.width > candWidth && tempRect.height > candHeight)
			{
				area = cv::contourArea(contours[i]);
				ratio = area / tempRect.area();
				//if (ratio < 0.5)
				{
					exactRect = tempRect + cv::Point(pt.x, pt.y);
					defectRois.push_back(exactRect);
				}
			}
		}

		/*************************计算大块破损*************************/
		cv::Mat breakDefect;
		double meanVal;
		meanVal = cv::mean(src, bigMask).val[0];
		binary = src < meanVal - breakTh;
		cv::Mat erodeMask;
		cv::erode(bigMask, erodeMask, cv::Mat(), cv::Point(-1, -1), 5);
		cv::bitwise_and(erodeMask, binary, breakDefect);
		cv::findContours(breakDefect.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		for (int i = 0; i < contours.size(); i++)
		{
			tempRect = cv::boundingRect(contours[i]);
			if (tempRect.width > breakWidth && tempRect.height > breakWidth)
			{
				exactRect = tempRect + cv::Point(pt.x, pt.y);
				defectRois.push_back(exactRect);
			}
		}


		/*************************计算铁片破损*************************/
		cv::Mat ironDefect;
		binary = templImg - src;
		ironDefect = binary > 100;
		cv::bitwise_and(ironDefect, ironMask, ironDefect);
		cv::findContours(ironDefect.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		for (int i = 0; i < contours.size(); i++)
		{
			tempRect = cv::boundingRect(contours[i]);
			if (tempRect.width > breakWidth && tempRect.height > breakWidth)
			{
				exactRect = tempRect + cv::Point(pt.x, pt.y);
				defectRois.push_back(exactRect);
			}
		}

		/*************************缺陷汇总*************************/
		std::vector<cv::Rect> clusterRois;
		clusterRoi(defectRois, clusterRois, 8);
		defectRois.clear();
		for (int i = 0; i < clusterRois.size(); ++i)
		{
			tempRect = clusterRois[i];
			if (tempRect.width > width && tempRect.height > height)
			{
				defectRois.push_back(tempRect);
			}
		}

		dst = crackImg + breakDefect + ironDefect;
	}

	void Detector::calGap(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, std::vector<cv::Rect>& defectRois, cv::Point pt)
	{
		//读取参数
		int flag = parms["Gap"]["Flag"];
		int th = parms["Gap"]["Th"];
		int abTh = parms["Gap"]["AbTh"];
		int midTh = parms["Gap"]["MidTh"];
		int width = parms["Gap"]["Width"];
		
		if (flag == 0)
		{
			return;
		}

		dst = cv::Mat::zeros(src.size(), CV_8UC1);
		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		if (contours.size() == 0)
		{
			return;
		}
		double area, maxArea = FLT_MIN;
		cv::Rect tempRect, maxRect;
		int index = -1;
		for (int i = 0; i < contours.size(); ++i)
		{
			tempRect = cv::boundingRect(contours[i]);
			area = tempRect.area();
			if (area > maxArea)
			{
				maxArea = area;
				maxRect = tempRect;
				index = i;
			}
		}
		std::vector<std::vector<cv::Point> > hull(contours.size());
		cv::convexHull(cv::Mat(contours[index]), hull[0], false);
		cv::Mat externMask = cv::Mat::zeros(src.size(), CV_8UC1);
		cv::drawContours(externMask, hull, 0, cv::Scalar(255), -1);
		cv::erode(externMask, externMask, cv::Mat(), cv::Point(-1, -1), 15);

		cv::Mat binary;
		//cv::threshold(src, binary, 0, 255, cv::THRESH_OTSU);
		binary = src > 240;
		int an = 3;
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(an * 2 + 1, an * 2 + 1), cv::Point(an ,an));
		cv::morphologyEx(binary, binary, cv::MORPH_OPEN, element);
		cv::bitwise_and(binary, externMask, binary);
		cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		if (contours.size() == 0)
		{
			return;
		}
		maxArea = FLT_MIN;
		index = -1;
		for (int i = 0; i < contours.size(); ++i)
		{
			tempRect = cv::boundingRect(contours[i]);
			area = tempRect.area();
			if (area > maxArea)
			{
				maxArea = area;
				maxRect = tempRect;
				index = i;
			}
		}
		cv::Rect roi = cv::Rect(maxRect.x + 20, maxRect.y - 20, maxRect.width - 40, 40);
		roi &= cv::Rect(0, 0, src.cols, src.rows);

		cv::Mat roiImg, roiMask;
		roiImg = src(roi);
		binary = roiImg < 240;
		/********************相对值计算******************/
		double meanVal = cv::mean(roiImg, binary).val[0];
		binary = roiImg < meanVal * 1.2;
		meanVal = cv::mean(roiImg, binary).val[0];
		binary = roiImg< meanVal - th;
		/********************绝对值计算******************/
		cv::Mat abBinary = roiImg < abTh;
		binary += abBinary;
		//其他方法
		cv::Mat midImg;
		cv::medianBlur(roiImg, midImg, 15);
		cv::Mat diffImg, midBinary;
		diffImg = midImg - roiImg;
		midBinary = diffImg > midTh;
		binary += midBinary;

		cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		if (contours.size() == 0)
		{
			return;
		}
		binary = cv::Scalar(0);
		std::vector<cv::Rect> tempRects;
		for (int i = 0; i < contours.size(); ++i)
		{
			tempRect = cv::boundingRect(contours[i]);
			if (tempRect.width > binary.cols * 0.05)
			{
				cv::drawContours(binary, contours, i, cv::Scalar(255), -1);
				tempRects.push_back(tempRect);
			}
		}
		binary.copyTo(dst(roi));
		

		//计算宽度
		int y, sum = cv::countNonZero(binary) / maxRect.width;
		std::vector<int> lengths;
		for (int i = 0; i < tempRects.size(); ++i)
		{
			maxRect = tempRects[i];
			for (int col = maxRect.x; col < maxRect.x + maxRect.width; ++col)
			{
				y = 0;
				for (int row = maxRect.y; row < maxRect.y + maxRect.height; ++row)
				{
					if (binary.at<uchar>(row, col))
					{
						y++;
					}
				}
				if (y >= sum)
				{
					lengths.push_back(y);
				}
			}

		}
		
		std::sort(lengths.begin(), lengths.end(), greater<int>());
		int size = lengths.size();
		int maxValue = *max_element(lengths.begin(), lengths.end());
		int *count = new int[maxValue + 1];
		for (int i = 0; i < lengths.size(); ++i)
		{
			count[lengths[i]]++;
		}
		vector<int>::iterator new_end;
		new_end = unique(lengths.begin(), lengths.end());//"删除"相邻的重复元素
		lengths.erase(new_end, lengths.end());//删除(真正的删除)重复的元素

		int actualWidth = 0;
		for (int i = 0; i < lengths.size(); ++i)
		{
			if (count[lengths[i]] > 0.05 * size)
			{
				actualWidth = lengths[i];
				break;
			}
		}

		if (actualWidth >= width)
		{
			defectRois.push_back(roi + pt);
		}


		//int tq = 100;
	}

} // namespace line2Dup
