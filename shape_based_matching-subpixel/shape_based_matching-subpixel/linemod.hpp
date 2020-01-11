/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <map>
#include "cuda_icp/icp.h"

/****************************************************************************************\
*                                 LINE-MOD                                               *
\****************************************************************************************/

namespace matchHL {
namespace linemod {

struct FindObject
{
	cv::Rect roi;
	float angle;
	float score;
	cv::Mat img;

};

//! @addtogroup rgbd
//! @{

/**
 * \brief Discriminant feature described by its location and label.
 */
struct  Feature
{
   int x; ///< x offset
   int y; ///< y offset
   int label; ///< Quantization

   Feature() : x(0), y(0), label(0) {}
   Feature(int x, int y, int label);

  void read(const cv::FileNode& fn);
  void write(cv::FileStorage& fs) const;
};

inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

struct  Template
{
   int width;
   int height;
   int pyramid_level;
   std::vector<Feature> features;

   //add by HuangLi 2019/08/08
   float scale;
   float angle;
   int tl_x;
   int tl_y;
   cv::Mat templImg;
   cv::Mat templMask;

  void read(const cv::FileNode& fn);
  void write(cv::FileStorage& fs) const;
};

/**
 * \brief Represents a modality operating over an image pyramid.
 */
class  QuantizedPyramid
{
public:
  // Virtual destructor
  virtual ~QuantizedPyramid() {}

  /**
   * \brief Compute quantized image at current pyramid level for online detection.
   *
   * \param[out] dst The destination 8-bit image. For each pixel at most one bit is set,
   *                 representing its classification.
   */
   virtual void quantize(CV_OUT cv::Mat& dst) const =0;

  /**
   * \brief Extract most discriminant features at current pyramid level to form a new template.
   *
   * \param[out] templ The new template.
   */
   virtual bool extractTemplate(CV_OUT Template& templ) const =0;

  /**
   * \brief Go to the next pyramid level.
   *
   * \todo Allow pyramid scale factor other than 2
   */
   virtual void pyrDown() =0;

protected:
  /// Candidate feature with a score
  struct Candidate
  {
    Candidate(int x, int y, int label, float score);

    /// Sort candidates with high score to the front
    bool operator<(const Candidate& rhs) const
    {
      return score > rhs.score;
    }

    Feature f;
    float score;
  };

  /**
   * \brief Choose candidate features so that they are not bunched together.
   *
   * \param[in]  candidates   Candidate features sorted by score.
   * \param[out] features     Destination vector of selected features.
   * \param[in]  num_features Number of candidates to select.
   * \param[in]  distance     Hint for desired distance between features.
   */
  static bool selectScatteredFeatures(const std::vector<Candidate>& candidates,
                                      std::vector<Feature>& features,
                                      size_t num_features, float distance);
};

inline QuantizedPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

/**
 * \brief Interface for modalities that plug into the LINE template matching representation.
 *
 * \todo Max response, to allow optimization of summing (255/MAX) features as uint8
 */
class  Modality
{
public:
  // Virtual destructor
  virtual ~Modality() {}

  /**
   * \brief Form a quantized image pyramid from a source image.
   *
   * \param[in] src  The source image. Type depends on the modality.
   * \param[in] mask Optional mask. If not empty, unmasked pixels are set to zero
   *                 in quantized image and cannot be extracted as features.
   */
   cv::Ptr<QuantizedPyramid> process(const cv::Mat& src,
                    const cv::Mat& mask = cv::Mat()) const
  {
    return processImpl(src, mask);
  }

   virtual cv::String name() const =0;

   virtual void read(const cv::FileNode& fn) =0;
  virtual void write(cv::FileStorage& fs) const =0;

  /**
   * \brief Create modality by name.
   *
   * The following modality types are supported:
   * - "ColorGradient"
   * - "DepthNormal"
   */
   static cv::Ptr<Modality> create(const cv::String& modality_type);

  /**
   * \brief Load a modality from file.
   */
   static cv::Ptr<Modality> create(const cv::FileNode& fn);

protected:
  // Indirection is because process() has a default parameter.
  virtual cv::Ptr<QuantizedPyramid> processImpl(const cv::Mat& src,
                        const cv::Mat& mask) const =0;
};

/**
 * \brief Modality that computes quantized gradient orientations from a color image.
 */
class  ColorGradient : public Modality
{
public:
  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  ColorGradient();

  /**
   * \brief Constructor.
   *
   * \param weak_threshold   When quantizing, discard gradients with magnitude less than this.
   * \param num_features     How many features a template must contain.
   * \param strong_threshold Consider as candidate features only gradients whose norms are
   *                         larger than this.
   */
  ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

  virtual cv::String name() const;

  virtual void read(const cv::FileNode& fn);
  virtual void write(cv::FileStorage& fs) const;

  float weak_threshold;
  size_t num_features;
  float strong_threshold;

protected:
  virtual cv::Ptr<QuantizedPyramid> processImpl(const cv::Mat& src,
                        const cv::Mat& mask) const;
};

/**
 * \brief Modality that computes quantized surface normals from a dense depth map.
 */
class  DepthNormal : public Modality
{
public:
  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  DepthNormal();

  /**
   * \brief Constructor.
   *
   * \param distance_threshold   Ignore pixels beyond this distance.
   * \param difference_threshold When computing normals, ignore contributions of pixels whose
   *                             depth difference with the central pixel is above this threshold.
   * \param num_features         How many features a template must contain.
   * \param extract_threshold    Consider as candidate feature only if there are no differing
   *                             orientations within a distance of extract_threshold.
   */
  DepthNormal(int distance_threshold, int difference_threshold, size_t num_features,
              int extract_threshold);

  virtual cv::String name() const;

  virtual void read(const cv::FileNode& fn);
  virtual void write(cv::FileStorage& fs) const;

  int distance_threshold;
  int difference_threshold;
  size_t num_features;
  int extract_threshold;

protected:
  virtual cv::Ptr<QuantizedPyramid> processImpl(const cv::Mat& src,
                        const cv::Mat& mask) const;
};

/**
 * \brief Debug function to colormap a quantized image for viewing.
 */
 void colormap(const cv::Mat& quantized, CV_OUT cv::Mat& dst);

/**
 * \brief Represents a successful template match.
 */
struct  Match
{
   Match()
  {
  }

   Match(int x, int y, float similarity, const cv::String& class_id, int template_id);

  /// Sort matches with high similarity to the front
  bool operator<(const Match& rhs) const
  {
    // Secondarily sort on template_id for the sake of duplicate removal
    if (similarity != rhs.similarity)
      return similarity > rhs.similarity;
    else
      return template_id < rhs.template_id;
  }

  bool operator==(const Match& rhs) const
  {
    return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
  }

   int x;
   int y;
   float similarity;
   cv::String class_id;
   int template_id;
};




inline
Match::Match(int _x, int _y, float _similarity, const cv::String& _class_id, int _template_id)
    : x(_x), y(_y), similarity(_similarity), class_id(_class_id), template_id(_template_id)
{}

/**
 * \brief Object detector using the LINE template matching algorithm with any set of
 * modalities.
 */
class  Detector
{
public:
  /**
   * \brief Empty constructor, initialize with read().
   */
   Detector();

  /**
   * \brief Constructor.
   *
   * \param modalities       Modalities to use (color gradients, depth normals, ...).
   * \param T_pyramid        Value of the sampling step T at each pyramid level. The
   *                         number of pyramid levels is T_pyramid.size().
   */
   Detector(const std::vector< cv::Ptr<Modality> >& modalities, const std::vector<int>& T_pyramid);

  /**
  * \Remove candidate targets for region duplication ratio=0.8
  * \param      vecSrc    Source data.
  * \param      vecDst	  Dst data.
  * \param      dDis	  Distance between two targets by coordinate(X, Y)
  */
  //add by HuangLi 2019/08/06
 //  void clusterCoordinate(std::vector<int>& vecSrc, std::vector<int>& vecDst, double dDis);
   //void clusterCoordinate(std::vector<int>& vecSrc);
  
  /**
   * \brief Detect objects by template matching.
   *
   * Matches globally at the lowest pyramid level, then refines locally stepping up the pyramid.
   *
   * \param      sources   Source images, one for each modality.
   * \param      threshold Similarity threshold, a percentage between 0 and 100.
   * \param[out] matches   Template matches, sorted by similarity score.
   * \param      class_ids If non-empty, only search for the desired object classes.
   * \param[out] quantized_images Optionally return vector<cv::Mat> of quantized images.
   * \param      masks     The masks for consideration during matching. The masks should be CV_8UC1
   *                       where 255 represents a valid pixel.  If non-empty, the vector must be
   *                       the same size as sources.  Each element must be
   *                       empty or the same size as its corresponding source.
   */
   void match(const cv::Mat& source, float threshold, CV_OUT std::vector<Match>& matches,
             const std::vector<cv::String>& class_ids = std::vector<cv::String>(),
             const cv::Mat& mask = cv::Mat()) const;

  /**
   * \brief Add new object template.
   *
   * \param      sources      Source images, one for each modality.
   * \param      class_id     Object class ID.
   * \param      object_mask  Mask separating object from background.
   * \param[out] bounding_box Optionally return bounding box of the extracted features.
   *
   * \return Template ID, or -1 if failed to extract a valid template.
   */
   int addTemplate(const std::vector<cv::Mat>& sources, const cv::String& class_id,
          const cv::Mat& object_mask, CV_OUT cv::Rect* bounding_box = NULL);

  //add by HuangLi 2019/08/06
   int addTemplate(const cv::Mat& sources, const cv::String& class_id,
	  const cv::Mat& object_mask, cv::Rect* bounding_box = NULL, float angle = 0.0f, float scale = 1.0f);

   //add by HuangLi 2019/08/13
   /**
   * \init ICP.
   * \param	     image  Source image.
   * \return	 ICP scene edge.
   */
   Scene_edge initScene(const cv::Mat image);

   //add by HuangLi 2019/08/13
   /**
   * \calculation ICP transformation.
   * \param	     image  Source image.
   * \param      class_id Object class ID.
   * \param      template_id Object template index.
   * \return	 ICP affined matrix.
   */
   cv::Mat getIcpAffine(const cv::Mat image, Match& match, Scene_edge& scene, bool valid = true, cv::Rect& roi = cv::Rect());

  /**
   * \brief Add a new object template computed by external means.
   */
   int addSyntheticTemplate(const std::vector<Template>& templates, const cv::String& class_id);

  /**
   * \brief Get the modalities used by this detector.
   *
   * You are not permitted to add/remove modalities, but you may dynamic_cast them to
   * tweak parameters.
   */
   const std::vector< cv::Ptr<Modality> >& getModalities() const { return modalities; }

  /**
   * \brief Get sampling step T at pyramid_level.
   */
   int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

  /**
   * \brief Get number of pyramid levels used by this detector.
   */
   int pyramidLevels() const { return pyramid_levels; }

  /**
   * \brief Get the template pyramid identified by template_id.
   *
   * For example, with 2 modalities (Gradient, Normal) and two pyramid levels
   * (L0, L1), the order is (GradientL0, NormalL0, GradientL1, NormalL1).
   */
   const std::vector<Template>& getTemplates(const cv::String& class_id, int template_id) const;

   int numTemplates() const;
   int numTemplates(const cv::String& class_id) const;
   int numClasses() const { return static_cast<int>(class_templates.size()); }

   std::vector<cv::String> classIds() const;

   void read(const cv::FileNode& fn);
  void write(cv::FileStorage& fs) const;

  cv::String readClass(const cv::FileNode& fn, const cv::String &class_id_override = "");
  void writeClass(const cv::String& class_id, cv::FileStorage& fs) const;

   void readClasses(const std::vector<cv::String>& class_ids,
                   const cv::String& format = "templates_%s.yml.gz");
   void writeClasses(const cv::String& format = "templates_%s.yml.gz") const;


   //add by HuangLi 2019/08/14
	void createModel(const cv::Mat& src, const cv::Mat& mask, std::string modelName = "train_origin");
	void loadModel();
	float getStartAngle(){ return startAngle; }
	void  setStartAngle(float startAngle){ this->startAngle = startAngle; }
	float getEndAngle(){ return endAngle; }
	void  setEndAngl(float endAngle){ this->endAngle = endAngle; }
	float getStepAngle(){ return stepAngle; }
	void  setStepAngle(float stepAngle){ this->stepAngle = stepAngle; }
	float getStartScale(){ return startScale; }
	void  setStartScale(float startScale){ this->startScale = startScale; }
	float getEndScale(){ return endScale; }
	void  setEndScale(float endScale){ this->endScale = endScale; }
	float getStepScale(){ return stepScale; }
	void  setStepScale(float stepScale){ this->stepScale = stepScale; }
	//void drawResponse(const std::vector<matchHL::linemod::Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset, int T);
	void drawResponse(const std::vector<Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset);
	void drawResponse(const std::vector<Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset, cuda_icp::RegistrationResult& icpAffine);
	//Get ROI image of src corresponding to the template image
	cv::Mat getRoiImg(cv::Mat& src, cv::Rect roi);

	//add by HuangLi 2019/08/14
	void detect(cv::Mat &src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, int amount = 10000, cv::Mat &mask = cv::Mat());

	private:
	   float	startAngle;
	   float	endAngle;
	   float	stepAngle;
	   float	startScale;
	   float	endScale;
	   float	stepScale;
	   float	matchThreshold;
		
protected:
  std::vector< cv::Ptr<Modality> > modalities;
  int pyramid_levels;
  std::vector<int> T_at_level;

  typedef std::vector<Template> TemplatePyramid;
  typedef std::map<cv::String, std::vector<TemplatePyramid> > TemplatesMap;
  TemplatesMap class_templates;

  typedef std::vector<cv::Mat> LinearMemories;
  // Indexed as [pyramid level][modality][quantized label]
  typedef std::vector< std::vector<LinearMemories> > LinearMemoryPyramid;

  void matchClass(const LinearMemoryPyramid& lm_pyramid,
                  const std::vector<cv::Size>& sizes,
                  float threshold, std::vector<Match>& matches,
                  const cv::String& class_id,
                  const std::vector<TemplatePyramid>& template_pyramids) const;
};

//template void Detector::clusterCoordinate(const std::vector<matchHL::linemod::Match>& vecSrc, std::vector<matchHL::linemod::Match>& vecDst, double dDis);

/**
 * \brief Factory function for detector using LINE algorithm with color gradients.
 *
 * Default parameter settings suitable for VGA images.
 */
 cv::Ptr<linemod::Detector> getDefaultLINE();
 cv::Ptr<linemod::Detector> getDefaultLINE(float _weak_threshold, size_t _num_features, float _strong_threshold);

/**
 * \brief Factory function for detector using LINE-MOD algorithm with color gradients
 * and depth normals.
 *
 * Default parameter settings suitable for VGA images.
 */
 cv::Ptr<linemod::Detector> getDefaultLINEMOD();

//! @}

} // namespace linemod

namespace shape_based_matching
{
	class shapeInfo
	{
	public:
		cv::Mat src;
		cv::Mat mask;

		std::vector<float> angle_range;
		std::vector<float> scale_range;

		float angle_step = 15;
		float scale_step = 0.5;
		float eps = 0.00001f;

		class shape_and_info{
		public:
			cv::Mat src;
			cv::Mat mask;
			float angle;
			float scale;
			shape_and_info(cv::Mat src_, cv::Mat mask_, float angle_, float scale_){
				src = src_;
				mask = mask_;
				angle = angle_;
				scale = scale_;
			}
		};
		std::vector<shape_and_info> infos;

		shapeInfo(cv::Mat src, cv::Mat mask = cv::Mat()){
			this->src = src;
			if(mask.empty()){
				// make sure we have masks
				this->mask = cv::Mat(src.size(), CV_8UC1, {255});
			}else{
				this->mask = mask;
			}
		}

		static cv::Mat transform(cv::Mat src, float angle, float scale){
			cv::Mat dst;

			cv::Point2f center(src.cols/2.0f, src.rows/2.0f);
			cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
			cv::warpAffine(src, dst, rot_mat, src.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

			return dst;
		}
		static void save_infos(std::vector<shapeInfo::shape_and_info>& infos, cv::Mat src, cv::Mat mask, cv::String path = "infos.yaml"){
			cv::FileStorage fs(path, cv::FileStorage::WRITE);
			fs << "src" << src;
			fs << "mask" << mask;
			fs << "infos"
			   << "[";
			for (int i = 0; i < infos.size(); i++)
			{
				fs << "{";
				fs << "angle" << infos[i].angle;
				fs << "scale" << infos[i].scale;
				fs << "}";
			}
			fs << "]";
		}
		static std::vector<std::vector<float>> load_infos(cv::Mat& src, cv::Mat& mask, cv::String path = "info.yaml"){
			cv::FileStorage fs(path, cv::FileStorage::READ);

			fs["src"] >> src;
			fs["mask"] >> mask;
			std::vector<std::vector<float>> infos;
			cv::FileNode infos_fn = fs["infos"];
			cv::FileNodeIterator it = infos_fn.begin(), it_end = infos_fn.end();
			for (int i = 0; it != it_end; ++it, i++)
			{
				std::vector<float> info;
				info.push_back(float((*it)["angle"]));
				info.push_back(float((*it)["scale"]));
				infos.push_back(info);
			}
		}

		void produce_infos(){
			assert(angle_range.size() <= 2);
			assert(scale_range.size() <= 2);
			assert(angle_step > eps*10);
			assert(scale_step > eps*10);

			// make sure range not empty
			if(angle_range.size() == 0){
				angle_range.push_back(0);
			}
			if(scale_range.size() == 0){
				scale_range.push_back(1);
			}

			if(angle_range.size() == 1 && scale_range.size() == 1){
				float angle = angle_range[0];
				float scale = scale_range[0];
				cv::Mat src_transformed = transform(src, angle, scale);
				cv::Mat mask_transformed = transform(mask, angle, scale);
				mask_transformed = mask_transformed > 0; //make sure it's a mask after transform
				infos.emplace_back(src_transformed, mask_transformed, angle, scale);

			}else if(angle_range.size() == 1 && scale_range.size() == 2){
				assert(scale_range[1] > scale_range[0]);
				float angle = angle_range[0];
				for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step){
					cv::Mat src_transformed = transform(src, angle, scale);
					cv::Mat mask_transformed = transform(mask, angle, scale);
					mask_transformed = mask_transformed > 0; //make sure it's a mask after transform
					infos.emplace_back(src_transformed, mask_transformed, angle, scale);
				}
			}else if(angle_range.size() == 2 && scale_range.size() == 1){
				assert(angle_range[1] > angle_range[0]);
				float scale = scale_range[0];
				for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step){
					cv::Mat src_transformed = transform(src, angle, scale);
					cv::Mat mask_transformed = transform(mask, angle, scale);
					mask_transformed = mask_transformed > 0; //make sure it's a mask after transform
					infos.emplace_back(src_transformed, mask_transformed, angle, scale);
				}
			}else if(angle_range.size() == 2 && scale_range.size() == 2){
				assert(scale_range[1] > scale_range[0]);
				assert(angle_range[1] > angle_range[0]);
				for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step){
					for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step){
						cv::Mat src_transformed = transform(src, angle, scale);
						cv::Mat mask_transformed = transform(mask, angle, scale);
						mask_transformed = mask_transformed > 0; //make sure it's a mask after transform
						infos.emplace_back(src_transformed, mask_transformed, angle, scale);
					}
				}
			}
		}
	};
}

} // namespace matchHL

