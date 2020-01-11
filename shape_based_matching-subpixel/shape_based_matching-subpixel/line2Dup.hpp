#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>

#include "MIPP/mipp.h"  // for SIMD in different platforms
#include "cuda_icp/icp.h"

namespace line2Dup
{

	struct DefectParms
	{
		int th;
		int width;
		int height;
		int size;
		int erode;

		bool bSurface;
		bool bGap;
		bool bBreak;
	};

struct FindObject
{
	std::vector<cv::Point2f> pts;
	float angle;
	float score;
	std::string classID;
	cv::Mat templAffine;
};

struct Feature
{
    int x;
    int y;
    int label;
	int theta;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    Feature() : x(0), y(0), label(0) {}
    Feature(int x, int y, int label);
};
inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

struct Template
{
    int width;
    int height;
    int tl_x;
    int tl_y;
    int pyramid_level;
    std::vector<Feature> features;

	//add by HuangLi 2019/08/08
	float scale;
	float angle;
	cv::Mat templImg;
	cv::Mat templMask;
	cv::Point pt0;
	cv::Point pt1;
	cv::Point pt2;
	cv::Point pt3;
	
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
};

class ColorGradientPyramid
{
public:
    ColorGradientPyramid(const cv::Mat &src, const cv::Mat &mask,
                                             float weak_threshold, size_t num_features,
                                             float strong_threshold);

    void quantize(cv::Mat &dst) const;

    bool extractTemplate(Template &templ, std::vector<cv::Point2f>& pts) const;

    void pyrDown();

public:
    void update();
    /// Candidate feature with a score
    struct Candidate
    {
		Candidate(int x, int y, int label, float score);

        /// Sort candidates with high score to the front
        bool operator<(const Candidate &rhs) const
        {
            return score > rhs.score;
        }

        Feature f;
        float score;
    };

	cv::Mat src;
	cv::Mat mask;

    int pyramid_level;
    cv::Mat angle;
    cv::Mat magnitude;
	cv::Mat angle_ori;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    static bool selectScatteredFeatures(const std::vector<Candidate> &candidates, std::vector<Feature> &features,
                                                                size_t num_features, float distance, cv::Mat& edgeMask);
};
inline ColorGradientPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

class ColorGradient
{
public:
    ColorGradient();
    ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

    std::string name() const;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;

    cv::Ptr<ColorGradientPyramid> process(const cv::Mat src, const cv::Mat &mask = cv::Mat()) const
    {
        return cv::makePtr<ColorGradientPyramid>(src, mask, weak_threshold, num_features, strong_threshold);
    }
};

struct Match
{
    Match()
    {
    }

    Match(int x, int y, float similarity, const std::string &class_id, int template_id, float angle);

    /// Sort matches with high similarity to the front
    bool operator<(const Match &rhs) const
    {
        // Secondarily sort on template_id for the sake of duplicate removal
        if (similarity != rhs.similarity)
            return similarity > rhs.similarity;
        else
            return template_id < rhs.template_id;
    }

    bool operator==(const Match &rhs) const
    {
        return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
    }

	int x;
    int y;
    float similarity;
    std::string class_id;
    int template_id;
	float angle;
};

inline Match::Match(int _x, int _y, float _similarity, const std::string &_class_id, int _template_id, float _angle)
        : x(_x), y(_y), similarity(_similarity), class_id(_class_id), template_id(_template_id), angle(_angle)
{
}

class Detector
{
public:
    /**
         * \brief Empty constructor, initialize with read().
         */
	Detector();

    Detector(std::vector<int> T);
	Detector(int num_features = 64, std::vector<int> T = { 4, 8 }, float weak_thresh = 10.0f, float strong_thresh = 20.0f);

	std::map<std::string, std::vector<Match> > match(cv::Mat sources, float threshold, const std::vector<std::string> &class_ids = std::vector<std::string>(), 
																		const cv::Mat masks = cv::Mat(), const float startAngle = 9999, const float endAngle = 9999);

    int addTemplate(const cv::Mat sources, const std::string &class_id,
                                    const cv::Mat &object_mask, const cv::Mat &edge_mask, int num_features = 0, cv::Rect* bounding_box = NULL,
									float angle = 0.0f, float scale = 1.0f, std::vector<cv::Point2f>& pts = std::vector<cv::Point2f>());

	//int addTemplate_rotate(const std::string &class_id, int zero_id, float theta, cv::Point2f center);
	int addTemplate_rotate(const cv::Mat& source, const cv::Mat &object_mask, const cv::Mat &edge_mask, const std::string &class_id, int zero_id, float theta, cv::Point2f center, cv::Rect* bounding_box,
												float angle = 0.0f, float scale = 1.0f, std::vector<cv::Point2f>& pts = std::vector<cv::Point2f>());

	//add by HuangLi 2019/08/13
	/**
	* \init ICP.
	* \param	     image  Source image.
	* \return		 ICP scene edge.
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
	cv::Mat getIcpAffine(Match& match, Scene_edge& scene, cuda_icp::RegistrationResult& result, bool& valid, cv::Rect& roi = cv::Rect());

	cv::Mat getIcpAffineDefect(const cv::Mat image, Match& match, Scene_edge& scene, cuda_icp::RegistrationResult& result, bool& valid, cv::Rect& roi = cv::Rect());

    const cv::Ptr<ColorGradient> &getModalities() const { return modality; }

    int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

    int pyramidLevels() const { return pyramid_levels; }

	const std::vector<Template> &getTemplates(const std::string &class_id, int template_id) const;

    int numTemplates() const;
    int numTemplates(const std::string &class_id) const;
    int numClasses() const { return static_cast<int>(class_templates.size()); }

    std::vector<std::string> classIds() const;

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs);

	void readClass(cv::FileStorage &fs);
    std::string readClass(const cv::FileNode &fn, const std::string &class_id_override = "");
    void writeClass(const std::string &class_id, cv::FileStorage &fs);

	int readClasses(const std::string str, std::vector<std::string> &class_ids = std::vector<std::string>());
	void writeClasses(const std::string str);

	//add by HuangLi 2019/08/14
	void createModel(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask = cv::Mat(), cv::Mat& invalidMask = cv::Mat(), std::string modelName = "train", cv::Rect roi = cv::Rect());
	int saveModel(std::string path);
	void detect(cv::Mat &src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, int amount = 10000, cv::Mat &mask = cv::Mat());

	void setClassID(std::vector<std::string>& classIds) { m_classIds = classIds; }
	std::vector<std::string> getClassID() { return m_classIds; }

	int getLowContrast() { return lowContrast; }
	void setLowContrast(float lowContrast) { this->lowContrast = lowContrast; }
	int getHighContrast() { return highContrast; }
	void setHighContrast(float highContrast) { this->highContrast = highContrast; }
	float getStartAngle() { return startAngle; }
	void  setStartAngle(float startAngle) { this->startAngle = startAngle; }
	float getEndAngle() { return endAngle; }
	void  setEndAngl(float endAngle) { this->endAngle = endAngle; }
	float getStepAngle() { return stepAngle; }
	void  setStepAngle(float stepAngle) { this->stepAngle = stepAngle; }
	float getStartScale() { return startScale; }
	void  setStartScale(float startScale) { this->startScale = startScale; }
	float getEndScale() { return endScale; }
	void  setEndScale(float endScale) { this->endScale = endScale; }
	float getStepScale() { return stepScale; }
	void  setStepScale(float stepScale) { this->stepScale = stepScale; }
	float getIcp() { return bIcp; }
	void  setIcp(bool bIcp) { this->bIcp = bIcp; }
	void getFeatures(const std::vector<Template>& templates, int num_modalities, std::vector<cv::Point>& pts, cv::Point offset);
	void drawResponse(const std::vector<Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset);
	void drawResponse(const std::vector<Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset, cuda_icp::RegistrationResult& icpAffine);

	void detectPose(cv::Mat& src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, int amount = 1, cv::Mat& mask = cv::Mat());
	void detectPoseAndDefect(cv::Mat& src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, int amount = 1, cv::Mat &mask = cv::Mat());

	int detectObject(cv::Mat& src, cv::Mat& dst, std::vector<FindObject>& findObjects, float matchThreshold, std::vector<std::string>& classIds, 
																			float startAngle = -360, float endAngle = 360,  int amount = 1, cv::Mat& mask = cv::Mat());
	//flag 是否归0角度统一计算
	int detectObjectDefect(cv::Mat& src, cv::Mat& dst, line2Dup::Template& templateInfo, float matchThreshold, int amount = 1, cv::Mat& mask = cv::Mat(), bool flag = 0);
	
	void getCurrentTemplImg(Template& templ);

	// 设置回调函数的函数
	void setCallbackFunc(void(*func) (cv::Mat& src, void* psend, void* precv));
private:
	void(*m_callbackFunc)(cv::Mat& src, void* psend, void* precv);

	void calSurface(cv::Mat& src, cv::Mat& templImg, cv::Mat& mask, cv::Mat& dst, std::vector<cv::Rect>& defectRois, cv::Point pt = cv::Point(0, 0));
	void calGap(cv::Mat& src, cv::Mat& mask, cv::Mat& dst, std::vector<cv::Rect>& defectRois, cv::Point pt = cv::Point(0, 0));

public:
	//DefectParms parms;
	std::map<std::string, std::map<std::string, int>> parms;
	cv::Mat m_templateImg;
	cv::Mat m_templMask;
	

private:
	int		lowContrast;
	int		highContrast;
	float	startAngle;
	float	endAngle;
	float	stepAngle;
	float	startScale;
	float	endScale;
	float	stepScale;
	float	matchThreshold;
	bool	bIcp;

	bool	bExitModel;
	std::vector<std::string> m_classIds;

protected:
    cv::Ptr<ColorGradient> modality;
    int pyramid_levels;
    std::vector<int> T_at_level;

    typedef std::vector<Template> TemplatePyramid;
    typedef std::map<std::string, std::vector<TemplatePyramid>> TemplatesMap;
    TemplatesMap class_templates;

    typedef std::vector<cv::Mat> LinearMemories;
    // Indexed as [pyramid level][ColorGradient][quantized label]
    typedef std::vector<std::vector<LinearMemories>> LinearMemoryPyramid;

    void matchClass(const LinearMemoryPyramid &lm_pyramid, const std::vector<cv::Size> &sizes, float threshold, std::vector<Match> &matches, const std::string &class_id,
                                    const std::vector<TemplatePyramid> &template_pyramids, const float startAngle = -9999, const float endAngle = 9999) const;
};

} // namespace line2Dup

namespace shape_based_matching {
class shapeInfo_producer{
public:
    cv::Mat src;
    cv::Mat mask;
	cv::Mat edgeMask;

    std::vector<float> angle_range;
    std::vector<float> scale_range;

    float angle_step = 15;
    float scale_step = 0.5;
    float eps = 1e-6;

    class Info
	{
    public:
        float angle;
        float scale;

        Info(float angle_, float scale_)
		{
            angle = angle_;
            scale = scale_;
        }
    };
    std::vector<Info> infos;

    shapeInfo_producer(cv::Mat& src, cv::Mat& mask = cv::Mat(), cv::Mat& edgeMask = cv::Mat())
	{

        this->src = src;
        if(mask.empty())
		{
            // make sure we have masks
            this->mask = cv::Mat(src.size(), CV_8UC1, {255});
        }
		else
		{
            this->mask = mask;
        }

		if (edgeMask.empty())
		{
			// make sure we have masks
			this->edgeMask = cv::Mat(src.size(), CV_8UC1, { 0 });
		}
		else
		{
			this->edgeMask = edgeMask;
		}
    }

    static cv::Mat transform(cv::Mat src, float angle, float scale)
	{
        cv::Mat dst;

        cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
        cv::warpAffine(src, dst, rot_mat, src.size());

        return dst;
    }

    static void save_infos(std::vector<shapeInfo_producer::Info>& infos, std::string path = "infos.yaml"){
        cv::FileStorage fs(path, cv::FileStorage::WRITE);

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
    static std::vector<Info> load_infos(std::string path = "info.yaml"){
        cv::FileStorage fs(path, cv::FileStorage::READ);

        std::vector<Info> infos;

        cv::FileNode infos_fn = fs["infos"];
        cv::FileNodeIterator it = infos_fn.begin(), it_end = infos_fn.end();
        for (int i = 0; it != it_end; ++it, i++)
        {
            infos.emplace_back(float((*it)["angle"]), float((*it)["scale"]));
        }
        return infos;
    }

    void produce_infos()
	{
        assert(angle_range.size() <= 2);
        assert(scale_range.size() <= 2);
        assert(angle_step > eps*10);
        assert(scale_step > eps*10);

        // make sure range not empty
        if(angle_range.size() == 0)
		{
            angle_range.push_back(0);
        }
        if(scale_range.size() == 0)
		{
            scale_range.push_back(1);
        }

        if(angle_range.size() == 1 && scale_range.size() == 1)
		{
            float angle = angle_range[0];
            float scale = scale_range[0];
            infos.emplace_back(angle, scale);

        }
		else if(angle_range.size() == 1 && scale_range.size() == 2)
		{
            assert(scale_range[1] > scale_range[0]);
            float angle = angle_range[0];
            for(float scale = scale_range[0]; scale <= scale_range[1] + eps; scale += scale_step)
			{
                infos.emplace_back(angle, scale);
            }
        }
		else if(angle_range.size() == 2 && scale_range.size() == 1)
		{
            assert(angle_range[1] > angle_range[0]);
            float scale = scale_range[0];

			if (angle_range[0] * angle_range[1] < 0)
			{
				//把0角度放到第一个，为了显示， 考虑step位数四舍五入，适当加大范围
				infos.emplace_back(0.0f, scale);
				for (float angle = -angle_step; angle >= angle_range[0] - eps; angle -= angle_step)
				{
					infos.emplace_back(angle, scale);
				}
				for (float angle = angle_step; angle <= angle_range[1] + eps; angle += angle_step)
				{
					infos.emplace_back(angle, scale);
				}
				/*for (float angle = angle_range[0]; angle <= angle_range[1] + eps; angle += angle_step)
				{
					infos.emplace_back(angle, scale);
				}*/
			}
			else
			{
				for (float angle = angle_range[0]; angle <= angle_range[1] + eps; angle += angle_step)
				{
					infos.emplace_back(angle, scale);
				}
			}

        }
		else if(angle_range.size() == 2 && scale_range.size() == 2)
		{
            assert(scale_range[1] > scale_range[0]);
            assert(angle_range[1] > angle_range[0]);

			if (angle_range[0] * angle_range[1] < 0)
			{
				//把0角度放到第一个，为了显示
				for (float scale = scale_range[0]; scale <= scale_range[1] + eps; scale += scale_step)
				{
					infos.emplace_back(0.0f, scale);
					for (float angle = -angle_step; angle >= angle_range[0] - eps; angle -= angle_step)
					{
						infos.emplace_back(angle, scale);
					}
					for (float angle = angle_step; angle <= angle_range[1] + eps; angle += angle_step)
					{
						infos.emplace_back(angle, scale);
					}
				}

				/*for (float scale = scale_range[0]; scale <= scale_range[1] + eps; scale += scale_step)
				{
					for (float angle = angle_range[0]; angle <= angle_range[1] + eps; angle += angle_step)
					{
						infos.emplace_back(angle, scale);
					}
				}*/
			}
			else
			{
				for (float scale = scale_range[0]; scale <= scale_range[1] + eps; scale += scale_step)
				{
					for (float angle = angle_range[0]; angle <= angle_range[1] + eps; angle += angle_step)
					{
						infos.emplace_back(angle, scale);
					}
				}
			}

        }
    }

    cv::Mat src_of(const Info& info)
	{
        return transform(src, info.angle, info.scale);
    }

    cv::Mat mask_of(const Info& info)
	{
        return transform(mask, info.angle, info.scale);
    }

	cv::Mat edgeMask_of(const Info& info)
	{
		return transform(edgeMask, info.angle, info.scale);
	}
};

}

#endif
