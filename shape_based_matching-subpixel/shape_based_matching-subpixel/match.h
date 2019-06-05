#ifndef MATCH_H
#define MATCH_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include ".\\match\\line2Dup.h"

//模型结构
typedef struct
{

}sModex;

//模型集合
typedef struct
{
	std::vector<sModex> objVec;
}sModeObj;

//匹配结果
typedef struct
{
	std::string classID;//目标类别
	cv::Rect rect;//目标位置
	float similarity;//目标相似度


}sRect;

class CmatchBase
{
public:
	CmatchBase() {};
	virtual ~CmatchBase() {};

	virtual int creatModel(cv::Mat &src, cv::Mat &mask = cv::Mat(), std::string classID = "") = 0;

	virtual int detect(cv::Mat &src, cv::Mat &mask = cv::Mat(), std::string classID = "") = 0;

protected:

	sModeObj m_sModeObj;//生成的模型
	std::vector<sRect> m_sRectVec;//匹配结果框

};

class CmatchHL :public CmatchBase
{
public:
	CmatchHL();
	~CmatchHL();

	int creatModel(cv::Mat &src, cv::Mat &mask = cv::Mat(), std::string classID = "");
	int loadModel(std::string classID = "");
	int detect(cv::Mat &src, cv::Mat &dst, std::string classID = "");
	int	setParmeter(float startAngle, float endAngle, float stepAngle, float startScale, float endScale, float stepScale, int numFeatures);

	//参数
	float	m_similarity;
	float	m_startAngle;
	float	m_endAngle;
	float	m_stepAngle;
	float	m_startScale;
	float	m_endScale;
	float	m_stepScale;
	int		m_numFeatures;
	line2Dup::Detector m_detector;
	std::vector<std::string> m_ids;
	std::vector<shape_based_matching::shapeInfo_producer::Info>  m_infos;

};

//
#endif
