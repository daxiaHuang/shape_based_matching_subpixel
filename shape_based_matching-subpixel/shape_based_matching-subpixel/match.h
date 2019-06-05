#ifndef MATCH_H
#define MATCH_H

#include <vector>
#include <opencv2/opencv.hpp>


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

}sRect;

class CmatchBase
{
public:
	CmatchBase() {}
	virtual ~CmatchBase() {}

	virtual int creatModel(cv::Mat &src, cv::Mat &mask = cv::Mat()) = 0;

	virtual int detect(cv::Mat &src, cv::Mat &mask = cv::Mat()) = 0;

protected:

	sModeObj m_sModeObj;//生成的模型
	std::vector<sRect> m_sRectVec;//匹配结果框

};

class CmatchX :public CmatchBase
{
public:
	CmatchX() {}
	~CmatchX() {}

	int creatModel(cv::Mat &src, cv::Mat &mask = cv::Mat()) = 0;
	int detect(cv::Mat &src, cv::Mat &mask = cv::Mat()) = 0;

	//参数
	int a;
	int b;

};

//
#endif
