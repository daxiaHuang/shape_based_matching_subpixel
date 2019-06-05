#include "detector.h"
#include ".\\match\\match.h"
#include <atlstr.h>

static void TcharToString(TCHAR* tchar, char * _char)
{
	int iLen = WideCharToMultiByte(CP_ACP, 0, tchar, -1, NULL, 0, NULL, NULL);
	WideCharToMultiByte(CP_ACP, 0, tchar, -1, _char, iLen, NULL, NULL);
}

Detector::Detector(std::string modelName)
{
	m_modelType = modelName;
}

Detector::~Detector()
{

}

int Detector::preparatory(cv::Mat& src)
{
	return 1;
}

void Detector::setIDstring(std::string IDstring)
{

}

void Detector::setParameters(int flag_save, std::string dir)
{

}

int Detector::calibration(cv::Mat& src, cv::Mat& dst, std::string name)
{
	if (src.channels() == 1)
	{
		cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
	}
	else
	{
		dst = src;
	}

	return 1;
}

bool Detector::loadParameters(std::map<std::string, std::map<std::string, double>> &parameters)
{
	
	m_parameters = parameters;

	loadRoiParm();
	return true;
}

int Detector::loadRoiParm()
{
	TCHAR szKeyValue[MAX_PATH] = { 0 };
	char filename[256];
	sprintf(filename, ".\\Parameter\\%s.ini", m_modelType.c_str());
	CString szIniPath(filename);
	if (!::PathFileExists(szIniPath))
	{
		return 0;
	}

	int i, j, total, num, type, x, y, width, height;
	CString str, strSub;
	char strVal[256];
	cv::Rect tempRect;

	//º”‘ÿ—µ¡∑«¯”Ú
	str.Format(_T("Template"));
	num = ::GetPrivateProfileInt(str, _T("num"), 0, szIniPath);
	if (num == 0)
	{
		return 0;
	}
	strSub.Format(_T("roi%02d"), 0);
	::GetPrivateProfileString(str, strSub, NULL, szKeyValue, MAX_PATH, szIniPath);
	TcharToString(szKeyValue, strVal);
	x = y = width = height = 0;
	if (strlen(strVal) != 0)
	{
		sscanf_s(strVal, "%d|%d|%d|%d", &x, &y, &width, &height);
		if (x <= 0 || y <= 0)
		{
			return 0;
		}
		m_templRect = cv::Rect(x, y, width, height);
	}


	//º”‘ÿºÏ≤‚«¯”Ú
	str.Format(_T("Detect"));
	num = ::GetPrivateProfileInt(str, _T("num"), 0, szIniPath);
	if (num == 0)
	{
		return 0;
	}
	strSub.Format(_T("roi%02d"), 0);
	::GetPrivateProfileString(str, strSub, NULL, szKeyValue, MAX_PATH, szIniPath);
	TcharToString(szKeyValue, strVal);
	x = y = width = height = 0;
	if (strlen(strVal) != 0)
	{
		sscanf_s(strVal, "%d|%d|%d|%d", &x, &y, &width, &height);
		if (x <= 0 || y <= 0)
		{
			return 0;
		}
		m_roiRect = cv::Rect(x, y, width, height);
	}

	return 1;
}

int Detector::Detect_main(cv::Mat &src, cv::Mat &dst, std::string name, std::vector<aContour>& contourVec, std::vector<aContour>& contourVec1)
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

	int method = m_parameters["A_Common"]["G_method"];
	int numFeature = m_parameters["B_train"]["G_numFeature"];
	float startAngle, endAngle, stepAngle, startScale, endScale, stepScale;
	startAngle = m_parameters["B_train"]["A_startAngle"];
	endAngle = m_parameters["B_train"]["B_endAngle"];
	stepAngle = m_parameters["B_train"]["C_stepAngle"];
	startScale = m_parameters["B_train"]["D_startScale"];
	endScale = m_parameters["B_train"]["E_endScale"];
	stepScale = m_parameters["B_train"]["F_stepScale"];
	int weakThreah = m_parameters["B_train"]["H_weak"];
	int strongThreah = m_parameters["B_train"]["I_strong"];

	CmatchHL *match = new CmatchHL();
	if (method == 0)//—µ¡∑
	{
		cv::Mat roiImg = grayImg(m_templRect);
		match->setParmeter(startAngle, endAngle, stepAngle, startScale, endScale, stepScale, numFeature, weakThreah, strongThreah);
		match->creatModel(roiImg, cv::Mat(), "object");
		cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
	}
	else  if (method == 1)//≤‚ ‘
	{
		cv::Mat roiImg = grayImg(m_roiRect);
		match->setParmeter(startAngle, endAngle, stepAngle, startScale, endScale, stepScale, numFeature, weakThreah, strongThreah);
		match->loadModel("object");
		int padding = 50;
		cv::Mat padded_img = cv::Mat(roiImg.rows + 2 * padding, roiImg.cols + 2 * padding, roiImg.type(), cv::Scalar::all(0));
		roiImg.copyTo(padded_img(cv::Rect(padding, padding, roiImg.cols, roiImg.rows)));
		int stride = 16;
		int n = padded_img.rows / stride;
		int m = padded_img.cols / stride;
		cv::Rect roi = cv::Rect(0, 0, stride*m, stride*n);
		roiImg = padded_img(roi).clone();
		match->detect(roiImg, dst, "object");
	}
	return 1;
}
