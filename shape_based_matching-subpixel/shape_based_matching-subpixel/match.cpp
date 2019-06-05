#include "match.h"
#include "line2Dup.h"
#include "cuda_icp/icp.h"

static std::string prefix = ".\\template\\";
static bool viewICP = true;
static int padding = 50;

template <class Type, class T> struct Cmp
{
	Cmp(Type p)
	{
		this->p = p;
	}
	bool operator()(T b1, T b2)
	{
		Type dis = pow(float(b2.x - b1.x), 2.0) + pow(float(b2.y - b1.y), 2.0);
		return dis < p * p;
	}
	Type p;
};

template<class T> void cluster(const std::vector<T>& vecSrc, std::vector<std::vector<T>>& vecDst, double dDis)
{
	if (vecSrc.size() == 0)
	{
		return;
	}
	int numbb = vecSrc.size();
	std::vector<int> vecIndex;
	int c = 1;
	switch (numbb)
	{
	case 1:
		vecDst = std::vector<std::vector<T>>(1);
		vecDst.push_back(vecSrc);
		return;
		break;
	default:
		vecIndex = std::vector<int>(numbb, 0);
		c = cv::partition(vecSrc, vecIndex, Cmp<double, T>(dDis));
		break;
	}

	cv::Point tempPoint;
	vecDst = std::vector<std::vector<T>>(c);
	for (int i = 0; i < c; i++)
	{
		std::vector<T> pts;
		for (int j = 0; j < vecIndex.size(); j++)
		{
			if (vecIndex[j] == i)
			{
				pts.push_back(vecSrc[j]);
			}
		}
		vecDst[c - i - 1] = pts;
	}
}

CmatchHL::CmatchHL()
{

}

CmatchHL::~CmatchHL()
{

}

int	CmatchHL::setParmeter(float startAngle, float endAngle, float stepAngle, float startScale, float endScale, float stepScale, int numFeatures)
{
	m_startAngle = startAngle;
	m_endAngle = endAngle;
	m_stepAngle = stepAngle;
	m_startScale = startScale;
	m_endScale = endScale;
	m_stepScale = stepScale;
	m_numFeatures = numFeatures;

	return 1;
}

int CmatchHL::creatModel(cv::Mat &src, cv::Mat &mask, std::string classID)
{
	if (src.empty())
	{
		return 0;
	}

	if (mask.empty())
	{
		mask = cv::Mat(src.size(), CV_8UC1, 255);
	}
	
	//line2Dup::Detector detector(m_numFeatures, { 4, 8 });
	line2Dup::Detector detector(m_numFeatures, { 4 });

	// padding to avoid rotating out
	cv::Mat padded_img = cv::Mat(src.rows + 2 * padding, src.cols + 2 * padding, src.type(), cv::Scalar::all(0));
	src.copyTo(padded_img(cv::Rect(padding, padding, src.cols, src.rows)));

	cv::Mat padded_mask = cv::Mat(mask.rows + 2 * padding, mask.cols + 2 * padding, mask.type(), cv::Scalar::all(0));
	mask.copyTo(padded_mask(cv::Rect(padding, padding, src.cols, src.rows)));

	shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
	shapes.angle_range = { m_startAngle, m_endAngle };
	shapes.angle_step = m_stepAngle;
	shapes.scale_range = { m_startScale, m_endScale };
	shapes.scale_step = m_stepScale;

	shapes.produce_infos();
	std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
	std::string class_id = classID;
	for (auto& info : shapes.infos) {
		imshow("train", shapes.src_of(info));
		cv::waitKey(10);
		std::cout << "\ninfo.angle: " << info.angle << std::endl;
		int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
		std::cout << "templ_id: " << templ_id << std::endl;
		if (templ_id != -1) {
			infos_have_templ.push_back(info);
		}
	}
	detector.writeClasses(prefix + "%s_templ.yaml");
	std::stringstream str;
	str << prefix << classID << "_info.yaml";
	shapes.save_infos(infos_have_templ, str.str());
	std::cout << "create model end" << std::endl << std::endl;


	return 1;
}

int CmatchHL::loadModel(std::string classID)
{
	//m_detector = line2Dup::Detector(m_numFeatures, { 4, 8 });
	m_detector = line2Dup::Detector(m_numFeatures, { 4 });
	m_ids.push_back(classID);
	m_detector.readClasses(m_ids, prefix + "%s_templ.yaml");
	// angle & scale are saved here, fetched by match id
	std::stringstream str;
	str << prefix << classID << "_info.yaml";
	m_infos = shape_based_matching::shapeInfo_producer::load_infos(str.str());

	return 1;
}


int CmatchHL::detect(cv::Mat &src, cv::Mat &dst, std::string classID)
{
	if (src.empty())
	{
		return 0;
	}

	if (src.channels() == 1)
	{
		cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
	}
	else
	{
		dst = src.clone();
	}

	std::vector<std::string> ids;
	ids.push_back(classID);
	auto matches = m_detector.match(src, 90, ids);
	std::cout << "matches.size(): " << matches.size() << std::endl;

	//cluster by diatance of Match.(x, y)
	std::vector<std::vector<line2Dup::Match> > clusterMatch;
	cluster<line2Dup::Match>(matches, clusterMatch, 30);
	matches.clear();
	for (int i = 0; i < clusterMatch.size(); i++)
	{
		int index = -1;
		double similarity, maxmSimilarity = 0;
		for (int j = 0; j < clusterMatch[i].size(); j++)
		{
			similarity = clusterMatch[i][j].similarity;
			if (similarity > maxmSimilarity)
			{
				maxmSimilarity = similarity;
				index = j;
			}
		}
		matches.push_back(clusterMatch[i][index]);
	}

	size_t top5 = matches.size();
	if (top5 > matches.size()) top5 = matches.size();

	// construct scene
	Scene_edge scene;
	// buffer
	std::vector<::Vec2f> pcd_buffer, normal_buffer;
	scene.init_Scene_edge_cpu(src, pcd_buffer, normal_buffer);

	cv::Mat edge_global;  // get edge
	{
		cv::Mat gray;
		if (src.channels() > 1) {
			cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
		}
		else {
			gray = src;
		}

		cv::Mat smoothed = gray;
		cv::Canny(smoothed, edge_global, 30, 60);

		if (edge_global.channels() == 1) cvtColor(edge_global, edge_global, cv::COLOR_GRAY2BGR);
	}

	for (int i = top5 - 1; i >= 0; i--)
	{
		cv::Mat edge = edge_global.clone();

		auto match = matches[i];
		auto templ = m_detector.getTemplates(classID, match.template_id);

		// 270 is width of template image
		// 100 is padding when training
		// tl_x/y: template croping topleft corner when training

		float r_scaled = m_infos[match.template_id].scale;

		// scaling won't affect this, because it has been determined by warpAffine
		// cv::warpAffine(src, dst, rot_mat, src.size()); last param
		float train_img_half_width = templ[0].width / 2.0f + padding;
		float train_img_half_height = templ[0].height / 2.0f + padding;

		// center x,y of train_img in test img
		cv::Point center(0, 0);
		center.x = match.x - templ[0].tl_x + train_img_half_width;
		center.y = match.y - templ[0].tl_y + train_img_half_height;

		std::vector<::Vec2f> model_pcd(templ[0].features.size());
		for (int i = 0; i < templ[0].features.size(); i++) {
			auto& feat = templ[0].features[i];
			model_pcd[i] = {
				float(feat.x + match.x),
				float(feat.y + match.y)
			};
		}
		cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);

		cv::Vec3b randColor;
		randColor[0] = 0;
		randColor[1] = 0;
		randColor[2] = 255;
		for (int i = 0; i < templ[0].features.size(); i++)
		{
			auto feat = templ[0].features[i];
			cv::circle(dst, { feat.x + match.x, feat.y + match.y }, 2, randColor, -1);
		}

		/*if (viewICP) 
		{
			imshow("icp", edge);
			cv::waitKey(0);
		}*/

		randColor[0] = 0;
		randColor[1] = 255;
		randColor[2] = 0;

		float error = 0.0;
		for (int i = 0; i < templ[0].features.size(); i++) 
		{
			auto feat = templ[0].features[i];
			float x = feat.x + match.x;
			float y = feat.y + match.y;
			float new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
			float new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
			cv::circle(dst, { int(new_x + 0.5f), int(new_y + 0.5f) }, 2, randColor, -1);

			error += ((x - new_x) + (y - new_y)) / 2;
		}
		error /= templ[0].features.size();

		//增加：通过误差判断是否ICP有效
		if (error < 3.0)
		{
			double init_angle = m_infos[match.template_id].angle;
			init_angle = init_angle >= 180 ? (init_angle - 360) : init_angle;

			double ori_diff_angle = init_angle;
			double icp_diff_angle = -std::atan(result.transformation_[1][0] / result.transformation_[0][0]) / CV_PI * 180 + init_angle;
			double improved_angle = icp_diff_angle - ori_diff_angle;

			if (viewICP)
			{
				float new_x = result.transformation_[0][0] * center.x + result.transformation_[0][1] * center.y + result.transformation_[0][2];
				float new_y = result.transformation_[1][0] * center.x + result.transformation_[1][1] * center.y + result.transformation_[1][2];
				cv::RotatedRect rotatedRectangle(cv::Point2f(new_x, new_y), cv::Size2f(2 * r_scaled * train_img_half_width, 2 * r_scaled * train_img_half_height), -icp_diff_angle);

				cv::Point2f pt[4];
				rotatedRectangle.points(pt);
				cv::line(dst, pt[0], pt[1], cv::Scalar(0, 0, 255));
				cv::line(dst, pt[1], pt[2], cv::Scalar(0, 0, 255));
				cv::line(dst, pt[2], pt[3], cv::Scalar(0, 0, 255));
				cv::line(dst, pt[3], pt[0], cv::Scalar(0, 0, 255));

				//imshow("icp", dst);
				//cv::waitKey(0);
			}

			std::cout << "\n---------------" << std::endl;
			std::cout << "scale: " << std::sqrt(result.transformation_[0][0] * result.transformation_[0][0] +
				result.transformation_[1][0] * result.transformation_[1][0]) << std::endl;
			std::cout << "init diff angle: " << ori_diff_angle << std::endl;
			std::cout << "improved angle: " << improved_angle << std::endl;
			std::cout << "match.template_id: " << match.template_id << std::endl;
			std::cout << "match.similarity: " << match.similarity << std::endl;
			std::cout << "origin angle: " << ori_diff_angle << "  affine angle: " << icp_diff_angle << std::endl;
		}
		else
		{
			if (viewICP)
			{
				float new_x = result.transformation_[0][0] * center.x + result.transformation_[0][1] * center.y + result.transformation_[0][2];
				float new_y = result.transformation_[1][0] * center.x + result.transformation_[1][1] * center.y + result.transformation_[1][2];
				cv::RotatedRect rotatedRectangle(cv::Point2f(center.x, center.y), cv::Size2f(2 * r_scaled * train_img_half_width, 2 * r_scaled * train_img_half_height), -m_infos[match.template_id].angle);

				cv::Point2f pt[4];
				rotatedRectangle.points(pt);
				cv::line(dst, pt[0], pt[1], cv::Scalar(0, 0, 255));
				cv::line(dst, pt[1], pt[2], cv::Scalar(0, 0, 255));
				cv::line(dst, pt[2], pt[3], cv::Scalar(0, 0, 255));
				cv::line(dst, pt[3], pt[0], cv::Scalar(0, 0, 255));

				//imshow("icp", dst);
				//cv::waitKey(0);
			}

			std::cout << "\n---------------" << std::endl;
			std::cout << "scale: " << std::sqrt(result.transformation_[0][0] * result.transformation_[0][0] +
				result.transformation_[1][0] * result.transformation_[1][0]) << std::endl;
			std::cout << "init diff angle: " << -m_infos[match.template_id].angle << std::endl;
			std::cout << "match.template_id: " << match.template_id << std::endl;
			std::cout << "match.similarity: " << match.similarity << std::endl;
		}
		
	}

	std::cout << "test end" << std::endl << std::endl;


	return 1;
}