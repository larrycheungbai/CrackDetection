#ifndef CRACK_FEATURES_H_
#define CRACK_FEATURES_H_


//opencv headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <cv.h>
#include <limits>
#include <fstream>
#include <iostream>
#include <direct.h>
#include <io.h>
using namespace cv;
using namespace std;

class CrackFeatures
{
private:
	vector<Mat> m_filterBank;
	Mat  m_textonDict;
	Mat  m_minMat;
	Mat  m_maxMat;
	int  m_textonNumber;
	int  m_featureVectNumber;
	int  m_filterSize;
private:
	void loadScalingMats();
	void loadTextonDictionary();
	void loadFilterBank();
public:
	CrackFeatures();
//	vector<float> getFilterResponses(Mat & inputImage);
	vector<int> getTextonHist(const Mat & inputImage, int stepSize);
	Mat genFeatureVec(const Mat & imgPatch);
	void genHsvHist(const Mat & imagePatch, Mat & h_hist, Mat & s_hist);
	Mat getFilterResponsesMap(const Mat inputImage, int stepSize);
	int getFeatureVectorLength()
	{
		return m_featureVectNumber;
	}
	int getFilterSize()
	{
		return m_filterSize;
	}
	int getTextonNumber()
	{
		return m_textonNumber;
	}
};


#endif