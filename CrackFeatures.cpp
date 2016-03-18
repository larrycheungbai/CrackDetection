#include "CrackFeatures.h"
#include "VIA_Utilities.h"
#include "LBP.h"
#include "histogram.h"
using namespace VIA_Utilities;

/*
Author:			Lei Zhang
Function name:  CrackFeatures()
Date Created:	08-21-2015
Description:	Constructor of the crack features class.
Input:          const char* filename, the file name.
Output:			None
*/
CrackFeatures::CrackFeatures()
{
	//basis of the texton
	m_textonNumber = 20;
	//lenth of the hand-crafted feature vector
	m_featureVectNumber = 93;
	//size of the 
	m_filterSize = 65;
	//load the pre-learned filter bank.
	loadFilterBank();
	//load the pre-trained texton dictionary.
	loadTextonDictionary();
	//load the matrices for scaling the matrix (for SVM)
	loadScalingMats();
}
void CrackFeatures::loadFilterBank()
{
	int kernel_size = m_filterSize;
	for (int i = 0; i < 48; i++)
	{
		Mat filter = Mat(kernel_size, kernel_size, CV_32F);
		string filter_image_name_txt = cv::format("Filter_Bank/Filter_bank_%d.txt", i + 1);
		readMatFromFile(filter, filter_image_name_txt.c_str());
		m_filterBank.push_back(filter);
	}
}
void CrackFeatures::loadScalingMats()
{
	m_minMat = Mat(1, m_featureVectNumber, CV_32F);
	m_maxMat = Mat(1, m_featureVectNumber, CV_32F);
	readMatFromFile(m_minMat, "data_scaling/train_dataset_min_mat.txt");
	readMatFromFile(m_maxMat, "data_scaling/train_dataset_max_mat.txt");
}
void CrackFeatures::loadTextonDictionary()
{
	m_textonDict = Mat(m_textonNumber, 48, CV_32F);
	readMatFromFile(m_textonDict, "Filter_Bank/texton_dictionary_20.txt");
}
/*
Author:			Lei Zhang
Function name:  getFilterResponses(Mat & inputImage)
Date Created:	08-21-2015
Description:	get the filter responses of the input Image patch form the filter bank.
Input:			cv::Mat & nputImage.
Output:			the vector stored the responses from the filter bank.
*/
/*
vector<float> CrackFeatures::getFilterResponses(Mat & inputImage) // vector<float> & vec_filterResponse)
{
if (m_filterBank.empty())
{
cout << "The filter bank is empty, please initalize it first!";// vector for Samples or Labels is empty!");
vector<float> vec_filterResponse(m_textonDict.cols, 0);
return vec_filterResponse;
}
flip(inputImage, inputImage, 1);
flip(inputImage, inputImage, 0);
Mat grayInputImage;
cvtColor(inputImage, grayInputImage, CV_RGB2GRAY);
cv::Mat input_f;
grayInputImage.convertTo(input_f, CV_64F);
//	int center_row = (inputImage.rows - 1) / 2;
//	int center_col = (inputImage.cols - 1) / 2;
Mat filter_response_mat = Mat(m_textonDict.rows, m_textonDict.cols, m_textonDict.type());
filter_response_mat.setTo(0);
for (int i = 0; i < m_filterBank.size(); i++)
{
Mat output;
// result = filter .* inputImage(u-x,v - y);
// sum(sum(result))
Mat filter_f;
m_filterBank[i].convertTo(filter_f, CV_64F);
multiply(input_f, filter_f, output);
float filter_response = sum(output)[0];
//vec_filterResponse.push_back(filter_response);
filter_response_mat.col(i).setTo(filter_response);
}
//lookup the texton dictionary
//Mat filter_response_vec = Mat(1, vec_filterResponse.size(), CV_32FC1);
//memcpy(filter_response_vec.data, vec_filterResponse.data(), vec_filterResponse.size()*sizeof(float));
//Mat filter_response_mat = repeat(filter_response_vec, 40, 1);
subtract(filter_response_mat, m_textonDict, filter_response_mat);
multiply(filter_response_mat, filter_response_mat, filter_response_mat);
//find the nearest distance between the fiter_response and the
double shortest_dist = DBL_MAX;
int text_idx = 0;
for (int i = 0; i < filter_response_mat.rows; i++)
{
double distance_text = sum(filter_response_mat.row(i))[0];
if (shortest_dist  > distance_text)
{
shortest_dist = distance_text;
text_idx = i;
}
}
const float * pAtom = m_textonDict.ptr<float>(text_idx);
vector<float> vec_filterResponse(pAtom, pAtom + m_textonDict.cols);
return vec_filterResponse;
}
*/

/*
Author:			Lei Zhang
Function name:  genFeatureVec(Mat & imgPatch)
Date Created:	08-21-2015
Description:	generate feature vector of a given image patch.
Input:			cv::Mat & imgPatch.
Output:			the feature vector are stored into a row matrix.
*/
Mat CrackFeatures::genFeatureVec(const Mat & imgPatch)
{
	vector<Mat> bgr_planes;
	vector<float>  vec_feature;
	Mat dst;
	Mat olbp; // lbp image
	Mat uniformLBPDescriptor;
	split(imgPatch, bgr_planes);
	//(1)mean of each channel in RGB space   3
	float meanB = mean(bgr_planes[0])[0];
	float meanG = mean(bgr_planes[1])[0];
	float meanR = mean(bgr_planes[2])[0];
	vec_feature.push_back(meanB);
	vec_feature.push_back(meanG);
	vec_feature.push_back(meanR);
	vector<Mat > hsv_planes;
	Mat hsv_image;
	cvtColor(imgPatch, hsv_image, CV_BGR2HSV);
	split(hsv_image, hsv_planes);
	//(2)mean of each channel in HSV space 3
	float meanH = mean(hsv_planes[0])[0];
	float meanS = mean(hsv_planes[1])[0];
	float meanV = mean(hsv_planes[2])[0];
	vec_feature.push_back(meanH);
	vec_feature.push_back(meanS);
	vec_feature.push_back(meanV);
	////(3)Hue Histogram = 5 (4) Saturation Histogram = 3
	Mat h_hist;
	Mat s_hist;
	genHsvHist(imgPatch, h_hist, s_hist);
	const float * p_h_hist = h_hist.ptr<float>(0);
	const float * p_s_hist = s_hist.ptr<float>(0);
	const int offset_h = std::max(h_hist.cols, h_hist.rows);
	const int offset_s = std::max(s_hist.cols, s_hist.rows);
	vector<float> vec_h_hist(p_h_hist, p_h_hist + offset_h);
	vector<float> vec_s_hist(p_s_hist, p_s_hist + offset_s);
	vec_feature.insert(vec_feature.end(), vec_h_hist.begin(), vec_h_hist.end());
	vec_feature.insert(vec_feature.end(), vec_s_hist.begin(), vec_s_hist.end());
	//(5)Local Binary Pattern  59
	cv::cvtColor(imgPatch, dst, CV_BGR2GRAY);
	GaussianBlur(dst, dst, Size(7, 7), 5, 3, BORDER_CONSTANT);
	lbp::OLBP(dst, olbp);
	vector<int> vec_uniform_lbp = lbp::uniform59Descriptor(olbp);;
	//insert the uniformed lbp code
	vec_feature.insert(vec_feature.end(), vec_uniform_lbp.begin(), vec_uniform_lbp.end());
	//(6)Texton Histogram = 20   
	//insert the filter bank response
	vector<int> vec_texton_hist = getTextonHist(imgPatch, 5);
	vec_feature.insert(vec_feature.end(), vec_texton_hist.begin(), vec_texton_hist.end());
	//(1)mean(RGB) = 3     (2)mean(HSV)= 3  
	//(3)Hue Histogram = 5 (4) Saturation Histogram = 3
	//(5)LBP = 59          (6)Texton Histogram = 20   
	//3 + 3 + 5 + 3 + 59 + 20 = 93
	Mat feature_vec = Mat(1, vec_feature.size(), CV_32FC1);
	memcpy(feature_vec.data, vec_feature.data(), vec_feature.size()*sizeof(float));
	return feature_vec;
}
/*
Author:			Lei Zhang
Function name:  genHsvHist(const Mat & imagePatch, Mat & h_hist, Mat & s_hist)
Date Created:	08-21-2015
Description:	generate histgrams HSV color space.
Input:			cv::Mat & imagePatch.
Input:          Mat & h_hist, used to store the hue histgram
Input:          Mat & s_hist, used to store the saturation histgram
Output:			None
*/
void CrackFeatures::genHsvHist(const Mat & imagePatch, Mat & h_hist, Mat & s_hist)
{
	Mat hsv_base;
	cvtColor(imagePatch, hsv_base, CV_BGR2HSV);
	vector<Mat > hsv_planes;
	split(hsv_base, hsv_planes);
	const int h_bins = 5;
	const int s_bins = 3;
	float h_ranges[] = { 0, 256};
	const float * p_h_ranges[] = { h_ranges };
	float s_ranges[] = { 0, 180};
	const float * p_s_ranges[] = { s_ranges };
	MatND hist_base;
	bool uniform = true;
	bool accumulate = false;
	//for the hue
	calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &h_bins, p_h_ranges, uniform, accumulate);
	//for the Saturation
	calcHist(&hsv_planes[1], 1, 0, Mat(), s_hist, 1, &s_bins, p_s_ranges, uniform, accumulate);
}


/*
Author:			Lei Zhang
Function name:  getTextonHist(const Mat & inputImage, int stepSize) 
Date Created:	08-21-2015
Description:	generate texton histgram.
Input:			cv::Mat & inputImage, image patch.
Input:          int stepSize, step size to generate the filter response map.
Output:			texton histgram stored in vector
*/

//assume the patch is bigger than the filter
//returen the texton histgram to see the distribution
vector<int> CrackFeatures::getTextonHist(const Mat & inputImage, int stepSize) // vector<float> & vec_filterResponse)
{
	//test_text_hist.txt
	Mat filter_response_maps = getFilterResponsesMap(inputImage, stepSize);
	int responses_num = filter_response_maps.rows;
	Mat filter_response_mat = Mat(m_textonDict.rows, m_textonDict.cols, m_textonDict.type());
	filter_response_mat.setTo(0);
	vector<int> texton_hist(m_textonNumber, 0);
	for (int i = 0; i <responses_num; i++)
	{
		//lookup the nearest centroid of the k-means
		filter_response_mat = repeat(filter_response_maps.row(i), m_textonDict.rows, 1);
		subtract(filter_response_mat, m_textonDict, filter_response_mat);
		multiply(filter_response_mat, filter_response_mat, filter_response_mat);
		double shortest_dist = DBL_MAX;
		int text_idx = 0;
		for (int k = 0; k < filter_response_mat.rows; k++)
		{
			double distance_text = sum(filter_response_mat.row(k))[0];
			if (shortest_dist  > distance_text)
			{
				shortest_dist = distance_text;
				text_idx = k;
			}
		}
		texton_hist[text_idx] ++;
	}
	return texton_hist;
}

/*
Author:			Lei Zhang
Function name:  getFilterResponsesMap(const Mat inputImage, int stepSize)
Date Created:	08-21-2015
Description:	generate filter response map.
Input:			cv::Mat & inputImage, image patch.
Input:          int stepSize, step size to generate the filter response map.
Output:			filter response map is stored in mat.
*/
//this filter response map is a matrix
//assume the patch is bigger than the filter
//currently the stepSize is 5 (preferred)
Mat CrackFeatures::getFilterResponsesMap(const Mat inputImage, int stepSize)
{
	if (m_filterBank.empty())
	{

		cout << "The filter bank is empty!" << endl;
		cout << " Please initialize the filter bank first" << endl;
	}
	//each row is responses from all the filters in the filter bank.
	int responseNum = m_filterBank.size();
	//
	Mat grayInputImage;
	cvtColor(inputImage, grayInputImage, CV_RGB2GRAY);
	//the valid filter response is a map of the size of 
	//mapsize =  patchSize - kernelSize + 1  (1 dimension case)
	//(map_rows, map_cols) = (mapsize, mapsize)  (2 dimension case) 
	int kernelSize = m_filterBank[0].cols;
	int patchSize = inputImage.cols;
	int mapSize = patchSize - kernelSize + 1;
	Mat responseMap = Mat(mapSize, mapSize, CV_32F);
	int halfDiff = 0.5*(patchSize - kernelSize);
	int patch_center_x = 0.5*(patchSize - 1);
	int patch_center_y = 0.5*(patchSize - 1);
	int left_roi = patch_center_x - halfDiff;
	int top_roi = patch_center_y - halfDiff;
	Rect roi_filter_res(left_roi, top_roi, mapSize, mapSize);
	//the filter is box 
	int half_length = 0.5*(mapSize - 1);
	int half_response_length = floor(half_length / stepSize);
	//  include (0,0);
	int full_response_length = 2 * half_response_length + 1;
	Mat filter_response_map = Mat(full_response_length * full_response_length, responseNum, CV_32F);
	for (int i = 0; i < m_filterBank.size(); i++)
	{
		//flip the current kernel
		Mat kernel;
		Mat dst;
		//flip the kernel
		flip(m_filterBank[i], kernel, -1);
		//now it is real convolution
		filter2D(grayInputImage, dst, CV_32F, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
		//writeMatToFile(dst, "full_conv_result.txt");
		//get the valid filter response
		responseMap = dst(roi_filter_res);
		//writeMatToFile(responseMap, "valid_responseMat.txt");
		//writeMatToFile(dst, "integrated_image_con.txt");
		int  response_count = 0;
		//select keypoint on the responseMat;
		for (int j = -half_response_length; j <= half_response_length; j++)
		{
			//get center response
			int row_idx = half_length + j * stepSize;
			for (int k = -half_response_length; k <= half_response_length; k++)
			{
				int col_idx = half_length + k *stepSize;
				filter_response_map.at<float>(response_count, i) = responseMap.at<float>(row_idx, col_idx);
				response_count++;
			}
		}
	}
	return filter_response_map;
}