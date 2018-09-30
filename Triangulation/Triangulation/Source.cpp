#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>

// This code takes in a video stream from a single camera, then computes the distance of features found in front of it. The depths information is in the depths Matrix
// 
// To use this code with other camera, please change the kIntrinsics variable to match the instrinsics matrix of the new camera. 
// 
// Feel free to adjust other constant parameters in the code also, they will have the following effects:
// 1. kMaxFeaturesToDetect: changes the maximum number of features to detect, increasing it gives higher accuracy, but lower frame rate
// 2. kMinNMatchesToAccept: changes the threshold of the minimum number of feature matches needed for a pair of frames to be considered as useful, 
// increasing it gives higher accuracy but will increase the number of frames being discarded, which will reduce the frame rate
// 3. kNFrameSkip: the number of frame difference that the code will use to compare, e.g. if kNFrameSkip is set to 3, the present frame will be 
// compared to 3 frames before it, increasing it will give higher accuracy, but can reduce the matches found, as well as requiring more memory
// 4. kPercentMatchesToUse (line 162): the percentage of the matches to use, e.g. if it is set to 0.7, 70% of the matches will be used, and 
// the other 30% will be discarded. Increasing it will reduce the chance of taking in outliers but will also reduce the inliers which will be 
// useful in calculation. Outliers are also filtered out with the RANSAC algorithm when computing the homography matrix
// 
// This code uses OpenCV with no gpu support as it has already achieved more than 20 fps, and OpenCV cuda only works with Nvidia GPUs
// To link this code with OpenCV, please do the following:
// 1. include the directory of <YOUR_OPENCV_FOLDER>\build\include to the project
// 2. use the  <YOUR_OPENCV_FOLDER>\build\x64\vc15\lib\opencv_world343.lib as the library file
// 
// To fuse this code with other modules of the robot, please do the following
// 1. change the 1.0 number on line 316 to the absolute displacement from other sensors
// 2. use the depths Matrix as the output of this module

using namespace cv;

const int kMaxFeaturesToDetect = 2000;
const int kMinNMatchesToAccept = 5;
//we will skip kNFrameSkip frames then compare
const short kNFrameSkip = 10;
//the camera extrinsics is calibrated with MATLAB calibrateCamera module
const cv::Mat1f kIntrinsics = (cv::Mat1f(3, 3) <<
	618.7524, 0, 0,
	0, 625.8908, 0,
	288.7750, 250.0992, 1);

Ptr<ORB> orb = ORB::create(kMaxFeaturesToDetect);
Mat des0;
Mat des1;
Mat img0;
Mat img1;
Mat pts0;
Mat img1_copy;
Mat& pts0_ref = pts0;
Mat pts1;
Mat& pts1_ref = pts1;
Mat homography_mask;
std::vector <Mat> rotations_mat;
std::vector <Mat> translations_mat;
std::vector <Mat> normals_mat;
Mat intrinsics_inv = kIntrinsics.inv();
std::vector<KeyPoint> kp0;
std::vector<KeyPoint>& kp0_ref = kp0;
std::vector<KeyPoint> kp1;
std::vector<KeyPoint>& kp1_ref = kp1;
std::vector<double> depths;
std::vector<double>& depths_ref = depths;
Ptr<BFMatcher> bf = BFMatcher::create(NORM_HAMMING, true);
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
std::vector< DMatch > matches;
std::queue<Mat> img_queue;

Mat FindHeatmapColor(double val) {
	int h = (int)((1 - val) * 180);
	int s = 255;
	int v = (int)(val * 255);

	Mat hsv(1, 1, CV_8UC3, Scalar(h, s, v));
	Mat bgr(1, 1, CV_8UC3, Scalar(0, 0, 0));

	cvtColor(hsv, bgr, CV_HSV2BGR);
	return bgr;
}
double FindDepth(const std::vector<int>& pt0, const std::vector<int>& pt1, const Mat& rel_rot, const Mat& rel_trans, const Mat& inv_intrinsics) {
	//computes the X Y Z coordinate of the point observed as pt0 and pt1 in the image 0 and 1,
	//this is done by solving the intersection between the 2 rays
	//the XYZ coordinate is wrt to the camera 0 which has its projection center at the origin, X Y following the img0,
	//and Z pointing from the projection center perpendicular to the image plane
	//pt0 and pt1 is given as an array of length 2[x, y]
	//the variable notation is referenced with the lecture book"""

	//the projection center of the 0th cam is 0, 0, 0
	std::vector<double> p(3, 0.0);
	std::vector<double> q;

	//q = p+trans
	for (int i = 0; i < 3; i++) {
		q.push_back(p[i] + rel_trans.at<double>(i, 0));
	}

	//pts0 and pts1 that will be used in the calculation to avoid corrupting the old value
	std::vector<int> pt0_calc = pt0;
	std::vector<int> pt1_calc = pt1;

	//convert to homogeneous coordinates
	pt0_calc.push_back(1);
	pt1_calc.push_back(1);

	Mat pt0_mat = (cv::Mat1f(3, 1) << pt0_calc[0], pt0_calc[1], pt0_calc[2]);
	Mat pt1_mat = (cv::Mat1f(3, 1) << pt1_calc[0], pt1_calc[1], pt1_calc[2]);

	Mat r = inv_intrinsics * pt0_mat;

	//Mat s = rel_rot.t() * inv_intrinsics * pt1_mat;
	Mat s = inv_intrinsics.t() * rel_rot;
	s = s.t();
	s = s * pt1_mat;

	//compute the direct solution for the depth, the equation is solved with Matlab and only the direct solution is used for increased performance
	double depth = p[2] / 2 + q[2] / 2 +
		(
		(double)s.at<int>(2, 0) * (
			p[0] * pow((double)r.at<int>(1, 0), 2)*(double)s.at<int>(0, 0)
			+ p[0] * pow((double)r.at<int>(2, 0), 2)*(double)s.at<int>(0, 0)
			+ p[1] * pow((double)r.at<int>(0, 0), 2)*(double)s.at<int>(1, 0)
			+ p[1] * pow((double)r.at<int>(2, 0), 2)*(double)s.at<int>(1, 0)
			+ p[2] * pow((double)r.at<int>(0, 0), 2)*(double)s.at<int>(2, 0)
			+ p[2] * pow((double)r.at<int>(1, 0), 2)*(double)s.at<int>(2, 0)
			- q[0] * pow((double)r.at<int>(1, 0), 2)*(double)s.at<int>(0, 0)
			- q[0] * pow((double)r.at<int>(2, 0), 2)*(double)s.at<int>(0, 0)
			- q[1] * pow((double)r.at<int>(0, 0), 2)*(double)s.at<int>(1, 0)
			- q[1] * pow((double)r.at<int>(2, 0), 2)*(double)s.at<int>(1, 0)
			- q[2] * pow((double)r.at<int>(0, 0), 2)*(double)s.at<int>(2, 0)
			- q[2] * pow((double)r.at<int>(1, 0), 2)*(double)s.at<int>(2, 0)
			- p[0] * (double)r.at<int>(0, 0) * (double)r.at<int>(1, 0) * (double)s.at<int>(1, 0)
			- p[1] * (double)r.at<int>(0, 0) * (double)r.at<int>(1, 0) * (double)s.at<int>(0, 0)
			- p[0] * (double)r.at<int>(0, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(2, 0)
			- p[2] * (double)r.at<int>(0, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(0, 0)
			- p[1] * (double)r.at<int>(1, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(2, 0)
			- p[2] * (double)r.at<int>(1, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(1, 0)
			+ q[0] * (double)r.at<int>(0, 0) * (double)r.at<int>(1, 0) * (double)s.at<int>(1, 0)
			+ q[1] * (double)r.at<int>(0, 0) * (double)r.at<int>(1, 0) * (double)s.at<int>(0, 0)
			+ q[0] * (double)r.at<int>(0, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(2, 0)
			+ q[2] * (double)r.at<int>(0, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(0, 0)
			+ q[1] * (double)r.at<int>(1, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(2, 0)
			+ q[2] * (double)r.at<int>(1, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(1, 0)
			)
			) / (
				2 * (
					pow((double)r.at<int>(0, 0), 2)*pow((double)s.at<int>(1, 0), 2)
					+ pow((double)r.at<int>(0, 0), 2)*pow((double)s.at<int>(2, 0), 2)
					- 2 * (double)r.at<int>(0, 0) * (double)r.at<int>(1, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(1, 0)
					- 2 * (double)r.at<int>(0, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(2, 0)
					+ pow((double)r.at<int>(1, 0), 2)*pow((double)s.at<int>(0, 0), 2)
					+ pow((double)r.at<int>(1, 0), 2)*pow((double)s.at<int>(2, 0), 2)
					- 2 * (double)r.at<int>(1, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(1, 0) * (double)s.at<int>(2, 0)
					+ pow((double)r.at<int>(2, 0), 2)*pow((double)s.at<int>(0, 0), 2)
					+ pow((double)r.at<int>(2, 0), 2)*pow((double)s.at<int>(1, 0), 2)
					)
				) - (
				(double)r.at<int>(2, 0) * (
					p[0] * (double)r.at<int>(0, 0) * pow((double)s.at<int>(1, 0), 2)
					+ p[0] * (double)r.at<int>(0, 0) * pow((double)s.at<int>(2, 0), 2)
					+ p[1] * (double)r.at<int>(1, 0) * pow((double)s.at<int>(0, 0), 2)
					+ p[1] * (double)r.at<int>(1, 0) * pow((double)s.at<int>(2, 0), 2)
					+ p[2] * (double)r.at<int>(2, 0) * pow((double)s.at<int>(0, 0), 2)
					+ p[2] * (double)r.at<int>(2, 0) * pow((double)s.at<int>(1, 0), 2)
					- q[0] * (double)r.at<int>(0, 0) * pow((double)s.at<int>(1, 0), 2)
					- q[0] * (double)r.at<int>(0, 0) * pow((double)s.at<int>(2, 0), 2)
					- q[1] * (double)r.at<int>(1, 0) * pow((double)s.at<int>(0, 0), 2)
					- q[1] * (double)r.at<int>(1, 0) * pow((double)s.at<int>(2, 0), 2)
					- q[2] * (double)r.at<int>(2, 0) * pow((double)s.at<int>(0, 0), 2)
					- q[2] * (double)r.at<int>(2, 0) * pow((double)s.at<int>(1, 0), 2)
					- p[0] * (double)r.at<int>(1, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(1, 0)
					- p[1] * (double)r.at<int>(0, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(1, 0)
					- p[0] * (double)r.at<int>(2, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(2, 0)
					- p[2] * (double)r.at<int>(0, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(2, 0)
					- p[1] * (double)r.at<int>(2, 0) * (double)s.at<int>(1, 0) * (double)s.at<int>(2, 0)
					- p[2] * (double)r.at<int>(1, 0) * (double)s.at<int>(1, 0) * (double)s.at<int>(2, 0)
					+ q[0] * (double)r.at<int>(1, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(1, 0)
					+ q[1] * (double)r.at<int>(0, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(1, 0)
					+ q[0] * (double)r.at<int>(2, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(2, 0)
					+ q[2] * (double)r.at<int>(0, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(2, 0)
					+ q[1] * (double)r.at<int>(2, 0) * (double)s.at<int>(1, 0) * (double)s.at<int>(2, 0)
					+ q[2] * (double)r.at<int>(1, 0) * (double)s.at<int>(1, 0) * (double)s.at<int>(2, 0)
					)
					) / (
						2 * (pow((double)r.at<int>(0, 0), 2)*pow((double)s.at<int>(1, 0), 2)
							+ pow((double)r.at<int>(0, 0), 2)*pow((double)s.at<int>(2, 0), 2)
							- 2 * (double)r.at<int>(0, 0) * (double)r.at<int>(1, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(1, 0)
							- 2 * (double)r.at<int>(0, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(0, 0) * (double)s.at<int>(2, 0)
							+ pow((double)r.at<int>(1, 0), 2)*pow((double)s.at<int>(0, 0), 2)
							+ pow((double)r.at<int>(1, 0), 2)*pow((double)s.at<int>(2, 0), 2)
							- 2 * (double)r.at<int>(1, 0) * (double)r.at<int>(2, 0) * (double)s.at<int>(1, 0) * (double)s.at<int>(2, 0)
							+ pow((double)r.at<int>(2, 0), 2)*pow((double)s.at<int>(0, 0), 2)
							+ pow((double)r.at<int>(2, 0), 2)*pow((double)s.at<int>(1, 0), 2)
							)
						);

	return depth;
}

std::vector<double> FindDepthMultipleImage(const Mat& pts0, const Mat& pts1, const Mat& rel_rot, const Mat& rel_trans, const Mat intrinsics) {
	//takes an array of 2xN with [x,y], then find the depth of all of the pixels
	std::vector<double> depths;


	//compute the depth for each pair of corresponding point
	for (int i = 0; i < pts0.size().height; i++) {
		//create vector
		std::vector<int> pt0 = { pts0.at<int>(i, 0),pts0.at<int>(i, 1) };
		std::vector<int> pt1 = { pts1.at<int>(i, 0),pts1.at<int>(i, 1) };
		depths.push_back(FindDepth(pt0, pt1, rel_rot, rel_trans, intrinsics));
	}

	return depths;
}


std::vector<double> FindDepthsFromOrientationN(const Mat pts0, const Mat pts1, int img_width, int img_height,
	const Mat& rel_rot_arr, const Mat& rel_trans_arr, const Mat& inv_intrinsics, int n) {
	//find the depth of the image using the nth orientation
	//this is because the mathematical equation of rotational and translational matrix will give out 4 possible solutions with only 1 being physically correct
	Mat pts0_normalized = pts0;
	Mat pts1_normalized = pts1;

	depths = FindDepthMultipleImage(pts0_normalized, pts1_normalized, rel_rot_arr, rel_trans_arr, inv_intrinsics);

	return depths;
}

double FindMagnitude(const Mat& mat) {
	double magnitude = 0;
	for (int i = 0; i < mat.size().height; i++) {
		magnitude += pow((double)mat.at<int>(i, 0), 2);
	}
	magnitude = sqrt(magnitude);
	return magnitude;
}

void Normalize1DMat(Mat& mat, double magnitude = 1) {
	//normalize a vector of type Mat to make its magnitude = the provided magnitude (magnitude = 1 will result in finding the unit vector)
	double old_magnitude = FindMagnitude(mat);

	for (int i = 0; i < mat.size().height; i++) {
		mat.at<int>(i, 0) = (double)mat.at<int>(i, 0) * magnitude / old_magnitude;
	}
}

void FindDepthsFromOrientation(const Mat pts0, const Mat pts1, int img_width, int img_height, const std::vector<Mat>& rel_rot_arr,
	const std::vector<Mat>& rel_trans_arr, const Mat& inv_intrinsics, std::vector<double>& depths, double displacement) {
	//find the depths given 4 possible orientations of the camera by trying each of the possible solution
	int final_n = -1;
	std::vector<double> results_n;

	//try each possible solutions
	for (int n = 0; n < 4; n++) {
		//normalized the translation
		Mat normalized_translation_arr = rel_trans_arr[n].clone();
		Mat& normalized_translation_arr_ref = normalized_translation_arr;
		Normalize1DMat(normalized_translation_arr_ref, displacement);

		//depths = FindDepthsFromOrientationN(pts0, pts1, img_width, img_height, rel_rot_arr[n], rel_trans_arr[n], inv_intrinsics, n);
		depths = FindDepthsFromOrientationN(pts0, pts1, img_width, img_height, rel_rot_arr[n], normalized_translation_arr, inv_intrinsics, n);

		//check if the points are all in front of the camera
		if (!std::any_of(depths.begin(), depths.end(), [](int i) {return i < 0; })) {
			//all point is in front of the camera
			final_n = n;
			break;
		}
		else {
			//get the number of points that are in the back of the camera
			int count = (int)(std::count_if(depths.begin(), depths.end(), [](int i) {return i < 0; }));
			results_n.push_back(count);
			if (n == 3) {
				//use final n = 3 as the answer will be rejected by the try/catch statement in the main() anyway
				final_n = 3;
				//depths = FindDepthsFromOrientationN(pts0, pts1, img_width, img_height, rel_rot_arr[n], rel_trans_arr[n], inv_intrinsics, final_n);
				depths = FindDepthsFromOrientationN(pts0, pts1, img_width, img_height, rel_rot_arr[n], normalized_translation_arr, inv_intrinsics, final_n);
			}
		}
	}
}

void GetCorrespondingPointsFromMatches(const std::vector<DMatch>& matches, const std::vector<KeyPoint>& kp0, const std::vector<KeyPoint>& kp1, Mat& pts0, Mat& pts1) {
	//filter good pair of corresponding points and load them into pts0 and pts1

	//filter kPercentMatchesToUse percent from all of the corresponding matches
	const float kPercentMatchesToUse = 0.7f;

	int n_match = int(kPercentMatchesToUse*matches.size());
	//create vector of distance
	std::vector<float> distance_vec;
	for (int i = 0; i < matches.size(); i++) {
		distance_vec.push_back(matches[i].distance);
	}

	//sort the distance vector (this will sort it smallest first)
	std::sort(std::begin(distance_vec), std::end(distance_vec));
	float thresh_distance = 99999;
	if (matches.size() > 10) {
		thresh_distance = distance_vec[(int)((1 - kPercentMatchesToUse)*distance_vec.size())];
	}

	for (int i = 0; i < n_match; i++) {
		DMatch m = matches[i];

		//filter the matches to see if it is in the first kPercentMatchesToUse percent of the corresponding matches
		if (m.distance > thresh_distance) {
			continue;
		}

		//required format for the find hommography
		pts1.push_back(kp1[m.trainIdx].pt);
		pts0.push_back(kp0[m.queryIdx].pt);
	}

}

int main(int, char**)
{
	// open the default camera
	VideoCapture cap(0);

	// check if we succeeded in opening the camera
	if (!cap.isOpened())
		return -1;

	int e1 = 0;
	int e2 = 0;
	double time = 0;

	//create windows to show images for visualization purposes
	namedWindow("img0", 1);
	namedWindow("img1", 1);
	namedWindow("img1_color", 1);

	//capture initial frames into the queue to compare with newer images
	for (int i = 0; i < kNFrameSkip; i++) {
		Mat initial_frame;
		cap >> initial_frame;
		cvtColor(initial_frame, initial_frame, COLOR_BGR2GRAY);
		img_queue.push(initial_frame);
	}

	for (;;)
	{
		//start the timer for performance measurement
		e1 = (int)(getTickCount());

		//get the past image from the queue
		Mat img0 = img_queue.front();
		Mat img1_color;
		img_queue.pop();

		//get a new frame from camera
		cap >> img1;
		cvtColor(img1, img1, COLOR_BGR2GRAY);

		//prepare the image for visualization purposes later in the code
		img1_copy = img1.clone();
		cvtColor(img1_copy, img1_color, COLOR_GRAY2BGR);

		//detect the features in both the current image and the past image
		orb->detectAndCompute(img0, noArray(), kp0_ref, des0);
		orb->detectAndCompute(img1, noArray(), kp1_ref, des1);

		//try to match the features, skip the frame if any error happens
		try {
			bf->match(des0, des1, matches);
		}
		catch (Exception e) {
			std::cout << "can't detect any features" << std::endl;
			imshow("img0", img0);
			imshow("img1", img1);
			imshow("img1_color", img1_color);
			img_queue.push(img1);
			img1_color.release();
			kp0_ref.clear();
			kp1_ref.clear();
			pts0.release();
			pts1.release();
			continue;
		}

		//check if the number of matches is too little, if yes, discard the image
		if (matches.size() < kMinNMatchesToAccept) {
			std::cout << "too little number of matches" << std::endl;
			imshow("img0", img0);
			imshow("img1", img1);
			imshow("img1_color", img1_color);
			img_queue.push(img1);
			img1_color.release();
			kp0_ref.clear();
			kp1_ref.clear();
			pts0.release();
			pts1.release();
			continue;
		}

		//get the matrix of pts0 and pts1 of corresponding points of feature matches
		GetCorrespondingPointsFromMatches(matches, kp0_ref, kp1_ref, pts0_ref, pts1_ref);

		//find the homography matrix that links between the two images, then decompose it into rotation and translation matrices
		//then use the rotation and translation matrices to get the depth of each points
		try {
			Mat homography_mat = findHomography(pts0, pts1, homography_mask);
			int solutions = decomposeHomographyMat(homography_mat, kIntrinsics, rotations_mat, translations_mat, normals_mat);
			FindDepthsFromOrientation(
				pts0,
				pts1,
				img0.size().width,
				img0.size().height,
				rotations_mat,
				translations_mat,
				intrinsics_inv,
				depths,
				1.0);
		}
		catch (Exception e) {
			std::cout << "not enough features to compute, passing" << std::endl;
			imshow("img0", img0);
			imshow("img1", img1);
			imshow("img1_color", img1_color);
			img_queue.push(img1);
			img1_color.release();
			kp0_ref.clear();
			kp1_ref.clear();
			pts0.release();
			pts1.release();
			continue;
		}

		//scale the depths into the range of [0,1] for easy visualization
		double min_depth = *std::min_element(depths.begin(), depths.end());
		double max_depth = *std::max_element(depths.begin(), depths.end());
		double range = max_depth - min_depth;
		for (int i = 0; i < depths.size(); i++) {
			depths[i] = (depths[i] - min_depth) / range;
		}

		//if the depths are NaN, discard the image
		if (isnan(max_depth)) {
			std::cout << "depths are NaNs" << std::endl;
			imshow("img0", img0);
			imshow("img1", img1);
			imshow("img1_color", img1_color);
			img_queue.push(img1);
			img1_color.release();
			kp0_ref.clear();
			kp1_ref.clear();
			pts0.release();
			pts1.release();
			continue;
		}

		//display the depths as circles where red means close and black means far, using the rainbow scale
		for (int i = 0; i < depths.size(); i++) {
			double proximity = 1 - depths[i];
			int x = (int)pts1.at<Point2f>(i, 0).x;
			int y = (int)pts1.at<Point2f>(i, 0).y;

			Mat bgr = FindHeatmapColor(proximity);
			int b = bgr.at<Vec3b>(0, 0)[0];
			int g = bgr.at<Vec3b>(0, 0)[1];
			int r = bgr.at<Vec3b>(0, 0)[2];

			circle(img1_color, Point(x, y), 5, Scalar(b, g, r), 2);
		}

		imshow("img0", img0);
		imshow("img1", img1);
		imshow("img1_color", img1_color);

		img1_color.release();
		kp0_ref.clear();
		kp1_ref.clear();
		pts0.release();
		pts1.release();

		img_queue.push(img1);
		e2 = (int)(getTickCount());
		time = (e2 - e1) / getTickFrequency();
		std::cout << "time = " << time << std::endl;

		//check to see if esc is pressed, if esc is pressed, terminate the program
		if (waitKey(30) >= 0) break;
	}

	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}