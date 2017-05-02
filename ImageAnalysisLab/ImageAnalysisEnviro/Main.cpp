#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <opencv2\imgproc.hpp>
#include <opencv\cv.h>
#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <array>
#include <cmath>

using namespace std;
using namespace cv;

// Function Prototypes
Mat blurImage_Image_Analysis(Mat inputImg);
Mat convertRGB2HSV_Image_Analysis(Mat sourceImg);
Mat erodeImage_Image_Analysis(int HSV_ERODE, Mat sourceImg);
Mat dilateImage_Image_Analysis(int HSV_DILATE, Mat sourceImg);
void snagKeyboardEvents(int &keyboardChoice);
void on_trackbar(int, void*);
void setUpTrackerBars(void);
void thresh_callback(int, void*);


// This Project is intended to be used to do soft coded image analysis. Desired result is find values and properties of image 
// to quickly implement into main sections of drone code
// TrackerBar HSV Integer Values

// These are the Global HSV Values used for Quick Image Analysis
int H_LOW = 0;
int H_HIGH = 255;
int S_LOW = 0;
int S_HIGH = 255;
int V_LOW = 0;
int V_HIGH = 255;
int HSV_DILATE = 1;
int HSV_ERODE = 1;
int HSV_BLUR = 1;
int key;
int ESCAPE_KEY = 27;

// Some Global Boolean Values
bool toggleHSV = false;
bool toggleXmasTrees = false;
bool countObjects = false;
bool allowEnterance = false;

// Counter and Image Matrices
int _counter = 0;
Mat BGRImg,HSVImg,ThresholdImg;

int main()
{   
	// Read in Video Feed -- This Option Shows HSV Manipulation for real time processing 
	VideoCapture cap(0);
	if (!cap.isOpened())return -1;
	
	// You could use this option to simply just read in one image instead of a live video feed
	//BGRImg = imread("Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\UnprocessedImages\\Picture2.jpg");
	
	// Call the tracker bar function to initialize trackerbars
	setUpTrackerBars();
	
	// Create a named window to display the image on screen with function imshow("Window Name");
	namedWindow("ImageWindow", CV_WINDOW_NORMAL); //this is to show image...we want to output image into file
	cout << "Hit Escape Key To Terminate Program" << endl;

	while ((key = waitKey(50)) != ESCAPE_KEY)
	{
		// Comment this out below if you are reading in a still image
		cap >> BGRImg;
		//img = imread("Z:\\Desktop\\Honors_Thesis\\OpenCV_Chapter_Notes_And_Scripts\\UnprocessedImages\\FlowerBlossoms\\DJI_00" + to_string(14 + universal_counter) + ".jpg");
		// Snag the keyboard event to see if any switches need to be made from HSV to BGR
		snagKeyboardEvents(key);

		// We need to do a conversion between the BGR Full color image and the HSV image. Then take the
		// HSV image and run through the inRange function to filter out unwanted values
		ThresholdImg = convertRGB2HSV_Image_Analysis(BGRImg); // Returns the threshold image using the trackervalues we updated
		ThresholdImg = blurImage_Image_Analysis(ThresholdImg);
		// We need to check if the user wishes to see the HSV Image or the Regular Image
		if (toggleHSV) imshow("ImageWindow", ThresholdImg);
		else imshow("ImageWindow", BGRImg);

		// Count Objects -- Put Whatever Code Here
		if (countObjects)
		{
			countObjects = !countObjects;
			// Put Code To Run Here when you push the key that corresponds to the countObjects Boolean. Count Objects can be changed
 		}

	}
destroyAllWindows();

}

Mat convertRGB2HSV_Image_Analysis(Mat sourceImg)
{
	// HSV Values for Morphological Operations
	Mat Threshold, HSVImg, Threshold_Temp;
	cvtColor(BGRImg, HSVImg, CV_BGR2HSV);
	inRange(HSVImg, Scalar(H_LOW,S_LOW,V_LOW), Scalar(H_HIGH,S_HIGH,V_HIGH), Threshold_Temp); // turn hsv into threshold
	Threshold_Temp = dilateImage_Image_Analysis(HSV_DILATE, Threshold_Temp); // dilate
	Threshold = erodeImage_Image_Analysis(HSV_ERODE, Threshold_Temp);   // erode
	return Threshold;
}

Mat dilateImage_Image_Analysis(int HSV_DILATE, Mat sourceImg)
{
	// We are assuming that the dilation effect will be a rectangular dilation
	Mat outputImage;
	// Create the structuring element for dilation
	Mat element = getStructuringElement(0, Size(2 * HSV_DILATE + 1, 2 * HSV_DILATE + 1),
		Point(HSV_DILATE, HSV_DILATE));
	dilate(sourceImg, outputImage, element);
	return outputImage;
}

Mat erodeImage_Image_Analysis(int HSV_ERODE, Mat sourceImg)
{
	Mat outputImg;
	// Create the structuring element for erosion
	Mat element = getStructuringElement(1, Size(2 * HSV_ERODE + 1, 2 * HSV_ERODE + 1),
		Point(HSV_ERODE, HSV_ERODE));
	// Erode the image using the structuring element
	erode(sourceImg, outputImg, element);
	return outputImg;
}

Mat blurImage_Image_Analysis(Mat inputImg)
{
	// Simply blur the image to different values, which will be used to determine if the blurred filter Image will 
	// lead to better image recognition
	Mat blurredImage;
	medianBlur(inputImg, blurredImage, HSV_BLUR);
	return blurredImage;
}

void setUpTrackerBars(void)
{
	// Need to call the window in the same function as the trackbars themselves
	namedWindow("TrackerBarWindow", CV_WINDOW_NORMAL);
	
	// HSV Values
	createTrackbar("H_MIN", "TrackerBarWindow", &H_LOW, H_HIGH, on_trackbar);
	createTrackbar("H_MAX", "TrackerBarWindow", &H_HIGH, H_HIGH, on_trackbar);
	createTrackbar("S_MIN", "TrackerBarWindow", &S_LOW, S_HIGH, on_trackbar);
	createTrackbar("S_MAX", "TrackerBarWindow", &S_HIGH, S_HIGH, on_trackbar);
	createTrackbar("V_MIN", "TrackerBarWindow", &V_LOW, V_HIGH, on_trackbar);
	createTrackbar("V_MAX", "TrackerBarWindow", &V_HIGH, V_HIGH, on_trackbar);
	
	//Dilate, Erode, and threshhold
	createTrackbar("Dilate", "TrackerBarWindow", &HSV_DILATE, 50, on_trackbar);
	createTrackbar("Erode", "TrackerBarWindow", &HSV_ERODE, 50, on_trackbar);
	createTrackbar("Blur", "TrackerBarWindow", &HSV_BLUR, 50, on_trackbar);
}

void on_trackbar(int, void*)
{
	/*
	When you drag the trackerbar across the screen, the referenced value you enter is used to gage.
	Apparently it somehow handles the mouse event on its own without putting in the while loop
	*/
	// We want to make sure these values are never zero
	if (HSV_DILATE == 0)
		HSV_DILATE = 1;
	if (HSV_ERODE == 0)
		HSV_ERODE = 1;
	if (HSV_BLUR % 2 == 0)
		HSV_BLUR += 1;
}

void snagKeyboardEvents(int &keyboardChoice)
{
	// hitting the keyboard key t will show the HSV image and not the regular image
	if (keyboardChoice == 't')
	{
		toggleHSV = !toggleHSV;
	}
	// by toggling the keyboard key f we will run a full analysis on the filefolder of images
	if (keyboardChoice == 'f')
	{
		toggleXmasTrees = !toggleXmasTrees;
	}
	// by pressing the letter c, count the objects
	if (keyboardChoice == 'c')
	{
		countObjects = !countObjects;
	}
}

void thresh_callback(int, void*)
{
	// Finding Threshold with Gray Image and Canny function
	Mat src; Mat src_gray;
	int thresh = 100;
	int max_thresh = 255;
	RNG rng(12345);

	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	/// Show in a window
	namedWindow("Contours", CV_WINDOW_NORMAL);
	imshow("Contours", drawing);
}



