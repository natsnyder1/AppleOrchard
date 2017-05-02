
#include <iostream>
#include <fstream>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv\cv.h>
#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <array>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

// The objective of this code is to identify christmas trees in a stochastic environment. The objectives are three fold.
// 1.) Count as many trees accurately as possible. 
// 2.) Allow users to interact with the software so they can count trees in anindividual patch of an image.
// 3.) Document the geographical infomration in an organized method to allow file writing with it.


// The tree Image class will be designed to read a tree image in, find the amount of trees, their x,y location within the image
// and the final count. 

// Function Prototypes
void onMouse_SelectCorners(int event, int x, int y, int, void* userInput);
void ChristmasTreeGUI();
void followBall();
uint8_t zoomInZoomOut();

// Class Describing the contents of high altitude christmas tree images 
class christmasTreeImageAnalyzer
{
	private:
		// Nested Inner Classes
		class innerHSVObject
		{
		public:	
			// Member variables
			int H_LOW;
			int H_HIGH;
			int S_LOW;
			int S_HIGH;
			int V_LOW;
			int V_HIGH;
			int HSV_DILATE;
			int HSV_ERODE;
			int HSV_BLUR;
			Scalar LOW;
			Scalar HIGH;


			innerHSVObject()
			{
				set();
			}

			virtual void set()
			{
				// Virtual override function
			}

		};
		class XmasHSV: public innerHSVObject
		{
			private:

			public:
				XmasHSV()
				{
					set();
				}

				void set()
				{
					// Virtual override function
					H_LOW = 28;
					H_HIGH = 40;
					S_LOW = 63;
					S_HIGH = 176;
					V_LOW = 150;
					V_HIGH = 194;
					HSV_DILATE = 4;
					HSV_ERODE = 7;
					HSV_BLUR = 1;
					LOW = Scalar(H_LOW, S_HIGH, V_LOW);
					HIGH = Scalar(H_HIGH, S_HIGH, V_HIGH);

				}

		};
		class ImgStats
		{
		private:
			vector<double> dataSet;

			void getData(vector<pair<double, Point2i> >* data)
			{
				for (vector<pair<double, Point2i> >::iterator itT = data->begin(); itT != data->end(); ++itT)
				{
					dataSet.push_back(itT->first);
				}
			}

		public:
			double mean;
			double median;
			pair<double,int> mode;
			double std_dev;
			double variance;
			double interquartile_Mean;
			double interquartile_Std_Dev;
			double interquartile_Variance;
			double interquartile_Mode;

			ImgStats(vector<pair<double, Point2i> >* _dataSet)
			{
				getData(_dataSet);
				findMean();
				findMedian();
				findMode();
				findStdDev();
				findIQMean();
				findIQStdDev();
				findIQMode();
			}

			void findMean()
			{
				int sum = 0;
				int average = 0;
				for (vector<double>::iterator it = dataSet.begin(); it != dataSet.end(); ++it)
				{
					sum += *it;
				}
				mean = sum / dataSet.size();
			}

			void findStdDev()
			{
				double absDiff = 0;
				double temp = 0;
				for (vector<double>::iterator it = dataSet.begin(); it != dataSet.end(); ++it)
				{
					absDiff = abs(*it - mean);
					temp += pow(absDiff, 2);
				}

				std_dev = sqrt( ( temp / (dataSet.size() - 1) ) );
				variance = pow(std_dev, 2);
			}

			void findMedian()
			{
				// 1. Sort Array ascending
				std::sort(dataSet.begin(), dataSet.end(), sort_double());

				// 2. count middle
				int middleNumber;
				if (dataSet.size() % 2 == 0) // Even
				{
					middleNumber = dataSet.size() / 2;
					median = (dataSet[middleNumber - 1] + dataSet[middleNumber]) / 2;
				}
				else
				{
					middleNumber = (dataSet.size() + 1) / 2;
					median = dataSet[middleNumber];
				}
			}

			void findMode()
			{
				vector<double> uniqueNumbers;
				vector<int> numberFrequency;
				vector< pair< double, int> > modeData;
				// 1. Data must be sorted
				vector<double> dataCopy = dataSet;
				sort(dataCopy.begin(), dataCopy.end(), sort_double());

				// 2. Go through each number
				for (vector<double>::iterator itD = dataCopy.begin(); itD != dataCopy.end(); ++itD)
				{

					// Compare Numbers for frequency and uniqueness
					bool found = false; // Found is set to true when it finds the number itD points to in unique numbers data set
					for (vector<double>::iterator itU = uniqueNumbers.begin(); itU != uniqueNumbers.end(); ++itU)
					{
						// If the contents of the dataSet matches an entry within unique numbers, then the number has already been counted
						if (*itD == *itU)
						{
							found = true;
						}
					}
					// If the value found is false, then it means itD is a unique number and should be added to the itU dataset 
					if (found != true)
					{
						uniqueNumbers.push_back(*itD);
					}

				}

				for (vector<double>::iterator itU = uniqueNumbers.begin(); itU != uniqueNumbers.end(); ++itU)
				{
					int count = 0;
					for (vector<double>::iterator itD = uniqueNumbers.begin(); itD != uniqueNumbers.end(); ++itD)
					{
						if (*itU == *itD)
						{
							count++;
						}
					}
					modeData.push_back(pair<double, int>(*itU, count));
				}
				sort(modeData.begin(), modeData.end(), sortMode());
				mode = modeData[0];
			}

			void findIQMean()
			{
				int sum = 0;
				int average = 0;
				int count = 0;
				for (vector<double>::iterator it = dataSet.begin() + dataSet.size() / 4; it != dataSet.end() - dataSet.size() / 4; ++it)
				{
					sum += *it;
					count++;
				}
				interquartile_Mean = sum / count;
			}

			void findIQStdDev()
			{
				double absDiff = 0;
				double temp = 0;
				for (vector<double>::iterator it = dataSet.begin() + dataSet.size()/4; it != dataSet.end() - dataSet.size()/4; ++it)
				{
					absDiff = abs(*it - interquartile_Mean);
					temp += pow(absDiff, 2);
				}

				interquartile_Std_Dev = sqrt((temp / (dataSet.size() - 1)));
				interquartile_Variance = pow(std_dev, 2);
			}

			void findIQMode()
			{
				vector<double> uniqueNumbers;
				vector<int> numberFrequency;
				vector< pair< double, int> > modeData;
				// 1. Data must be sorted
				vector<double> dataCopy = dataSet;
				std::sort(dataCopy.begin(), dataCopy.end(), sort_double());

				// 2. Go through each number

				for (vector<double>::iterator itD = dataCopy.begin() + dataCopy.size()/4; itD != dataCopy.end() - dataCopy.size()/4; ++itD)
				{

					// Compare Numbers for frequency and uniqueness
					bool found = false; // Found is set to true when it finds the number itD points to in unique numbers data set
					for (vector<double>::iterator itU = uniqueNumbers.begin(); itU != uniqueNumbers.end(); ++itU)
					{
						// If the contents of the dataSet matches an entry within unique numbers, then the number has already been counted
						if (*itD == *itU)
						{
							found = true;
						}
					}
					// If the value found is false, then it means itD is a unique number and should be added to the itU dataset 
					if (found != true)
					{
						uniqueNumbers.push_back(*itD);
					}

				}

				for (vector<double>::iterator itU = uniqueNumbers.begin(); itU != uniqueNumbers.end(); ++itU)
				{
					int count = 0;
					for (vector<double>::iterator itD = dataCopy.begin() + dataCopy.size()/4; itD != dataCopy.end() - dataCopy.size()/4; ++itD)
					{
						if (*itU == *itD)
						{
							count++;
						}
					}
					modeData.push_back(pair<double, int>(*itU, count));
				}
				std::sort(modeData.begin(), modeData.end(), sortMode());
				mode = modeData[0];
			}
		};

		// Sorting Predicate: Sorts a vector container of pairs in decending order 
		struct sort_pred_double_descending {
			bool operator()(const pair<double, Point2i> &left, const pair<double, Point2i> &right)
			{
				return left.first > right.first; // first value is the largest
			}
		};

		// Sorting Predicate: Sorts a vector container of pair<> in ascending order 
		struct sort_pred_double_ascending {
			bool operator()(const pair<double, Point2i> &left, const pair<double, Point2i> &right)
			{
				return left.first < right.first; // first value is the smallest
			}
		};
		struct sort_double
		{
			bool operator()(const double& left, const double& right)
			{
				return left < right;
			}
		};
		struct sortMode
		{
			bool operator()(const pair<double, int>& left, const pair<double, int> & right)
			{
				return right.second > left.second;
			}
		};

		// Variables
		Mat img; // Read in Image
		int imHeight; // Image Height
		int imWidth; // Image Width
		ImgStats* myStats = NULL;

		// Private function that acts upon img. converts image to threshold
		Mat& findThreshold()
		{
			Mat hsvImg;
			Mat threshold;
			cvtColor(img, hsvImg, CV_BGR2HSV);
		}

		// Simple dilation on an image using a dilation value provided
		Mat dilateImage(int ORCHARD_DILATE, Mat sourceImg)
		{
			// We are assuming that the dilation effect will be a rectangular dilation
			Mat outputImage;
			// Create the structuring element for dilation
			Mat element = getStructuringElement(0, Size(2 * ORCHARD_DILATE + 1, 2 * ORCHARD_DILATE + 1),
				Point(ORCHARD_DILATE, ORCHARD_DILATE));
			dilate(sourceImg, outputImage, element);
			return outputImage;
		}

		// Simple erosion on an image using a erosion value provided
		Mat erodeImage(int ORCHARD_ERODE, Mat sourceImg)
		{
			Mat outputImg;
			// Create the structuring element for erosion
			Mat element = getStructuringElement(1, Size(2 * ORCHARD_ERODE + 1, 2 * ORCHARD_ERODE + 1),
				Point(ORCHARD_ERODE, ORCHARD_ERODE));
			// Erode the image using the structuring element
			erode(sourceImg, outputImg, element);
			return outputImg;
		}

		// since all HSV_Value classes inheret from the innerBaseClass, this is the best way to make sure all operations can convert using the method
		Mat convertRGB2HSV(Mat BGRImg, innerHSVObject* ORCHARD)
		{
			// HSV Values for Morphological Operations
			ORCHARD->set();
			Mat Threshold, HSVImg, Threshold_Temp;
			cvtColor(BGRImg, HSVImg, CV_BGR2HSV);
			inRange(HSVImg, Scalar(ORCHARD->H_LOW, ORCHARD->S_LOW, ORCHARD->V_LOW), Scalar(ORCHARD->H_HIGH, ORCHARD->S_HIGH, ORCHARD->V_HIGH), Threshold_Temp);
			Threshold_Temp = dilateImage(ORCHARD->HSV_DILATE, Threshold_Temp); // dilate
			Threshold = erodeImage(ORCHARD->HSV_ERODE, Threshold_Temp);   // erode
			return Threshold;
		}


	public:
		// Public treeImage constructure
		christmasTreeImageAnalyzer(Mat& inputImg)
		{
			img = inputImg;
			imHeight = inputImg.rows;
			imWidth = inputImg.cols;
		}
		
		cv::Mat countTrees()
		{
			// 1. Convert the RGB Image into an HSV and subsequent threshold Image
			Mat bgr;
			Mat threshold, temp;
			XmasHSV HSV_Values;
			img.copyTo(bgr);
			threshold = convertRGB2HSV(bgr, &HSV_Values);

			// 2. With the new threshold image, find number of raw objects
			threshold.copyTo(temp);
			vector<pair<double, Point2i> > objectContainer;
			vector< vector<Point> > contours; // Contours of objects
			vector<Vec4i> hierarchy; // Hierarchy of Object
			double objectArea; // Area of Object
			Point2i xyCords; // Object Center Coordinates
			
			// find contours
			findContours(temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

			// loop through contours to find number of objects, sizes, and locations
			for (vector<vector<Point2i> >::iterator itT = contours.begin(); itT != contours.end(); ++itT)
			{
				// Find moment of objects
				Moments objMoment = moments((Mat)*itT);

				// Prevent little pieces of noisy data form getting into img. Img size 4000x3000
				if (objMoment.m00 > 50)
				{
					// Find Area and center
					objectArea = objMoment.m00; // finds object area
					xyCords = Point2i(objMoment.m10 / objectArea, objMoment.m01 / objectArea); // finds centerpoint
					// Store Object inside container
					objectContainer.push_back(pair<double, Point2i>(objectArea, xyCords));
				}

			}

			// Sort Results by Area - Ascending
			std::sort(objectContainer.begin(), objectContainer.end(), sort_pred_double_ascending());
			
			// Find stats on the christmas tree sizes
			myStats = new ImgStats(&objectContainer);

			// Find Coordinates of the tree with the median size
			Point2i treeCords;
			for (vector<pair<double, Point2i> >::iterator itO = objectContainer.begin(); itO != objectContainer.end(); ++itO)
			{
				// find the coordinates for the median tree size in the data
				if (itO->first == myStats->median)
				{
					treeCords = itO->second; // store coordinates
				}
			}
		
			// Set the HSV Scalar Value for christmas trees you find
			Mat hsvImg;
			cvtColor(bgr, hsvImg, CV_BGR2HSV);
			Vec3b intensity = hsvImg.at<Vec3b>(treeCords.y, treeCords.x);
			

			// count objects and show image with counted trees
			int treeCount = 0;
			int radius = sqrt(myStats->interquartile_Mean / 3.14);
			for (vector<pair<double, Point2i> >::iterator itO = objectContainer.begin(); itO != objectContainer.end(); ++itO)
			{
				// If the data is within 95% of the confidence interval
				if ( ( itO->first > (myStats->mean - 2*myStats->std_dev) ) && (itO->first < (myStats->mean + 2*myStats->std_dev) ) ) 
				{
					circle(bgr, itO->second, radius, Scalar(0, 0, 255), -1);
					++treeCount;
				}
				
			}
			string text = "Trees Found = " + to_string(treeCount);
			putText(bgr, text, Point(50, 50), FONT_HERSHEY_PLAIN, 4, Scalar(255, 0, 0), 3);
			return bgr;
		}

};

// Global Variables
int key;
Mat ROI;
Mat src;
int clickNum = 0;
int rectangleEventClickCounter = 0;
int cropEventClickCounter = 0;
char currentEvent;
cv::Point2i lastMouseLocation;
vector<Point2i> clickLocations;
Mat rectImg;
// Vector to hold images
vector<Mat> imageArray;
vector<cv::Mat>::iterator currentImageInArrayPointer;
uint8_t imageArrayIndexer = 0;

// Booleans
bool rectangleEventFinished = false;
bool croppedEventFinished = false;

//  Constants
const uint8_t ESCAPE_KEY = 27;
const uint8_t ZOOM_IN = 61;
const uint8_t ZOOM_OUT = 45;
const uint8_t UNDO = 44;
const uint8_t REDO = 46;


// Sorting Predicates
struct sort_point2i_xleft
{
	bool operator()(const Point2i& left, const Point2i& right)
	{
		return left.x < right.x;
	}
};
struct sort_point2i_xright
{
	bool operator()(const Point2i& left, const Point2i& right)
	{
		return left.x > right.x;
	}
};
struct sort_point2i_ytop
{
	bool operator()(const Point2i& top, const Point2i& bottom)
	{
		return top.y < bottom.y;
	}
};
struct sort_point2i_ybottom
{
	bool operator()(const Point2i& top, const Point2i& bottom)
	{
		return top.y > bottom.y;
	}
};


int main(void)
{		 
	ChristmasTreeGUI();
	waitKey(0);
}

// Mouse Callback Event #1
void onMouse_SelectCorners(int event, int x, int y, int, void* userInput)
{
	if (event != EVENT_LBUTTONDOWN) return;

	// Get the pointer input image
	Mat* img = (Mat*)userInput;
	// Draw circle
	cv::circle(*img, Point(x, y), 30, Scalar(0, 0, 255), 3);

	// Increment userClicks, and update the click number on the screen
	clickNum++;
	clickLocations.push_back(Point2i(x, y));

}

void onMouse_DrawingFunctions(int event, int x, int y, int, void* userInput)
{
	// If the user is using the rectangle feature 'r'. All code in this block makes the GUI functionality work with the user
	if (currentEvent == 'r')
	{	// If the mouse moves, then run this code
		if (event == cv::EVENT_MOUSEMOVE)
		{
			lastMouseLocation = cv::Point2i(x, y); // Store the last mouse location
		}
		
		// If the mouse pad was clicked down, run this code
		if (event == cv::EVENT_LBUTTONDOWN)
		{
			std::cout << "Mouse Down" << std::endl;
			clickLocations.push_back(Point2i(x, y));
			rectangleEventClickCounter++;
			rectangleEventFinished = false;
		}
		
		// If the user had just let the mousepad come up, this code will run
		if (event == cv::EVENT_LBUTTONUP)
		{
			rectangleEventClickCounter++;

			if (rectangleEventClickCounter == 2) // If the user made an initial click then it will trigger here
			{
				// Update the rectangle being drawn in the users screen with the latest event data
				std::cout << "Drawing Rectangle" << std::endl;
				Mat temp = *(imageArray.begin() + imageArrayIndexer - 1);
				cv::rectangle(temp, *(clickLocations.end() - 1), cv::Point2i(x, y), cv::Scalar(0, 0, 255), 2, 1); // Draw updated rectangle
				*currentImageInArrayPointer = temp;
				
				rectangleEventClickCounter = 0;
				rectangleEventFinished = true;

				imshow("XmasTree", *currentImageInArrayPointer);
				waitKey(25);
			}
		}

	}

	// If the user is using the crop function 'c'. This will find the region of interest the user is interested in and add the cropped image to the cureent image array pointer
	if (currentEvent == 'c')
	{	
		// If the mouse moves, then run this code
		if (event == cv::EVENT_MOUSEMOVE)
		{
			lastMouseLocation = cv::Point2i(x, y); // Store the last mouse location
		}

		// If the mousepad was clicked down, then run this code
		if (event == cv::EVENT_LBUTTONDOWN)
		{
			std::cout << "Mouse Down" << std::endl;
			clickLocations.push_back(Point2i(x, y));
			cropEventClickCounter++;
			croppedEventFinished = false;
		}

		// If the mousepad was down, as was released up, then this code segment will run
		if (event == cv::EVENT_LBUTTONUP)
		{
			 cropEventClickCounter++;
			 std::cout << "Mouse Up" << std::endl;

			if (cropEventClickCounter == 2) // If the user made an initial click then it will trigger here
			{
				// Crop Image
				std::cout << "Croppping Image" << std::endl;
				// Condition for Cropping
				if ((clickLocations.end() - 1)->x < x && (clickLocations.end() - 1)->y < y)
				{
					Mat tmp = cv::Mat(*currentImageInArrayPointer, cv::Rect(*(clickLocations.end() - 1), Point2i(x, y)));
					*currentImageInArrayPointer = tmp;
					destroyWindow("XmasTree");
					namedWindow("XmasTree");
					imshow("XmasTree", *currentImageInArrayPointer);
					waitKey(25);
				}
				cropEventClickCounter = 0;
				croppedEventFinished = true;
			}
		}
		
	}

}
// Main function call for User Event

// Zoom in and Zoom out:
uint8_t zoomInZoomOut()
{
		/// variables
		Mat src, dst, tmp;
		char* window_name = "Pyramids Demo";
		/// General instructions
		printf("\n Zoom In-Out demo  \n ");
		printf("------------------ \n");
		printf(" * [i] -> Zoom in  \n");
		printf(" * [o] -> Zoom out \n");
		printf(" * [ESC] -> Close program \n \n");

		/// Test image - Make sure it s divisible by 2^{n}
		src = imread("C:\\Users\\natsn\\Desktop\\Research\\ImageAnalysisLab\\Drone Code\\ChristmasTrees\\XmasTreeAerial.jpg");
		if (!src.rows)
		{
			printf(" No data! -- Exiting the program \n");
			return -1;
		}

		tmp = src;
		dst = tmp;

		/// Create window
		namedWindow(window_name, CV_WINDOW_AUTOSIZE);
		imshow(window_name, dst);

		/// Loop
		while (true)
		{
			int c;
			c = waitKey(10);

			if ((char)c == 27)
			{
				break;
			}
			if ((char)c == 'i')
			{
				pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
				printf("** Zoom In: Image x 2 \n");
			}
			else if ((char)c == 'o')
			{
				pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
				printf("** Zoom Out: Image / 2 \n");
			}

			imshow(window_name, dst);
			tmp = dst;
		}
		return 0;
}

// Draw, Navigate, and Count Object within the image
void ChristmasTreeGUI()
{
	int keyPressed;

	// Change this line to the file that needs an image read in
	src = imread("C:\\Users\\natsn\\Desktop\\Research\\ImageAnalysisLab\\Drone Code\\ChristmasTrees\\XmasTreeAerial.jpg");
	if (src.rows == 0)
	{
		cout << "Failed To Open!" << endl;
		exit(1);
	}
	// Store the image inside of the image array
	imageArray.push_back(src); // Store The Image in the image node
	currentImageInArrayPointer = imageArray.begin();  // Update the image pointer to point at the initial data

	// Window for Image
	namedWindow("XmasTree", WINDOW_AUTOSIZE);
	
	/* // Let user zoom in, zoom out, click a region of interest, and drag a circle around tree
	// r: Draw a rectangle using the mouse 
	// c: Crop image to specific dimensions
	// +: Zoom into image 
	// -: Zoom out of image
	// t: Train Machine- allows you to draw a circle over a tree for the computer to learn what the tree looks like
	// <: Undo - Show previous image    (if there is one)
	// >: Redo - Show next image (if there is one)
	*/
	
	cout << " Press Keys r, c, +, -, t, b, f To Trigger Events " << endl;
	cout << " r: Draw a rectangle using the mouse " << endl;
	cout << " c: Crop image to specific dimensions " << endl;
	cout << " +: Zoom into image " << endl;
	cout << " -: Zoom out of image " << endl;
	cout << " r: Count All Trees within the current image" << endl;
	cout << " t: Train Machine- allows you to draw a circle over a tree for the computer to learn what the tree looks like (NOT FINISHED) " << endl;
	cout << " <: Back - Show previous image    (if there is one) " << endl;
	cout << " >: Forward - Show previous image (if there is one) " << endl;
	cout << " Press Escape To Exit From Drawing Function" << endl;
	
	while (1)
	{
		// Find user keyboard event. WaitKey will wait the number of milliseconds you specify and then return the character on the keyboard that was pressed
		keyPressed = waitKey(10);
		// If the user wishes to draw a rectangle, then set up mouse events to allow drawing
		if (keyPressed == 'r')
		{
			cout << "Hello From Draw Rectangle!" << endl;
			currentEvent = 'r';
			// Allow the update of the rectangle on the screen while the mouse is held down
			
			// Initialize the Mouse Callback function - Any time a mouse click, movement, etc occurs, 
			// run the function specified inside of mouseCallBack function for the window you specify
			setMouseCallback("XmasTree", onMouse_DrawingFunctions);

			// Save last image we have to the image array
			imageArray.push_back(*currentImageInArrayPointer); // Save the image
			imageArrayIndexer++;
			currentImageInArrayPointer = imageArray.begin() + imageArrayIndexer;
			imshow("XmasTree", *currentImageInArrayPointer);
			waitKey(25);
			// Escape condition is the user wishes to stop drawing on their image by pressing the escape key
			while ((keyPressed = waitKey(50)) != ESCAPE_KEY && !rectangleEventFinished)
			{
				// Let the mouse events draw the picture for us.
				imshow("XmasTree", *currentImageInArrayPointer);
				waitKey(25);
				
			}
			std::cout << "Rectangle Successfully Drawn" << std::endl;
			std::cout << "Goodbye From Draw Rectangle!" << std::endl;
		}
		// Check To see if the user wishes to crop the image to their specifications
		if (keyPressed == 'c')
		{
			currentEvent = 'c';
			cout << "Hello From Crop Image" << endl;
			setMouseCallback("XmasTree", onMouse_DrawingFunctions);
			// The Array Should add the last image again to the arrayPointer so the crop function will not harm the source 
			imageArray.push_back(*currentImageInArrayPointer);
			imageArrayIndexer++;   
			currentImageInArrayPointer = imageArray.begin() + imageArrayIndexer;
			imshow("XmasTree", *currentImageInArrayPointer);
			keyPressed = waitKey(25);

			while ((keyPressed = waitKey(50)) != ESCAPE_KEY && !croppedEventFinished)
			{
				imshow("XmasTree", *currentImageInArrayPointer);
				waitKey(25);
			}
			std::cout << "Image Cropped Successfully" << std::endl;
			std::cout << "Goodbye From Crop Image!" << std::endl;
		}
		// If the user wants the number of trees to be counted, push f
		if (keyPressed == 'f')
		{
			cout << "Hello From Find Number of Trees! " << endl;
			cout << "Counting...." << endl;

			// The Array Should add the last image again to the arrayPointer so the crop function will not harm the source 
			imageArray.push_back(*currentImageInArrayPointer);
			imageArrayIndexer++;
			currentImageInArrayPointer = imageArray.begin() + imageArrayIndexer;
			
			christmasTreeImageAnalyzer aerialViewOfTrees(*currentImageInArrayPointer); // Initialize a Christmas Tree Analysis Object
			*currentImageInArrayPointer = aerialViewOfTrees.countTrees();
			destroyWindow("XmasTree");
			cout << "Trees Successfully Counted!" << endl;
			namedWindow("XmasTree");
			imshow("XmasTree", *currentImageInArrayPointer);
			waitKey(25);
			cout << "Gooodbye from Count Trees!" << endl;
		}

		// Check if the user wants to zoom into the image
		if (keyPressed == ZOOM_IN)
		{
			currentEvent = ZOOM_IN;
			cout << "Hello From Zoom In!" << std::endl;
			imageArray.push_back(*currentImageInArrayPointer); // Add The next node
			imageArrayIndexer++;
			currentImageInArrayPointer = imageArray.begin() + imageArrayIndexer;
			pyrUp(imageArray[imageArrayIndexer - 1], *currentImageInArrayPointer, Size(currentImageInArrayPointer->cols * 2, currentImageInArrayPointer->rows * 2));
			printf("** Zoom In: Image x 2 \n");

			imshow("XmasTree", *currentImageInArrayPointer);
			waitKey(25);

			cout << "Image Has been zoomed in on" << std::endl;
			cout << "Goodbye from Zoom In!" << std::endl;
		}

		if (keyPressed == ZOOM_OUT)
		{
			currentEvent = ZOOM_OUT;
			cout << "Hello From Zoom Out!" << std::endl;
			imageArray.push_back(*currentImageInArrayPointer); // Add The next node
			imageArrayIndexer++;
			currentImageInArrayPointer = imageArray.begin() + imageArrayIndexer;
			pyrDown(imageArray[imageArrayIndexer -1], *currentImageInArrayPointer, Size(currentImageInArrayPointer->cols / 2, currentImageInArrayPointer->rows / 2));
			printf("** Zoom Out: Image / 2 \n");

			imshow("XmasTree", *currentImageInArrayPointer);
			waitKey(25);

			cout << "Image Has been zoomed out on" << std::endl;
			cout << "Goodbye from Zoom Out!" << std::endl;
		}
		if (keyPressed == 't')
		{
			// Machine Training Code -- Update here
			currentEvent == 't';
			cout << "Hello From Zoom Out!" << std::endl;
			imageArray.push_back(*currentImageInArrayPointer); // Add The next node
			imageArrayIndexer++;
			currentImageInArrayPointer = imageArray.begin() + imageArrayIndexer;
			
		}
		// If User hits the '<', show next image in chain (Undo last step)
		if (keyPressed == UNDO)
		{
			// This means the user wants to go back to the last image they were seeing
			if (imageArrayIndexer > 0)
			{
				imageArrayIndexer--;
				currentImageInArrayPointer = imageArray.begin() + imageArrayIndexer;
				destroyWindow("XmasTree");
				namedWindow("XmasTree");
				imshow("XmasTree", *currentImageInArrayPointer);
				waitKey(25);
			}

		}
		// If User hits the '>', show next image in chain (Redo your undo)
		if (keyPressed == REDO)
		{
			// This means the user wants to go back to the last image they were seeing
			if ((imageArray.begin() + imageArrayIndexer + 1) != imageArray.end())
			{
				imageArrayIndexer++;
				currentImageInArrayPointer = imageArray.begin() + imageArrayIndexer;
				destroyWindow("XmasTree");
				namedWindow("XmasTree");
				imshow("XmasTree", *currentImageInArrayPointer);
				waitKey(25);
			}

		}

	}

}