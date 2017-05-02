

#include <iostream>
#include <opencv2\imgproc.hpp>
#include <opencv\cv.h>
#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <array>
#include <math.h>
#include <numeric>

using namespace std;
using namespace cv;

// Classes
class AppleOrchard
{
private:
	// Internal Classes that contain HSV Values and Predicates
	class innerBaseClass
	{
	public:
		int H_Min, H_Max, S_Min, S_Max, V_Min, V_Max, HSV_DILATE, HSV_ERODE, HSV_BLUR;
		Scalar* HSV_LOW = NULL;
		Scalar* HSV_HIGH = NULL;
		innerBaseClass()
		{
			set();
			message();
		}

		virtual void set()
		{
			H_Min = 0;
			H_Max = 0;
			S_Min = 0;
			S_Max = 0;
			V_Min = 0;
			V_Max = 0;
			HSV_DILATE = 0;
			HSV_ERODE = 0;
			HSV_BLUR = 0;
			Scalar* HSV_LOW = new Scalar(H_Min, S_Min, V_Min);
			Scalar* HSV_HIGH = new Scalar(H_Max, S_Max, V_Max);
		}
		virtual void message()
		{
			cout << "Base Class Fired" << endl;
			cout << "H_Max is: " << H_Max << endl;
		}
	};
	class Histogram
	{
	private:
		// Class Variables that Set Up Histogram
		const int WIDTH = 2048;
		const int HEIGHT = 2048;
		int yAxisTop = 0;	 // Sets some space between top of graph to top of image 
		int yAxisBottom = 0; // Sets some space between bottom of graph and top of image
		int xAxisLeft = 0; // Starts xaxis a bit of space away from the left side of image
		int xAxisRight = 0; // ends xaxis a bit of space away from the right side of image
		const int imageSections = 8; // The number of sections the graph is divided into for axis drawing operations
		const int numBins = 200; // The number of slices the data is cut into
		int xAxisDistance = 0; // The distance between the origin and the end of the x-axis 
		int interval = 0; // The x-distance between each bin displayed (xAxisDistance / numBins)
		const int scaleDownRatio = 6; // since the bins usually get relatively full, its a good idea to scale down
		int xSpacer = 0; // Used in "OutputRowHistogram" in conjunction with "interval" to step through the graph 

		// Variables for Constructing Histogram text
		int sumPixelsInBins = 0; // Total sum of all pixels counted
		int averageBinDensity = 0; // average pixels found in each bin summed and averaged from all bins
		double standard_deviation = 0;
		Point2d histogramText; // For writing words on Histogram

		// Class Variables for text output on RowDensity histogram
		string avgBins = "Average Bin Size: ";
		string numRows = "Number of Rows: ";
		string FilePath;
		int fontFace = CV_FONT_HERSHEY_PLAIN;
		const double fontScale = 2;
		const int thickness = 2;

		// Very important container that holds the histogram data
		vector<int> binContainer; // will hold all of the horizontal / vertical density bins for a given picture, depending which algorithm is called
		Mat thresholdImage; // given to object when class is instantiated

		// Private Methods

		// Function draws axis on the histogram image
		void drawAxis() // Input an Image to use
		{// Put x and y axis on our blank image
			line(histImg, Point(xAxisLeft, yAxisTop), Point(xAxisLeft, yAxisBottom), Scalar(255, 0, 0), 15); // y-axis
			line(histImg, Point(xAxisLeft, yAxisBottom), Point(xAxisRight, yAxisBottom), Scalar(255, 0, 0), 15); // x-axis
		}

		// Function executes when outputRowHistogram in outer class is called
		void MeanAndStandardDeviation_BinContainer()
		{
			double totalBinSum = 0;
			int numElements = binContainer.size();
			double temp = 0;
			double variance = 0;
			for (vector<int>::iterator itMean = binContainer.begin(); itMean != binContainer.end(); ++itMean)
			{
				totalBinSum += *itMean;
			}
			averageBinDensity = totalBinSum / numElements;

			for (vector<int>::iterator itStand = binContainer.begin(); itStand != binContainer.end(); ++itStand)
			{
				temp += pow(abs(averageBinDensity - *itStand), 2);
			}
			variance = temp / numElements;
			standard_deviation = sqrt(variance);
		}

		/* findThresholdDensity finds the density of a threshold image you provide by either slicing the image horizontally or verticaly.
		findThresholdDensity sets the binContainer variable which is used to output a threshold density histogram. Call this function almost immediately.
		Enter threshold Image you would like to find the bin density of
		Select either true or false for second paramter: False = Horizontal Binning. True = Veritical Binning */
		void findThresholdDensity(Mat thresholdImage, bool flip = false)
		{
			Mat orchardThreshold;
			thresholdImage.copyTo(orchardThreshold);
			// 1) Declare internal variables
			int numBins = 200; // This divides the picture into slivers
			int binWidth = 0; // Number of pixels within a sliver. Will be set differently depending on flip being true of false
			int counter = 0; // generic counter
			double whitePixelBinCount = 0; // the amount of white pixels counted in a bin
			vector <double> temp; // all purpose temporary vector

			// 2) Depending on the users input, the switch case will either run true or false. 
			// False indicates the threshold image will be horizontally binned. True indicates that the threshold image will be vertically binned
			binContainer.clear(); // Wipe binContainer clean
			binContainer.reserve(200); // reserve 200 spots for the vector
			switch (flip)
			{
			case false: // If false is exectued, then the threshold picture will be horizontally binned
				binWidth = orchardThreshold.rows / numBins; // Number of pixels within a bin. Here it corresponds to horizontal slivers
				// 3) Loop through all rows. Once row reaches binWidth, store the number of pixels counted in binContainer.
				// binWidth is currently set to 15 (3000/200).. so every 15 rows the pixel count will be stored
				for (int _row = 0; _row < orchardThreshold.rows; _row++)
				{
					for (int _column = 0; _column < orchardThreshold.cols; _column++)
					{ // We need to add the white pixels to find a total amount of pixels 
						whitePixelBinCount += orchardThreshold.at<uchar>(_row, _column) / 255;
					}
					counter++;
					if (counter == binWidth)
					{
						counter = 0;
						binContainer.push_back(whitePixelBinCount); // store bin pixel count in binContainer
						whitePixelBinCount = 0;
					}
				}
				break;

			case true: // If true is exectued, then the threshold image will be vertically binned
				binWidth = orchardThreshold.cols / numBins; // Number of pixels within a sliver. Here it corresponds to vertical slivers
				for (int _column = 0; _column < orchardThreshold.cols; _column++)
				{
					for (int _row = 0; _row < orchardThreshold.rows; _row++)
					{ // We need to sum the white pixels to find a total amount of pixels per bin
						whitePixelBinCount += orchardThreshold.at<uchar>(_row, _column) / 255;
					}
					counter++;
					if (counter == binWidth)
					{
						counter = 0;
						binContainer.push_back(whitePixelBinCount); // store bin pixel count in binContainer
						whitePixelBinCount = 0;
					}
				}
				break;

			default:
				cout << "Something Didnt Work Right" << endl;
				break;
			}
			// 4) Using the newly filled binContainer, find the mean and standard deviation of the binData
			MeanAndStandardDeviation_BinContainer();
		}

	public:
		// The Histogram image used for outputting
		Mat histImg; // used to output the threshold density histogram image

		Histogram(Mat& _thresholdImage, string _FilePath)
		{
			// Constructor will set object parameters for the histogram
			_thresholdImage.copyTo(thresholdImage);
			FilePath = _FilePath;
			histImg = Mat(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255)); // used to output the threshold density histogram image
			yAxisTop = histImg.rows / imageSections; // sets some space between top of graph to top of image 
			yAxisBottom = histImg.rows - yAxisTop; // sets some space between bottom of graph and top of image
			xAxisLeft = histImg.cols / imageSections; // starts xaxis a bit of space away from the left side of image
			xAxisRight = histImg.cols - xAxisLeft; // ends xaxis a bit of space away from the right side of image
			xAxisDistance = xAxisRight - xAxisLeft; // complete x-axis distance
			interval = xAxisDistance / numBins; // the step size between bins on the graph
			xSpacer = xAxisLeft; // xSpacer is user to step through the graph and draw it
			histogramText = Point2d(xAxisLeft, yAxisTop - 100);
		}

		void drawHistogram(bool flip)
		{
			// 1) Draw histogram axis
			drawAxis();

			// 2) We need to run findThresholdDensity to fill the bin container
			findThresholdDensity(thresholdImage, flip);

			// 3) Output Bins on Histogram by drawing rectangles on histogram image
			for (vector<int>::iterator countContainer = binContainer.begin(); countContainer != binContainer.end(); ++countContainer)
			{// Loop through the container and remove the bin sizes we have found
				rectangle(histImg, Point(xSpacer, yAxisBottom), Point(xSpacer + interval, yAxisBottom - (*countContainer / scaleDownRatio)), Scalar(0, 0, 255), -1);
				xSpacer += interval; // increment to the next bin space
			}

			// 4) Turn average value into a string so openCV can output it on your histogram
			avgBins += to_string(averageBinDensity);

			// 5) Draw Average Line and Output text
			line(histImg, Point(xAxisLeft, yAxisBottom - averageBinDensity / scaleDownRatio), Point(xAxisRight, yAxisBottom - averageBinDensity / scaleDownRatio), Scalar(0, 255, 0), 15); // Shows the average bin density on the histogram
			putText(histImg, avgBins, histogramText, fontFace, fontScale, Scalar(255, 0, 255), thickness);

			// 6) Write image to filepath that the user instantiated the object with
			imwrite(FilePath, histImg);
		}

		~Histogram();

	};

	class HSV_Values_Rows : public innerBaseClass
	{
	public:
		HSV_Values_Rows()
		{
			HSV_Values_Rows::set();
			HSV_Values_Rows::message();
		}
		virtual void set()
		{
			H_Min = 9;
			H_Max = 30;
			S_Min = 145;
			S_Max = 162;
			V_Min = 0;
			V_Max = 145;
			HSV_DILATE = 2;
			HSV_ERODE = 4;
			HSV_BLUR = 5;
			HSV_LOW = new Scalar(H_Min, S_Min, V_Min);
			HSV_HIGH = new Scalar(H_Max, S_Max, V_Max);
		}
		void message()
		{
			cout << "Inherited Class Fired" << endl;
			cout << "H_Max is: " << H_Max << endl;
		}

	};
	class HSV_Values_YellowApples : public innerBaseClass
	{
	public:
		HSV_Values_YellowApples()
		{
			set();
			message();
		}

		void set()
		{
			H_Min = 0;
			H_Max = 31;
			S_Min = 152;
			S_Max = 255;
			V_Min = 180;
			V_Max = 255;
			// These are the Key Scalar Values needed to make a correct threshold image
			HSV_DILATE = 1;
			HSV_ERODE = 2;
			HSV_BLUR = 1;
			HSV_LOW = new Scalar(H_Min, S_Min, V_Min);
			HSV_HIGH = new Scalar(H_Max, S_Max, V_Max);
		}
		void message()
		{
			cout << "Inherited Class Fired" << endl;
			cout << "H_Max is: " << H_Max << endl;
		}

	};
	class HSV_Values_RedApples : public innerBaseClass
	{
	public:
		HSV_Values_RedApples()
		{
			set();
			message();
		}

		void set()
		{
			// These Values are not correct
			H_Min = 0;
			H_Max = 0;
			S_Min = 0;
			S_Max = 0;
			V_Min = 0;
			V_Max = 0;
			// These are the Key Scalar Values needed to make a correct threshold image
			HSV_DILATE = 0;
			HSV_ERODE = 0;
			HSV_BLUR = 1;
			HSV_LOW = new Scalar(H_Min, S_Min, V_Min);
			HSV_HIGH = new Scalar(H_Max, S_Max, V_Max);
		}
		void message()
		{
			cout << "Inherited Class Fired" << endl;
			cout << "H_Max is: " << H_Max << endl;
		}

	};
	struct sort_pred_double_descending {
		bool operator()(const pair<double, Point2i> &left, const pair<double, Point2i> &right)
		{
			return left.first > right.first; // first value is the largest
		}
	};

	// Class Variables that Set Up Histogram
	const int WIDTH = 2048;
	const int HEIGHT = 2048;
	vector<int> binContainer; // will hold all of the horizontal / vertical density bins for a given picture, depending which algorithm is called

	// Variables for Constructing Histogram text
	int imageHeight = 0;
	int imageWidth = 0;
	int sumPixelsInBins = 0; // Total sum of all pixels counted
	int averageBinDensity = 0; // average pixels found in each bin
	double standard_deviation = 0;
	Point2d histogramText; // For writing words on Histogram

	// Class Variables for text output on RowDensity histogram
	string avgBins = "Average Bin Size: ";
	string numRows = "Number of Rows: ";
	int fontFace = CV_FONT_HERSHEY_PLAIN;
	const double fontScale = 4;
	const int thickness = 2;
	const int numBins = 200;

	// Variables For Counting Apples In First Row
	int appleMeanArea = 0;
	int appleStdDeviation = 0;
	int appleRadius = 0;
	int interquartileAppleMean = 0;
	int interquartileAppleStdDeviation = 0;
	int interquartileAppleRadius = 0;

	// Get and Set Operations for Constructor
	void GetSetObjectProperties(Mat _BGRImage) // This GetSetGraphProperties is meant for constructor 1
	{
		// Constructor will use these to set the objects parameters
		_BGRImage.copyTo(BGRImage);
		imageHeight = _BGRImage.rows;
		imageWidth = _BGRImage.cols;

		// Constructor will find the threshold Image for both the Apple Orchard Rows and the Apples themselves
		HSV_Values_YellowApples OrchardApples; //  HSV Values for Morphological Ops
		HSV_Values_Rows OrchardRows; // HSV Values for Morpholigical Ops
		thresholdImg_Rows = convertRGB2HSV(BGRImage, &OrchardRows); // Convert BRG to Threshold Image of Orchard Rows using Row HSV Values
		thresholdImg_Apples = convertRGB2HSV(BGRImage, &OrchardApples); // Convert BRG to Threshold Image of Orchard Rows using Row HSV Values
	}

	// Private Functions
	// Function executes when outputRowHistogram is called
	void MeanAndStandardDeviation_Rows()
	{
		double totalBinSum = 0;
		int numElements = binContainer.size();
		double temp = 0;
		double variance = 0;
		for (vector<int>::iterator itMean = binContainer.begin(); itMean != binContainer.end(); ++itMean)
		{
			totalBinSum += *itMean;
		}
		averageBinDensity = totalBinSum / numElements;

		for (vector<int>::iterator itStand = binContainer.begin(); itStand != binContainer.end(); ++itStand)
		{
			temp += pow(abs(averageBinDensity - *itStand), 2);
		}
		variance = temp / numElements;
		standard_deviation = sqrt(variance);
	}
	// Function exectutes when findApples is called. Calculates both mean & std_dev and interquartile mean & interquartile std_dev
	void MeanAndStandardDeviation_Apples()
	{
		// Define Temporary Variables -- This 
		int _size = appleCoordinateContainer.size(); // size of the container
		int quartile = _size / 4; // size of one quartile
		int tempSum = 0; // temporary sum

		// 1) Find the apple mean
		for (vector< pair<double, Point2i> >::iterator itA = appleCoordinateContainer.begin(); itA != appleCoordinateContainer.end(); ++itA)
		{
			tempSum += itA->first;
		}
		appleMeanArea = tempSum / _size;
		appleRadius = std::sqrt(appleMeanArea / 3.14);
		// 2) Find apple standard deviation
		tempSum = 0;
		for (vector< pair<double, Point2i> >::iterator itA = appleCoordinateContainer.begin(); itA != appleCoordinateContainer.end(); ++itA)
		{
			tempSum += std::pow(std::abs(itA->first - appleMeanArea), 2);
		}
		appleStdDeviation = std::sqrt((tempSum / _size));

		// 3) Find interquartile apple mean 
		tempSum = 0;
		for (vector< pair<double, Point2i> >::iterator itA = appleCoordinateContainer.begin() + quartile; itA < appleCoordinateContainer.begin() + 3 * quartile; ++itA)
		{
			tempSum += itA->first;
		}
		interquartileAppleMean = (tempSum / (_size / 2));
		interquartileAppleRadius = std::sqrt(interquartileAppleMean / 3.14);

		// 4) Find interquartile apple standard deviation
		tempSum = 0;
		for (vector< pair<double, Point2i> >::iterator itA = appleCoordinateContainer.begin() + quartile; itA < appleCoordinateContainer.begin() + 3 * quartile; ++itA)
		{
			tempSum += std::pow(std::abs(itA->first - interquartileAppleMean), 2);
		}
		interquartileAppleStdDeviation = std::sqrt((tempSum / (_size / 2)));
	}
	// Crop The Image To Only View Objects in First Row
	void cropBGRtoFirstRow(string FilePath)
	{
		int upperLeftYCoord = rowCoordinateContainer[1].second[1]; // Crop image from the start of the second row to the bottom of the first row
		int upperRightXCoord = 0; // Tiny bit of buffer room
		int buffer = 100; // To make sure no tree is getting cuttoff.
		int height = rowCoordinateContainer[0].second[1] - rowCoordinateContainer[1].second[1] - buffer;
		int width = 3999;
		Mat ROI(BGRImage, Rect(upperRightXCoord, upperLeftYCoord, width, height));
		ROI.copyTo(BGRImage_FirstRow);
		imwrite(FilePath, BGRImage_FirstRow);
	}

	/* findThresholdDensity finds the density of a threshold image you provide by either slicing the image horizontally or verticaly.
	findThresholdDensity sets the binContainer variable which is used to output a threshold density histogram. Call this function almost immediately.
	Enter number to select which operation to conduct: 1 for Row Operations, 2 for Apple Operations
	Select either true or false for second paramter: False = Horizontal Binning. True = Veritical Binning */
	void findThresholdDensity(Mat& inputImg, bool flip = false)
	{
		// 1) Store inputImg (Needs to be a threshold Image for Binning to work) into orcharThreshold
		Mat orchardThreshold;
		inputImg.copyTo(orchardThreshold);

		// 2) These will be used to bin the threshold image
		const int numBins = 200; // This divides the picture into slivers
		int binWidth = 0; // Number of pixels within a sliver. Will be set differently depending on flip being true of false
		int counter = 0; // generic counter
		double whitePixelBinCount = 0; // the amount of white pixels counted in a bin
		vector <double> temp; // all purpose temporary vector

		// 3) Loop through all rows. Once row reaches binWidth, store the number of pixels counted in binContainer.
		// binWidth is currently set to 15 (3000/200).. so every 15 rows the pixel count will be stored
		binContainer.reserve(200); // reserve 200 spots for the vector
		switch (flip)
		{
		case false: // If false is exectued, then the horizontal density histogram will be found
			binWidth = orchardThreshold.rows / numBins; // Number of pixels within a sliver. Here it corresponds to horizontal slivers
			for (int _row = 0; _row < orchardThreshold.rows; _row++)
			{
				for (int _column = 0; _column < orchardThreshold.cols; _column++)
				{ // We need to add the white pixels to find a total amount of pixels 
					whitePixelBinCount += orchardThreshold.at<uchar>(_row, _column) / 255;
				}
				counter++;
				if (counter == binWidth)
				{
					counter = 0;
					binContainer.push_back(whitePixelBinCount);
					whitePixelBinCount = 0;
				}
			}
			break;

		case true: // If true is exectued, then the vertical density histogram will be found 
			binWidth = orchardThreshold.cols / numBins; // Number of pixels within a sliver. Here it corresponds to vertical slivers
			for (int _column = 0; _column < orchardThreshold.cols; _column++)
			{
				for (int _row = 0; _row < orchardThreshold.rows; _row++)
				{ // We need to add the white pixels to find a total amount of pixels 
					whitePixelBinCount += orchardThreshold.at<uchar>(_row, _column) / 255;
				}
				counter++;
				if (counter == binWidth)
				{
					counter = 0;
					binContainer.push_back(whitePixelBinCount);
					whitePixelBinCount = 0;
				}
			}
			break;

		default:
			cout << "Something Didnt Work Right" << endl;
			break;
		}
		// 4) Add Buffer room at the end of the binContainer -- May rethink this
		//vector<int>zeros(5, 0);
		//vector<int>::iterator it = binContainer.end();
		//binContainer.insert(it, zeros.begin(), zeros.end());
		// 4) Now That binContainer is filled, go ahead and find the average and standard deviation
		MeanAndStandardDeviation_Rows();
	}

public:
	// Image Variables 
	Mat BGRImage; // The read-in BGR Orchard Image that initialized the object itself
	Mat BGRImage_FirstRow; // The BGR image cropped to the first row specifications
	// Threshold Images of the Rows and the Apples found using the inner HSV_Values classes
	Mat thresholdImg_Apples; // Threshold Image of the apples on the trees 
	Mat thresholdImg_Apples_FirstRow; // Threshold Image of the apples in first row
	Mat thresholdImg_Rows; // Threshold Image of the rows to identify trees
	int appleCount = 0;

	// Very Important Row Coordinate Container. Holds the Row information for all rows found..Usually only need first or second row
	vector<pair<string, array<int, 2> > > rowCoordinateContainer; // Holds the row information per picture. Format: "Row X", [Lower Cord, Upper Cord]
	vector<pair<double, Point2i> > appleCoordinateContainer; // Holds the Apple information per picture. Format: "<Area,(xCord,yCord)>"

	/* Defining Constructor Variables:
	inputImageFileName = The blank image you wish to construct your graph on
	thresholdImg = The image which you would like to find the density of white pixels for
	outputImageFileName = The output file you would like to write the histogram to
	HSV_Low, HSV_High = If you supply a RGB Img, you must also supply HSV inRange values
	numBins = assumed to be 200, change to more or less if you wish
	*/

	//_________________________________Constructor________________________//
	// You can setup your object with the internal graph that is provided
	AppleOrchard(Mat _BGRImage)
	{
		GetSetObjectProperties(_BGRImage);
	}

	//_________________________________Public Member Functions________________________//

	// To output the row density histogram, give the function the address of what folder to write the image to
	void outputRowHistogram(string outputFilePath)
	{// Use Histogram Object to output the Row Histogram
		Histogram* rowHistogram = new Histogram(thresholdImg_Rows, outputFilePath);
		rowHistogram->drawHistogram(false);
	}

	// To use findRowLocations, make sure you run findThresholdDensity first with the parameter ThresholdImg_Rows Matrix (Since that holds the row threshold)
	// This algorithm will use the binContainer to determine where the rows in the Orchard Image are located
	void findRowLocations()
	{ // This algorithm will find the rows inside a picture, and output their locations and coordinates on the BGR Image

		// 1) Call findThresholdDensity
		findThresholdDensity(thresholdImg_Rows, false);
		
		// 2) Find rows by using this algorithm Below
		const int CONFIDENCE_LEVEL = 5; // If we have over ten bins in a row that are greater than the average, than most likely its a row
		const int BELOW_AVERAGE_LEVEL = 5; // We need to see if the our bin count 
		const double rowComparisonValue = averageBinDensity + standard_deviation; // compares bin values to 3 deviations higher than the mean bin-value   
		const double fieldComparisonValue = averageBinDensity - standard_deviation;
		const int yAxisIncrement = imageHeight / numBins; // If Height is 3000 pixels, then the increment should be (3000/200 = 15)
		const int rowPadding = 100;
		int compareRowCount = 0; // Check to see if bin values correspond to rows
		int compareFieldCount = 0; // Check to see if bin values correspond to field space 
		int yImageValue = imageHeight; // height of picture
		int rowCounter = 0;
		bool rowStart = false;
		pair<string, array<int, 2> >* myRow = NULL;


		for (reverse_iterator<vector<int>::iterator> itb = binContainer.rbegin(); itb != binContainer.rend(); ++itb)
		{// starts at the end of the binContainer. The entries at the end will correspond to the first row

			// If the bin value is above the rowComparisonValue, and we are between rows, then we need to increment the compareRowCount variable 
			if ((*itb > averageBinDensity) && (rowStart == false))
			{
				++compareRowCount;
			}

			// If the bin value is not above the rowComparisonValue, and we are between rows, then we need to reset compareRowCount
			if ((*itb < averageBinDensity) && (rowStart == false))
			{
				compareRowCount = 0;
			}

			// If the bin value is above the rowComparisonValue, and we are on a row, then we do not need to do anything
			if ((*itb > averageBinDensity) && (rowStart == true))
			{
				compareFieldCount = 0;
			}

			// If the bin value is below the fieldComparisonValue, and a row has been found, then we need to increment compareFieldCounter
			if ((*itb < averageBinDensity) && (rowStart == true))
			{
				++compareFieldCount;
			}

			// If the compareRowCount is at confidence_level.. then we are going to store the row end coordinate and turn rowStart on
			if (compareRowCount == CONFIDENCE_LEVEL) // Then we have found a row. Let's store the y axis location 
			{
				rowStart = true;
				myRow = new pair<string, array<int, 2> >; // dynamically create another pair object that myRow will point at
				myRow->first = "Row " + to_string(++rowCounter);
				myRow->second[0] = NULL; // placeholder value
				myRow->second[1] = yImageValue + 5 * yAxisIncrement; // Bottom Row value. Since we checked to find 5 confidence markers, backtrack 5 spots to find the start
				compareRowCount = NULL; // Now that we found a row, reset compareRowCount
			}

			// If the compareFieldCount is at confidence_level.. then we are going to store the row begin coordinate and turn rowStart off
			if (compareFieldCount == CONFIDENCE_LEVEL) // Then we have found a row. Let's store the y axis location 
			{
				rowStart = false;
				myRow->second[0] = yImageValue + 5 * yAxisIncrement; // Top Row value. Since we checked to find 5 confidence markers, backtrack 5 spots to find the start
				rowCoordinateContainer.push_back(*myRow);
				compareFieldCount = NULL; // Now that we found the end of the row, lets set the compareFieldCount to zero to restart process
				myRow = NULL; // Now that we are done with the pointer, lets set it to NULL
			}

			yImageValue -= yAxisIncrement;
		}

	}

	// This Algorithm is simply used to output the row coordinates on the BGRImage. This is a great way to visualize how accurate findRowLocations worked
	void visualizeRowCoordinates(string outputFilePath)
	{
		// Draw rectangles around the rows found
		for (vector<pair<string, array<int, 2> > >::iterator itRow = rowCoordinateContainer.begin(); itRow != rowCoordinateContainer.end(); ++itRow)
		{
			// Draws a rectangle around each and every row identified. 
			rectangle(BGRImage, Point(10, itRow->second[0]), Point(BGRImage.cols - 10, itRow->second[1]), Scalar(0, 0, 255), 3);
			putText(BGRImage, itRow->first, Point(BGRImage.cols / 2, (itRow->second[0] + itRow->second[1]) / 2), fontFace, fontScale, Scalar(255, 0, 255), thickness);
		}

		// Output the file to the correct folder
		imwrite(outputFilePath, BGRImage);
	}

	// Call this Algorithm after you have identified the individual rows and wish to count the individual TREES inside of that row
	void CountTreesInRow()
	{

	}

	void writeImageToFile(string FilePath, Mat& ImageToWrite)
	{
		imwrite(FilePath, ImageToWrite);
	}

	// Call this Algorithm after you have identified the individual rows and wish to count the individual APPLES inside of that row
	void CountApplesInFirstRow(string FilePath)
	{
		// 1) We need to use the row container information to crop the BGR photo to only show the first row specifications
		cropBGRtoFirstRow(FilePath);
		// 2)  Assume most objects in picture are apples, this will be corrected with more ImageLabTesting Filtering Routines
		HSV_Values_YellowApples Apple_HSV_Values;
		thresholdImg_Apples_FirstRow = convertRGB2HSV(BGRImage_FirstRow, &Apple_HSV_Values); // converts cropped BGR Image to a cropped threshold image

		// 3) We will need to use the moments method to find all of the objects in the cropped image
		vector <vector <Point> >contours; // holds the contours found for the apples
		vector<Vec4i> hierarchy; // holds the hierarchial information for contours 
		Moments moment;          // moments are used to calculate area values and x-y coordinates
		double objectArea;	     // Used to store the object area 
		Point2i objectCenter;    // Used to store the x,y coordinates of the objects
		pair<double, Point2i> storeObjectData; // holds the pair of data


		findContours(thresholdImg_Apples_FirstRow, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		// 4) Take the contours, find areas, and store the areas, and the coordinates of the objects into a "pair vector"
		int count = -1;
		for (vector < vector<Point> >::iterator itc = contours.begin(); itc != contours.end(); ++itc)
		{
			++count;
			moment = moments((Mat)*itc);
			// 5) If an object is greater than an agreed upon small apple size, we will count it as an apple
			if (moment.m00 > 400) // Then it is a contender to be an apple
			{
				// 6) Store all the coordinates of the apples in a "pair" vector composed of their areas and their x,y coordinates in a Point
				objectArea = moment.m00; // holds object area
				objectCenter = Point2i(moment.m10 / objectArea, moment.m01 / objectArea); // holds centerpoint
				storeObjectData = pair<double, Point2i>(objectArea, objectCenter); // holds the pair of data
				appleCoordinateContainer.push_back(storeObjectData); // push data set into our container
			}
		}
		
		// 7) Sort the data from greatest to smallest. End goal is to find the mean, median and mode
		sort(appleCoordinateContainer.begin(), appleCoordinateContainer.end(), sort_pred_double_descending());
		appleCount = appleCoordinateContainer.size();
		// 8) Find the interquartile mean and use this information to determine an average circle radius
		MeanAndStandardDeviation_Apples();

		// 9) Draw circles around each of the found apples, and output the total count, and mean size at the top of the Image
		for (vector< pair<double, Point2i> >::iterator itA = appleCoordinateContainer.begin(); itA != appleCoordinateContainer.end(); ++itA)
		{
			circle(BGRImage_FirstRow, itA->second, interquartileAppleRadius, Scalar(0, 0, 255), 2);
		}
		// 10) Write image to a File to view
		string writeText = "Apples Found: " + to_string(appleCount);
		putText(BGRImage_FirstRow, writeText, Point(50, 50), fontFace, fontScale, Scalar(255, 0, 255), thickness);
		imwrite(FilePath, BGRImage_FirstRow);
	}

	// Simple dilation on an image using a dilation value provided
	void dilateImage(int ORCHARD_DILATE, Mat& sourceImg)
	{
		// We are assuming that the dilation effect will be a rectangular dilation
		Mat outputImage;
		// Create the structuring element for dilation
		Mat element = getStructuringElement(0, Size(2 * ORCHARD_DILATE + 1, 2 * ORCHARD_DILATE + 1),
			Point(ORCHARD_DILATE, ORCHARD_DILATE));
		dilate(sourceImg, outputImage, element);
		//return outputImage;
	}

	// Simple erosion on an image using a erosion value provided
	void erodeImage(int ORCHARD_ERODE, Mat& sourceImg)
	{
		Mat outputImg;
		// Create the structuring element for erosion
		Mat element = getStructuringElement(1, Size(2 * ORCHARD_ERODE + 1, 2 * ORCHARD_ERODE + 1),
			Point(ORCHARD_ERODE, ORCHARD_ERODE));
		// Erode the image using the structuring element
		erode(sourceImg, outputImg, element);
		//return outputImg;
	}

	// Simple Blur to take away some of the edge in objects
	void blurImage(int ORCHARD_BLUR, Mat& sourceImg)
	{
		// Simply blur the image to different values, which will be used to determine if the blurred filter Image will 
		// lead to better image recognition
		Mat blurredImage;
		medianBlur(sourceImg, blurredImage, ORCHARD_BLUR);
		//return blurredImage;
	}

	// since all HSV_Value classes inheret from the innerBaseClass, this is the best way to make sure all operations can convert using the method below
	Mat convertRGB2HSV(Mat& BGRImg, innerBaseClass* ORCHARD)
	{
		// HSV Values for Morphological Operations
		ORCHARD->set();
		Mat Threshold, HSVImg, Threshold_Temp;
		cvtColor(BGRImg, HSVImg, CV_BGR2HSV);
		inRange(HSVImg, Scalar(ORCHARD->H_Min, ORCHARD->S_Min, ORCHARD->V_Min), Scalar(ORCHARD->H_Max, ORCHARD->S_Max, ORCHARD->V_Max), Threshold_Temp);
		/*Threshold_Temp = dilateImage(ORCHARD->HSV_DILATE, Threshold_Temp); // dilate
		Threshold_Temp = erodeImage(ORCHARD->HSV_ERODE, Threshold_Temp);   // erode
		Threshold = blurImage(ORCHARD->HSV_BLUR, Threshold_Temp); // blur
		*/
		dilateImage(ORCHARD->HSV_DILATE, Threshold_Temp); // dilate
		erodeImage(ORCHARD->HSV_ERODE, Threshold_Temp);   // erode
		blurImage(ORCHARD->HSV_BLUR, Threshold_Temp); // blur
		Threshold_Temp.copyTo(Threshold);
		return Threshold;
	}
};

class BlossomOrchard
{
private:
	// Internal Classes Used for HSV Operations
	class innerBaseClass
	{
	public:
		int H_Min, H_Max, S_Min, S_Max, V_Min, V_Max, HSV_DILATE, HSV_ERODE, HSV_BLUR;
		Scalar* HSV_LOW = NULL;
		Scalar* HSV_HIGH = NULL;
		innerBaseClass()
		{
			set();
			message();
		}

		virtual void set()
		{
			H_Min = 0;
			H_Max = 0;
			S_Min = 0;
			S_Max = 0;
			V_Min = 0;
			V_Max = 0;
			HSV_DILATE = 0;
			HSV_ERODE = 0;
			HSV_BLUR = 1;
			Scalar* HSV_LOW = new Scalar(H_Min, S_Min, V_Min);
			Scalar* HSV_HIGH = new Scalar(H_Max, S_Max, V_Max);
		}
		virtual void message()
		{
			cout << "Base Class Fired" << endl;
			cout << "H_Max is: " << H_Max << endl;
		}
	};
	class Histogram
	{
	private:
		// Class Variables that Set Up Histogram
		const int WIDTH = 2048;
		const int HEIGHT = 2048;
		int yAxisTop = 0;	 // Sets some space between top of graph to top of image 
		int yAxisBottom = 0; // Sets some space between bottom of graph and top of image
		int xAxisLeft = 0; // Starts xaxis a bit of space away from the left side of image
		int xAxisRight = 0; // ends xaxis a bit of space away from the right side of image
		const int imageSections = 8; // The number of sections the graph is divided into for axis drawing operations
		const int numBins = 200; // The number of slices the data is cut into
		int xAxisDistance = 0; // The distance between the origin and the end of the x-axis 
		int interval = 0; // The x-distance between each bin displayed (xAxisDistance / numBins)
		const int scaleDownRatio = 6; // since the bins usually get relatively full, its a good idea to scale down
		int xSpacer = 0; // Used in "OutputRowHistogram" in conjunction with "interval" to step through the graph 

		// Variables for Constructing Histogram text
		int sumPixelsInBins = 0; // Total sum of all pixels counted
		int averageBinDensity = 0; // average pixels found in each bin summed and averaged from all bins
		double standard_deviation = 0;
		Point2d histogramText; // For writing words on Histogram

		// Class Variables for text output on RowDensity histogram
		string avgBins = "Average Bin Size: ";
		string numRows = "Number of Rows: ";
		string FilePath;
		int fontFace = CV_FONT_HERSHEY_PLAIN;
		const double fontScale = 2;
		const int thickness = 2;

		// Very important container that holds the histogram data
		vector<int> binContainer; // will hold all of the horizontal / vertical density bins for a given picture, depending which algorithm is called
		Mat thresholdImage; // given to object when class is instantiated

		// Private Methods

		// Function draws axis on the histogram image
		void drawAxis() // Input an Image to use
		{// Put x and y axis on our blank image
			line(histImg, Point(xAxisLeft, yAxisTop), Point(xAxisLeft, yAxisBottom), Scalar(255, 0, 0), 15); // y-axis
			line(histImg, Point(xAxisLeft, yAxisBottom), Point(xAxisRight, yAxisBottom), Scalar(255, 0, 0), 15); // x-axis
		}

		// Function executes when outputRowHistogram in outer class is called
		void MeanAndStandardDeviation_BinContainer()
		{
			double totalBinSum = 0;
			int numElements = binContainer.size();
			double temp = 0;
			double variance = 0;
			for (vector<int>::iterator itMean = binContainer.begin(); itMean != binContainer.end(); ++itMean)
			{
				totalBinSum += *itMean;
			}
			averageBinDensity = totalBinSum / numElements;

			for (vector<int>::iterator itStand = binContainer.begin(); itStand != binContainer.end(); ++itStand)
			{
				temp += pow(abs(averageBinDensity - *itStand), 2);
			}
			variance = temp / numElements;
			standard_deviation = sqrt(variance);
		}

		/* findThresholdDensity finds the density of a threshold image you provide by either slicing the image horizontally or verticaly.
		findThresholdDensity sets the binContainer variable which is used to output a threshold density histogram. Call this function almost immediately.
		Enter threshold Image you would like to find the bin density of
		Select either true or false for second paramter: False = Horizontal Binning. True = Veritical Binning */
		void findThresholdDensity(Mat thresholdImage, bool flip = false)
		{
			Mat orchardThreshold;
			thresholdImage.copyTo(orchardThreshold);
			// 1) Declare internal variables
			int numBins = 200; // This divides the picture into slivers
			int binWidth = 0; // Number of pixels within a sliver. Will be set differently depending on flip being true of false
			int counter = 0; // generic counter
			double whitePixelBinCount = 0; // the amount of white pixels counted in a bin
			vector <double> temp; // all purpose temporary vector

			// 2) Depending on the users input, the switch case will either run true or false. 
			// False indicates the threshold image will be horizontally binned. True indicates that the threshold image will be vertically binned
			binContainer.clear(); // Wipe binContainer clean
			binContainer.reserve(200); // reserve 200 spots for the vector
			switch (flip)
			{
			case false: // If false is exectued, then the threshold picture will be horizontally binned
				binWidth = orchardThreshold.rows / numBins; // Number of pixels within a bin. Here it corresponds to horizontal slivers
				// 3) Loop through all rows. Once row reaches binWidth, store the number of pixels counted in binContainer.
				// binWidth is currently set to 15 (3000/200).. so every 15 rows the pixel count will be stored
				for (int _row = 0; _row < orchardThreshold.rows; _row++)
				{
					for (int _column = 0; _column < orchardThreshold.cols; _column++)
					{ // We need to add the white pixels to find a total amount of pixels 
						whitePixelBinCount += orchardThreshold.at<uchar>(_row, _column) / 255;
					}
					counter++;
					if (counter == binWidth)
					{
						counter = 0;
						binContainer.push_back(whitePixelBinCount); // store bin pixel count in binContainer
						whitePixelBinCount = 0;
					}
				}
				break;

			case true: // If true is exectued, then the threshold image will be vertically binned
				binWidth = orchardThreshold.cols / numBins; // Number of pixels within a sliver. Here it corresponds to vertical slivers
				for (int _column = 0; _column < orchardThreshold.cols; _column++)
				{
					for (int _row = 0; _row < orchardThreshold.rows; _row++)
					{ // We need to sum the white pixels to find a total amount of pixels per bin
						whitePixelBinCount += orchardThreshold.at<uchar>(_row, _column) / 255;
					}
					counter++;
					if (counter == binWidth)
					{
						counter = 0;
						binContainer.push_back(whitePixelBinCount); // store bin pixel count in binContainer
						whitePixelBinCount = 0;
					}
				}
				break;

			default:
				cout << "Something Didnt Work Right" << endl;
				break;
			}
			// 4) Using the newly filled binContainer, find the mean and standard deviation of the binData
			MeanAndStandardDeviation_BinContainer();
		}

	public:
		// The Histogram image used for outputting
		Mat histImg; // used to output the threshold density histogram image

		Histogram(Mat& _thresholdImage, string _FilePath)
		{
			// Constructor will set object parameters for the histogram
			_thresholdImage.copyTo(thresholdImage);
			FilePath = _FilePath;
			histImg = Mat(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255)); // used to output the threshold density histogram image
			yAxisTop = histImg.rows / imageSections; // sets some space between top of graph to top of image 
			yAxisBottom = histImg.rows - yAxisTop; // sets some space between bottom of graph and top of image
			xAxisLeft = histImg.cols / imageSections; // starts xaxis a bit of space away from the left side of image
			xAxisRight = histImg.cols - xAxisLeft; // ends xaxis a bit of space away from the right side of image
			xAxisDistance = xAxisRight - xAxisLeft; // complete x-axis distance
			interval = xAxisDistance / numBins; // the step size between bins on the graph
			xSpacer = xAxisLeft; // xSpacer is user to step through the graph and draw it
			histogramText = Point2d(xAxisLeft, yAxisTop - 100);
		}

		void drawHistogram(bool flip)
		{
			// 1) Draw histogram axis
			drawAxis();

			// 2) We need to run findThresholdDensity to fill the bin container
			findThresholdDensity(thresholdImage, flip);

			// 3) Output Bins on Histogram by drawing rectangles on histogram image
			for (vector<int>::iterator countContainer = binContainer.begin(); countContainer != binContainer.end(); ++countContainer)
			{// Loop through the container and remove the bin sizes we have found
				rectangle(histImg, Point(xSpacer, yAxisBottom), Point(xSpacer + interval, yAxisBottom - (*countContainer / scaleDownRatio)), Scalar(0, 0, 255), -1);
				xSpacer += interval; // increment to the next bin space
			}

			// 4) Turn average value into a string so openCV can output it on your histogram
			avgBins += to_string(averageBinDensity);

			// 5) Draw Average Line and Output text
			line(histImg, Point(xAxisLeft, yAxisBottom - averageBinDensity / scaleDownRatio), Point(xAxisRight, yAxisBottom - averageBinDensity / scaleDownRatio), Scalar(0, 255, 0), 15); // Shows the average bin density on the histogram
			putText(histImg, avgBins, histogramText, fontFace, fontScale, Scalar(255, 0, 255), thickness);

			// 6) Write image to filepath that the user instantiated the object with
			imwrite(FilePath, histImg);
		}

		~Histogram();

	};
	class HSV_Values_TreeStems : public innerBaseClass
	{
	public:
		HSV_Values_TreeStems()
		{
			set();
			message();
		}
		void set()
		{
			H_Min = 131;
			H_Max = 169;
			S_Min = 0;
			S_Max = 38;
			V_Min = 114;
			V_Max = 197;
			HSV_DILATE = 2;
			HSV_ERODE = 1;
			HSV_BLUR = 19;
			HSV_LOW = new Scalar(H_Min, S_Min, V_Min);
			HSV_HIGH = new Scalar(H_Max, S_Max, V_Max);
		}
		void message()
		{
			cout << "Base Class Fired" << endl;
			cout << "H_Max is: " << H_Max << endl;
		}
	};
	class HSV_Values_BlossomRows : public innerBaseClass
	{
	public:
		HSV_Values_BlossomRows()
		{
			set();
			message();
		}
		void set()
		{
			H_Min = 157;
			H_Max = 255;
			S_Min = 44;
			S_Max = 255;
			V_Min = 0;
			V_Max = 255;
			HSV_DILATE = 2;
			HSV_ERODE = 3;
			HSV_BLUR = 1;
			HSV_LOW = new Scalar(H_Min, S_Min, V_Min);
			HSV_HIGH = new Scalar(H_Max, S_Max, V_Max);
		}
		void message()
		{
			cout << "Base Class Fired" << endl;
			cout << "H_Max is: " << H_Max << endl;
		}

	};
	class HSV_Values_Blossoms : public innerBaseClass
	{
	public:
		HSV_Values_Blossoms()
		{
			set();
			message();
		}

		void set()
		{
			H_Min = 105;
			H_Max = 148;
			S_Min = 12;
			S_Max = 58;
			V_Min = 201;
			V_Max = 255;
			HSV_DILATE = 1;
			HSV_ERODE = 1;
			HSV_BLUR = 1;
			HSV_LOW = new Scalar(H_Min, S_Min, V_Min);
			HSV_HIGH = new Scalar(H_Max, S_Max, V_Max);
		}
		void message()
		{
			cout << "Derived Class Fired" << endl;
			cout << "H_Max is: " << H_Max << endl;
		}

	};
	class HSV_Values_TreeCanopy : public innerBaseClass
	{
	public:
		HSV_Values_TreeCanopy()
		{
			set();
			message();
		}

		void set()
		{
			H_Min = 30;
			H_Max = 112;
			S_Min = 0;
			S_Max = 37;
			V_Min = 80;
			V_Max = 255;
			HSV_DILATE = 13;
			HSV_ERODE = 45;
			HSV_BLUR = 31;
			HSV_LOW = new Scalar(H_Min, S_Min, V_Min);
			HSV_HIGH = new Scalar(H_Max, S_Max, V_Max);
		}
		void message()
		{
			cout << "Derived Class Fired" << endl;
			cout << "H_Max is: " << H_Max << endl;
		}

	};
	struct sort_pred_double_descending {
		bool operator()(const pair<double, Point2i> &left, const pair<double, Point2i> &right)
		{
			return left.first > right.first; // first value is the largest
		}
	};

	// Class Variables for BlossomOrchard
	Mat HSVImg_temp; // Temporary Image that can be used for HSV transformations
	int imageHeight = 0; // The number of pixels that make up the height of the image
	int imageWidth = 0; // The number of pixels that make up the width of the image
	vector<int> binContainer; // will hold all of the horizontal / vertical density bins for a given picture, depending which algorithm is called

	// Variables for Histogram Related Things
	const int numBins = 200;
	int sumPixelsInBins = 0; // Total sum of all pixels counted
	int averageBinDensity = 0; // average pixels found in each bin summed and averaged from all bins
	double standard_deviation = 0;
	Point2d histogramText; // For writing words on Histogram

	// Class Variables for text output on RowDensity histogram
	string avgBins = "Average Bin Size: ";
	string numRows = "Number of Rows: ";
	int fontFace = CV_FONT_HERSHEY_PLAIN;
	const double fontScale = 5;
	const int thickness = 3;

	// Variables For Counting Apples In First Row
	int blossomMeanArea = 0;
	int blossomStdDeviation = 0;
	int blossomRadius = 0;
	int interquartileBlossomMean = 0;
	int interquartileBlossomStdDeviation = 0;
	int interquartileBlossomRadius = 0;

	// Private Functions

	// Get and Set Operations for Constructor
	void GetSetObjectProperties(Mat& _BGRImage) // This GetSetGraphProperties is meant for constructor 1
	{
		// Constructor will use these to set the objects parameters
		_BGRImage.copyTo(BGRImage);
		imageHeight = _BGRImage.rows;
		imageWidth = _BGRImage.cols;

		// Constructor will find the threshold Image for both the Apple Orchard Rows and the Apples themselves
		HSV_Values_Blossoms OrchardBlossoms; //  HSV Values for Morphological Ops
		HSV_Values_BlossomRows OrchardRows; // HSV Values for Morpholigical Ops
		thresholdImg_Rows = convertRGB2HSV(BGRImage, &OrchardRows); // Convert BRG to Threshold Image of Orchard Rows using Row HSV Values
		thresholdImg_Blossoms = convertRGB2HSV(BGRImage, &OrchardBlossoms); // Convert BRG to Threshold Image of Orchard Rows using Row HSV Values
	}

	/* findThresholdDensity finds the density of a threshold image you provide by either slicing the image horizontally or verticaly.
	findThresholdDensity sets the binContainer variable which is used to output a threshold density histogram. Call this function almost immediately.
	Enter threshold Image you would like to find the bin density of
	Select either true or false for second paramter: False = Horizontal Binning. True = Veritical Binning */
	void findThresholdDensity(Mat& thresholdImage, bool flip = false)
	{
		Mat orchardThreshold;
		thresholdImage.copyTo(orchardThreshold);
		// 1) Declare internal variables
		int numBins = 200; // This divides the picture into slivers
		int binWidth = 0; // Number of pixels within a sliver. Will be set differently depending on flip being true of false
		int counter = 0; // generic counter
		double whitePixelBinCount = 0; // the amount of white pixels counted in a bin
		vector <double> temp; // all purpose temporary vector

		// 2) Depending on the users input, the switch case will either run true or false. 
		// False indicates the threshold image will be horizontally binned. True indicates that the threshold image will be vertically binned
		binContainer.clear(); // Wipe binContainer clean
		binContainer.reserve(200); // reserve 200 spots for the vector
		switch (flip)
		{
		case false: // If false is exectued, then the threshold picture will be horizontally binned
			binWidth = orchardThreshold.rows / numBins; // Number of pixels within a bin. Here it corresponds to horizontal slivers
			// 3) Loop through all rows. Once row reaches binWidth, store the number of pixels counted in binContainer.
			// binWidth is currently set to 15 (3000/200).. so every 15 rows the pixel count will be stored
			for (int _row = 0; _row < orchardThreshold.rows; _row++)
			{
				for (int _column = 0; _column < orchardThreshold.cols; _column++)
				{ // We need to add the white pixels to find a total amount of pixels 
					whitePixelBinCount += orchardThreshold.at<uchar>(_row, _column) / 255;
				}
				counter++;
				if (counter == binWidth)
				{
					counter = 0;
					binContainer.push_back(whitePixelBinCount); // store bin pixel count in binContainer
					whitePixelBinCount = 0;
				}
			}
			break;

		case true: // If true is exectued, then the threshold image will be vertically binned
			binWidth = orchardThreshold.cols / numBins; // Number of pixels within a sliver. Here it corresponds to vertical slivers
			for (int _column = 0; _column < orchardThreshold.cols; _column++)
			{
				for (int _row = 0; _row < orchardThreshold.rows; _row++)
				{ // We need to sum the white pixels to find a total amount of pixels per bin
					whitePixelBinCount += orchardThreshold.at<uchar>(_row, _column) / 255;
				}
				counter++;
				if (counter == binWidth)
				{
					counter = 0;
					binContainer.push_back(whitePixelBinCount); // store bin pixel count in binContainer
					whitePixelBinCount = 0;
				}
			}
			break;

		default:
			cout << "Something Didnt Work Right" << endl;
			break;
		}
		// 4) Using the newly filled binContainer, find the mean and standard deviation of the binData
		MeanAndStandardDeviation_BinContainer();
	}

	// Function executes when outputRowHistogram is called
	void MeanAndStandardDeviation_BinContainer()
	{
		double totalBinSum = 0;
		int numElements = binContainer.size();
		double temp = 0;
		double variance = 0;
		for (vector<int>::iterator itMean = binContainer.begin(); itMean != binContainer.end(); ++itMean)
		{
			totalBinSum += *itMean;
		}
		averageBinDensity = totalBinSum / numElements;

		for (vector<int>::iterator itStand = binContainer.begin(); itStand != binContainer.end(); ++itStand)
		{
			temp += pow(abs(averageBinDensity - *itStand), 2);
		}
		variance = temp / numElements;
		standard_deviation = sqrt(variance);
	}

	// Function exectutes when findBlossoms is called. Calculates both mean & std_dev and interquartile mean & interquartile std_dev
	void MeanAndStandardDeviation_Blossoms()
	{
		// Define Temporary Variables -- This 
		int _size = blossomCoordinateContainer.size(); // size of the container
		int quartile = _size / 4; // size of one quartile
		int tempSum = 0; // temporary sum

		// 1) Find the apple mean
		for (vector< pair<double, Point2i> >::iterator itA = blossomCoordinateContainer.begin(); itA != blossomCoordinateContainer.end(); ++itA)
		{
			tempSum += itA->first;
		}
		blossomMeanArea = tempSum / _size;
		blossomRadius = std::sqrt(blossomMeanArea / 3.14);
		// 2) Find apple standard deviation
		tempSum = 0;
		for (vector< pair<double, Point2i> >::iterator itA = blossomCoordinateContainer.begin(); itA != blossomCoordinateContainer.end(); ++itA)
		{
			tempSum += std::pow(std::abs(itA->first - blossomMeanArea), 2);
		}
		blossomStdDeviation = std::sqrt((tempSum / _size));

		// 3) Find interquartile apple mean 
		tempSum = 0;
		for (vector< pair<double, Point2i> >::iterator itA = blossomCoordinateContainer.begin() + quartile; itA < blossomCoordinateContainer.begin() + 3 * quartile; ++itA)
		{
			tempSum += itA->first;
		}
		interquartileBlossomMean = (tempSum / (_size / 2));
		interquartileBlossomRadius = std::sqrt(interquartileBlossomMean / 3.14);

		// 4) Find interquartile apple standard deviation
		tempSum = 0;
		for (vector< pair<double, Point2i> >::iterator itA = blossomCoordinateContainer.begin() + quartile; itA < blossomCoordinateContainer.begin() + 3 * quartile; ++itA)
		{
			tempSum += std::pow(std::abs(itA->first - interquartileBlossomMean), 2);
		}
		interquartileBlossomStdDeviation = std::sqrt((tempSum / (_size / 2)));
	}

	// Function counts the bin container to determine tree locations within a cropped image. Called from countTreesInRow
	void findTreeCoordinates(bool BlossomCanopyOrTreeStem)
	{ // This algorithm will find the trees or inside an image, and output their locations and coordinates on the BGR Image

		// Will fill one of two containers. The canopy container or the  
	
		// 1) Declare Variables for local use
		const int CONFIDENCE_LEVEL_TREES = 1; // If we have over ten bins in a row that are greater than the average, than most likely its a row
		const int CONFIDENCE_LEVEL_FIELD = 4;
		const double treeComparisonValue = averageBinDensity + standard_deviation; // compares bin values to 3 deviations higher than the mean bin-value   
		const double fieldComparisonValue = averageBinDensity; //- .25 * standard_deviation;
		const int xAxisIncrement = imageWidth / numBins; // If Width is 4000 pixels, then the increment should be (4000/200 = 20)
		const int treePadding = 100;
		int compareTreeCount = 0; // Check to see if bin values correspond to rows
		int compareFieldCount = 0; // Check to see if bin values correspond to field space 
		int xImageValue = imageWidth; // Width of picture
		int treeCounter = 0;
		bool treeStart = false;
		pair<string, array<int, 2> >* myTree = NULL;

		// 2) Start at the back of the bin container (right side of image), and go through and determine how many trees are in the image
		for (reverse_iterator<vector<int>::iterator> itb = binContainer.rbegin(); itb != binContainer.rend(); ++itb)
		{// starts at the end of the binContainer. The entries at the end will correspond to the first tree

			// If the bin value is above the rowComparisonValue, and we are between rows, then we need to increment the compareRowCount variable 
			if ((*itb > treeComparisonValue) && (treeStart == false))
			{
				++compareTreeCount;
			}

			// If the bin value is not above the rowComparisonValue, and we are between rows, then we need to reset compareRowCount
			if ((*itb < treeComparisonValue) && (treeStart == false))
			{
				compareTreeCount = 0;
			}

			// If the bin value is above the rowComparisonValue, and we are on a row, then we do not need to do anything
			if ((*itb > fieldComparisonValue) && (treeStart == true))
			{
				compareFieldCount = 0;
			}

			// If the bin value is below the fieldComparisonValue, and a row has been found, then we need to increment compareFieldCounter
			if ((*itb < fieldComparisonValue) && (treeStart == true))
			{
				++compareFieldCount;
			}

			// If the compareRowCount is at confidence_level.. then we are going to store the tree end coordinate and turn rowStart on
			if (compareTreeCount == CONFIDENCE_LEVEL_TREES) // Then we have found a row. Let's store the y axis location 
			{
				treeStart = true;
				myTree = new pair<string, array<int, 2> >; // dynamically create another pair object that myRow will point at
				myTree->first = "Row " + to_string(++treeCounter);
				myTree->second[0] = NULL; // placeholder value
				myTree->second[1] = xImageValue + 5 * xAxisIncrement; // Bottom Row value. Since we checked to find 5 confidence markers, backtrack 5 spots to find the start
				compareTreeCount = NULL; // Now that we found a row, reset compareRowCount
			}

			// If the compareFieldCount is at confidence_level.. then we are going to store the row begin coordinate and turn rowStart off
			if (compareFieldCount == CONFIDENCE_LEVEL_FIELD) // Then we have found a row. Let's store the y axis location 
			{
				treeStart = false;
				myTree->second[0] = xImageValue + 5 * xAxisIncrement; // Top Row value. Since we checked to find 5 confidence markers, backtrack 5 spots to find the end position of the tree
				// if user selects true, then blossom treeCanopyCoordinateContainer will be filled
				if (BlossomCanopyOrTreeStem)
				{
					treeCanopyCoordinateContainer.push_back(*myTree);

				}
				// if user selects false, then the treeTrunkCoordinateContainer will be filled
				else
				{
					treeTrunkCoordinateContainer.push_back(*myTree);
				}
				compareFieldCount = 0; // Now that we found the end of the row, lets set the compareFieldCount to zero to restart process
				myTree = NULL; // Now that we are done with the pointer, lets set it to NULL
			}

			xImageValue -= xAxisIncrement;
		}

	}

	// We need to use the row container information to set the BGRImage_First_Row Matrix 
	// Function crops The BGRImage To Only View Objects in First Row. Sets BGRImage_First_Row - May need higher flight altitude
	void cropBGRtoFirstRow()
	{
		int upperLeftYCoord = rowCoordinateContainer[1].second[1]; // Crop image from the start of the second row to the bottom of the first row
		int upperRightXCoord = 0; // Tiny bit of buffer room
		int buffer = 100; // To make sure no tree is getting cuttoff.
		int height = rowCoordinateContainer[0].second[1] - rowCoordinateContainer[1].second[1] - buffer;
		int width = 3999;
		Mat ROI(BGRImage, Rect(upperRightXCoord, upperLeftYCoord - buffer, width, height));
		ROI.copyTo(BGRImage_FirstRow);
	}

public:
	// Image Variables 
	Mat BGRImage; // The read-in BGR Orchard Image that initialized the object itself
	Mat BGRImage_FirstRow; // The BGR image cropped to the first row specifications
	// Threshold Images of the Rows and the Apples found using the inner HSV_Values classes
	Mat thresholdImg_Blossoms; // Threshold Image of the apples on the trees 
	Mat thresholdImg_Blossoms_FirstRow; // Threshold Image of the blossoms in first row
	Mat thresholdImg_Trees_FirstRow; // Threshold Image of potential tree locations in first row
	Mat thresholdImg_Rows; // Threshold Image of the rows to identify trees
	int blossomCount = 0;

	// Very Important Coordinate Containers. Holds the Row, blossom, and tree information.
	vector<pair<string, array<int, 2> > > rowCoordinateContainer; // Holds the row information per picture. Format: "Row X", [Lower Cord, Upper Cord]
	vector<pair<double, Point2i> > blossomCoordinateContainer; // Holds the blossom information per picture. Format: "<Area,(xCord,yCord)>"
	vector<pair<string, array<int, 2> > > treeCanopyCoordinateContainer; // Holds the tree information per picture. Iterates Right to Left. Format: "Row X", [Left Cord, Right Cord]
	vector<pair<string, array<int, 2> > > treeTrunkCoordinateContainer; // Holds the tree information per picture. Iterates Right to Left. Format: "Row X", [Left Cord, Right Cord]


	/*
	Defining Constructor Variables:
	inputImageFileName = The blank image you wish to construct your graph on
	thresholdImg = The image which you would like to find the density of white pixels for
	outputImageFileName = The output file you would like to write the histogram to
	HSV_Low, HSV_High = If you supply a RGB Img, you must also supply HSV inRange values
	numBins = assumed to be 200, change to more or less if you wish
	*/

	//_________________________________Constructor________________________//
	// You can setup your object with the internal graph that is provided
	BlossomOrchard(Mat& _BGRImg)
	{
		GetSetObjectProperties(_BGRImg);
	}

	//_________________________________Public Member Functions________________________//

	// To output the row density histogram, give the function the address of what folder to write the image to. OUTPUT ONLY SETS NOTHING
	void outputRowHistogram(string outputFilePath)
	{
		// Use Histogram Object to Find the Histogram
		Histogram* rowHistogram = new Histogram(thresholdImg_Rows, outputFilePath);
		rowHistogram->drawHistogram(false); // False indicates a horizontal binning

	}

	// This algorithm will find the rows inside a picture, and output their locations and coordinates on the objects Read-In BGR Orchard Image
	// This algorithm will use the binContainer to determine where the rows in the Orchard Image are located
	// Once the row locations are found, the algorithm will take the row information and crop the BGR image to only the first row
	void findRowLocations()
	{
		// 1) Fill the binContainer with the blossom row data 
		findThresholdDensity(thresholdImg_Rows, false);

		// 2) Define Variables for determining Blossom Orchard rows
		const int CONFIDENCE_LEVEL = 5; // If we have over ten bins in a row that are greater than the average, than most likely its a row
		const int BELOW_AVERAGE_LEVEL = 5; // We need to see if the our bin count 
		const double rowComparisonValue = averageBinDensity + standard_deviation; // compares bin values to 3 deviations higher than the mean bin-value   
		const double fieldComparisonValue = averageBinDensity - standard_deviation;
		const int yAxisIncrement = imageHeight / numBins; // If Height is 3000 pixels, then the increment should be (3000/200 = 15)
		const int rowPadding = 100;
		int compareRowCount = 0; // Check to see if bin values correspond to rows
		int compareFieldCount = 0; // Check to see if bin values correspond to field space 
		int yImageValue = imageHeight; // height of picture
		int rowCounter = 0;
		bool rowStart = false;
		pair<string, array<int, 2> >* myRow = NULL;

		// 3) Iterate through the bin container to determine where the rows in the Orchard Image are located
		for (reverse_iterator<vector<int>::iterator> itb = binContainer.rbegin(); itb != binContainer.rend(); ++itb)
		{// starts at the end of the binContainer. The entries at the end will correspond to the first row

			// If the bin value is above the rowComparisonValue, and we are between rows, then we need to increment the compareRowCount variable 
			if ((*itb > averageBinDensity) && (rowStart == false))
			{
				++compareRowCount;
			}

			// If the bin value is not above the rowComparisonValue, and we are between rows, then we need to reset compareRowCount
			if ((*itb < averageBinDensity) && (rowStart == false))
			{
				compareRowCount = 0;
			}

			// If the bin value is above the rowComparisonValue, and we are on a row, then we do not need to do anything
			if ((*itb > averageBinDensity) && (rowStart == true))
			{
				compareFieldCount = 0;
			}

			// If the bin value is below the fieldComparisonValue, and a row has been found, then we need to increment compareFieldCounter
			if ((*itb < averageBinDensity) && (rowStart == true))
			{
				++compareFieldCount;
			}

			// If the compareRowCount is at confidence_level.. then we are going to store the row end coordinate and turn rowStart on
			if (compareRowCount == CONFIDENCE_LEVEL) // Then we have found a row. Let's store the y axis location 
			{
				rowStart = true;
				myRow = new pair<string, array<int, 2> >; // dynamically create another pair object that myRow will point at
				myRow->first = "Row " + to_string(++rowCounter);
				myRow->second[0] = NULL; // placeholder value
				myRow->second[1] = yImageValue + 5 * yAxisIncrement; // Bottom Row value. Since we checked to find 5 confidence markers, backtrack 5 spots to find the start
				compareRowCount = NULL; // Now that we found a row, reset compareRowCount
			}

			// If the compareFieldCount is at confidence_level.. then we are going to store the row begin coordinate and turn rowStart off
			if (compareFieldCount == CONFIDENCE_LEVEL) // Then we have found a row. Let's store the y axis location 
			{
				rowStart = false;
				myRow->second[0] = yImageValue + 5 * yAxisIncrement; // Top Row value. Since we checked to find 5 confidence markers, backtrack 5 spots to find the start
				rowCoordinateContainer.push_back(*myRow);
				compareFieldCount = NULL; // Now that we found the end of the row, lets set the compareFieldCount to zero to restart process
				myRow = NULL; // Now that we are done with the pointer, lets set it to NULL
			}

			yImageValue -= yAxisIncrement;
		}
		// 4) Using the rowCoordinateContainer information, crop the BGR Orchard Image to show only first row, and save it in BGRImg_First_Row Matrix
		cropBGRtoFirstRow();
	}

	// This Algorithm is simply used to output the row coordinates on the BGRImage. This is a great way to visualize how accurate findRowLocations worked
	void visualizeRowCoordinates(string outputFilePath)
	{
		// Draw rectangles around the rows found
		for (vector<pair<string, array<int, 2> > >::iterator itRow = rowCoordinateContainer.begin(); itRow != rowCoordinateContainer.end(); ++itRow)
		{
			// Draws a rectangle around each and every row identified. 
			rectangle(BGRImage, Point(10, itRow->second[0]), Point(BGRImage.cols - 10, itRow->second[1]), Scalar(0, 0, 255), 3);
			putText(BGRImage, itRow->first, Point(BGRImage.cols / 2, (itRow->second[0] + itRow->second[1]) / 2), fontFace, fontScale, Scalar(255, 0, 255), thickness);
		}

		// Output the file to the correct folder
		Mat _BGRImage;
		BGRImage.copyTo(_BGRImage);
		imwrite(outputFilePath, _BGRImage);
	}

	// To output the tree histogram, give give the function the address of what folder to write the image to
	void outputTreeHistogram(string FilePath)
	{
		// Create a Histogram object to visually see the bin container
		Histogram* treeHistogram = new Histogram(thresholdImg_Blossoms_FirstRow, FilePath);
		treeHistogram->drawHistogram(true);
	}

	// Call this Algorithm after you have identified the first row and wish to fill the treeCanopyCoordinateContainer
	void findTreesInFirstRow()
	{
		Mat tempFirstRowThreshold;
		HSV_Values_TreeCanopy Trees_HSV;
		// 1) Change BGRImage_FirstRow to thresholdImg_Blossoms_FirstRow so we may conduct HSV operations on the first row image
		tempFirstRowThreshold = convertRGB2HSV(BGRImage_FirstRow, &Trees_HSV);
		tempFirstRowThreshold.copyTo(thresholdImg_Trees_FirstRow);

		// 2) We need to reset the bincontainer. We need findThresholdDensity to iterate vertically over the cropped first row photo
		findThresholdDensity(thresholdImg_Trees_FirstRow, true);

		// 3) Now that the binContainer is filled with blossom values, lets run findTreeCoordinates to fill the treeCanopyCoordinateContainer
		findTreeCoordinates(true);
	}

	// Call this algorithms to the individual trees within the first row
	void findTreesAreaMethod(string FilePath)
	{
		// 1) Declare temp Matrix and HSV_Values for the tree operation
		Mat tempFirstRowThreshold;
		//HSV_Values_TreeStems Trees_HSV;
		HSV_Values_TreeCanopy Trees_HSV;

		// 2) Change BGRImage_FirstRow to thresholdImg_Blossoms_FirstRow so we may conduct HSV operations on the first row image
		tempFirstRowThreshold = convertRGB2HSV(BGRImage_FirstRow, &Trees_HSV);
		tempFirstRowThreshold.copyTo(thresholdImg_Trees_FirstRow);
		imwrite("Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\60BLOSSOMTHRESHOLD.jpg", thresholdImg_Trees_FirstRow);
		
		// 3) Use Moments method to find and collect areas of all threshold objects in thresholdImg_Blossoms_FirstRow
		// We will need to use the moments method to find all of the objects in the cropped image
		vector <vector <Point> >contours; // holds the contours found for the apples
		vector<Vec4i> hierarchy; // holds the hierarchial information for contours 
		Moments moment;          // moments are used to calculate area values and x-y coordinates
		double objectArea;	     // Used to store the object area 
		Point2i objectCenter;    // Used to store the x,y coordinates of the objects
		pair<double, Point2i> objectGeographicalInfo; // holds the pair of data
		pair < pair< double, Point2i >,  vector < Point> > storageInfo;
		vector <pair < pair< double, Point2i >,vector < Point> > > objectInformationContainer; // Looks Cray, its not. Vector holding a pair of -> [pair of (Object Area, CenterPoint), and vector of (contours)]
		vector <pair < pair< double, Point2i >, vector < Point> > > treeAreasAndCoordinates; // Main coordinate container which holds tree canopy locations 
		
		// Sorting predicates
		struct treeAreasAndCoordinates_sorting_predicate_double_descending {
			bool operator()(const pair < pair< double, Point2i >, vector < Point> > &left, const pair < pair< double, Point2i >, vector < Point> > &right)
			{
				return left.first.first > right.first.first; // first value is the largest
			}
		};
		struct treeAreasAndCoordinates_sorting_predicate_Point2i_XCord_descending {
			bool operator()(const pair < pair< double, Point2i >, vector < Point> > &left, const pair < pair< double, Point2i >, vector < Point> > &right)
			{
				return left.first.second.x > right.first.second.x; // first value is the largest
			}
		};
		struct treeAreasAndCoordinates_sorting_predicate_Point2i_XCord_ascending {
			bool operator()(const pair < pair< double, Point2i >, vector < Point> > &left, const pair < pair< double, Point2i >, vector < Point> > &right)
			{
				return left.first.second.x < right.first.second.x; // first value is the largest
			}
		};
		struct treeAreasAndCoordinates_sorting_predicate_XCord_Contours {
			bool operator()(const Point2i &left, const Point2i &right)
			{
				return left.x > right.x; // first value is the largest
			}
		};

		// 4) Find object contours
		findContours(thresholdImg_Trees_FirstRow, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		// 5) Find all object sizes and locations
		for (vector< vector<Point> >::iterator itC = contours.begin(); itC != contours.end(); ++itC)
		{
			moment = moments((Mat)*itC);
			objectArea = moment.m00; // holds object area
			objectCenter = Point2i(moment.m10 / objectArea, moment.m01 / objectArea); // holds centerpoint
			objectGeographicalInfo = pair<double, Point2i>(objectArea, objectCenter); // holds the pair of data
			storageInfo = pair < pair< double, Point2i >, vector < Point> >(objectGeographicalInfo, *itC);
			objectInformationContainer.push_back(storageInfo); // This now holds the object information and the contours of the object
		}
		
		// 6) Sort all objects in object size descending order 
		sort(objectInformationContainer.begin(), objectInformationContainer.end(), treeAreasAndCoordinates_sorting_predicate_double_descending());
		
		// 7) Now find all objects that are larger than 8000 between 250 down and 200 from bottom -- MAY NEED TO FIDDLE WITH VALUES 
		
		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itT = objectInformationContainer.begin(); itT != objectInformationContainer.end(); ++itT)
		{
			if (itT->first.first > 8000 && itT->first.second.y > 250 && itT->first.second.y < (BGRImage_FirstRow.rows - 200))
			{
				treeAreasAndCoordinates.push_back(*itT);
			}
		}
		// 8) Sort all objects in xCoordinate size order descending 
		sort(treeAreasAndCoordinates.begin(), treeAreasAndCoordinates.end(), treeAreasAndCoordinates_sorting_predicate_Point2i_XCord_descending());

		// 9) Sort all the x-contour coordinates from smallest to largest within the treeAreasAndCoordinates container
		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itT = treeAreasAndCoordinates.begin(); itT != treeAreasAndCoordinates.end(); ++itT)
		{
			sort(itT->second.begin(), itT->second.end(), treeAreasAndCoordinates_sorting_predicate_XCord_Contours());
		}
		
		// 10) We need to find any objects that are overlapping. If an object has an area that crosses over another object, keep the bigger objects area, and remove the smaller from the vector
		// Delete any contours that arent the extremes, these are now unneeded
		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itT = treeAreasAndCoordinates.begin(); itT != treeAreasAndCoordinates.end(); ++itT)
		{
			itT->second.erase(itT->second.begin() + 1, itT->second.end() - 2);
		}

		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itT = treeAreasAndCoordinates.begin()+1; itT != treeAreasAndCoordinates.end(); ++itT)
		{
			// Check to See if you are at the end of the vector.. If you are then we cant compare anything else
			if ( (itT+1) != treeAreasAndCoordinates.end() )
			{
				int itHR = (itT - 1)->second.begin()->x; // adjacent right object high x coordinate
				int itLR = ((itT - 1)->second.end() - 1)->x; // adjacent right object low x coordinate
				int itHC = itT->second.begin()->x; // current object high x coordinate
				int itLC = (itT->second.end()-1)->x; // current object low x coordinate
				int itHL = (itT + 1)->second.begin()->x; // adjacent left object high x coordinate
				
				 // Case 1: Double intersection: Both objects clash with middle object. Assign left block high rights coords, assign pseudo chords to center and right to prevent further intersection
				 if ( (itHC > itLR) && (itHL > itLC) )
				 {
					 // Assigning left block high rights coords
					 vector<Point>::const_iterator itFirst = (itT + 1)->second.begin();
					 (itT + 1)->second.insert(itFirst, Point2i(itHR,0));

					 //Assigning pseudo coords for center and right to prevent further intersection
					 itT->second.begin()->x = 5000; // HC
					 itT->second.begin()->y = 5000; // HC
					 (itT->second.end()-1)->x = 5000; // LC
					 (itT->second.end()-1)->y = 5000; // LC

					 (itT-1)->second.begin()->x = 5000; // HR
					 (itT-1)->second.begin()->y = 5000; // HR
					 ((itT-1)->second.end() - 1)->x = 5000; // LR
					 ((itT-1)->second.end() - 1)->y = 5000; // LR

				 }

				 // Case 2: Center object intersects with right and does not intersect with left
				 if ( (itHC > itLR) && (itLC > itHL) )
				 {
					 // Assign center block High Rights coords
					 vector<Point>::iterator it_First = itT->second.begin();	 
					 itT->second.insert(it_First, Point2i(itHR,0) );

					 // Assign right block pseudo cords
					 (itT - 1)->second.begin()->x = 5000; // HR
					 (itT - 1)->second.begin()->y = 5000; // HR
					 ((itT - 1)->second.end() - 1)->x = 5000; // LR
					 ((itT - 1)->second.end() - 1)->y = 5000; // LR
				 }

				 // Case 3: If left and center blocks are intersecting but right block does not
				 if ( (itHC < itLR)  && (itHL > itLC) )
				 {
					 // Assign left block the coords of high center
					 vector<Point>::iterator itFirst = (itT+1)->second.begin();
					 (itT+1)->second.insert(itFirst, Point2i(itHC,0));

					 // Assign pseudo cords for center 
					 itT->second.begin()->x = 5000; // HC
					 itT->second.begin()->y = 5000; // HC
					 (itT->second.end() - 1)->x = 5000; // LC
					 (itT->second.end() - 1)->y = 5000; // LC

				 }

				 // Case 4: If there is no intersection, then do nothing
				 if ( (itLR > itHC) && (itHC > itHL) )
				 {
					 __nop();
				 }

			}

		}

		// 11) Remove all unneeded content within the treeAreasandCoordinatesContainer
		vector <pair < pair< double, Point2i >, vector < Point> > > eraseObjects; // vector of iterators that point to positions inside the container to remove  
		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itF = treeAreasAndCoordinates.begin(); itF != treeAreasAndCoordinates.end(); ++itF)
		{
			if (itF->second.begin()->x == 5000)
			{
				eraseObjects.push_back(*itF);
			}
		}
		// 12) Use erase-remove idiom to find and remove all of the objects within the container that are undesirable
		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itF = eraseObjects.begin(); itF != eraseObjects.end(); ++itF)
		{
			treeAreasAndCoordinates.erase(remove(treeAreasAndCoordinates.begin(), treeAreasAndCoordinates.end(), *itF), treeAreasAndCoordinates.end());
		}

        // 13) Find mean and standard deviation
		int meanWidth = 0;
		int sum = 0;
		int std_dev_width = 0;
		int variance = 0;
		vector<int>treeWidth;
		// finding mean
		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itT = treeAreasAndCoordinates.begin(); itT != treeAreasAndCoordinates.end(); ++itT)
		{
			int width = (itT->second.begin()->x) - (itT->second.end() - 1)->x;
			sum += width;
			treeWidth.push_back(width);
		}
		meanWidth = sum / treeWidth.size();
		// finding std dev
		for (vector<int>::iterator itC = treeWidth.begin(); itC != treeWidth.end(); ++itC)
		{
			variance = abs(pow((*itC - meanWidth), 2));
		}
		std_dev_width = sqrt((variance / treeWidth.size()));

// This Section below needs work!!!

		// 10) Find reported tree areas that are too small and merge them with their larger neighbors
		vector <pair < pair< double, Point2i >, vector < Point> > > eraseVec; // vector of iterators that point to positions inside the container to remove  
		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itT = treeAreasAndCoordinates.begin(); itT != treeAreasAndCoordinates.end(); ++itT)
		{
			int currentWidth = 0;

			int leftNeighborWidth = 0; // how large the tree slice to the left of the current slice is
			int rightNeighborWidth = 0; // how large the tree slice to the right of the current slice is

			int leftMargin = 0; // how close the current tree slice is to the adjacent left tree slice 
			int rightMargin = 0;  // how close the current tree slice is to the adjacent right tree slice

			const int LEFT_MAX_MARGIN = 80;
			const int RIGHT_MAX_MARGIN = 80;
			
			// Case 1: End condition, No right neighboor
			if (itT == treeAreasAndCoordinates.begin())
			{
				 currentWidth = abs((itT->second.begin()->x) - (itT->second.end() - 1)->x);

				 leftNeighborWidth = abs(((itT + 1)->second.begin()->x) - ((itT + 1)->second.end() - 1)->x); // how large the tree slice to the left of the current slice is
				 leftMargin = abs((itT->second.end() - 1)->x - (itT + 1)->second.begin()->x); // how close the current tree slice is to the adjacent left tree slice 
				
				// Check to see which is the larger neighboor. This will be the one that absorbs the xCordinate of the current tree canopy slice
				int xCoordinate = 0;

				// If the left margins is greater than the max and the tree is not greater than a standard dev above mean, we know we have a stand alone tree... this means we should expand its width
				if ((leftMargin > LEFT_MAX_MARGIN) && (currentWidth < (meanWidth + std_dev_width)) )
				{
					if (((itT->second.end() - 1)->x - (itT + 1)->second.begin()->x) > meanWidth) // if the difference between the two is greater than mean width, than expand left by half of mean width
					{
						(itT->second.end() - 1)->x -= (double)(.5 * meanWidth);
					}
					else
					{
						(itT->second.end() - 1)->x = (itT + 1)->second.begin()->x + 1.1*LEFT_MAX_MARGIN; // 1.1 to give it a little buffer
					}
					// We now give the current data object the dont mess with me 5000 y - value
					itT->second.begin()->y = 5000; // Key that this object was just adjusted. All right side operations next iteration should not consider this object
				}
				

				// if the left margin is under max & the current width is smaller than mean - std_dev,  merge current object with left
				if ( (leftMargin < LEFT_MAX_MARGIN) && (currentWidth < (meanWidth - std_dev_width)) )
				{
					// Assign the rightmost coord to the next object in the container
					xCoordinate = itT->second.begin()->x;
					(itT + 1)->second.begin()->x = xCoordinate;

					// Assign pseudo values to the current itT so it does not interfere with any operations
					itT->second.begin()->x = 5000; // Psuedo Values
					itT->second.begin()->y = 5000; // Psuedo Values
					(itT->second.end()-1)->x = 5000; // Psuedo Values
					(itT->second.end()-1)->y = 5000; // Psuedo Values

					eraseVec.push_back(*itT); // store current iterator contents into the eraseVec to erase merged object
				}
			}


			// Case 2: End Condition, No left neighboor
			if ((itT + 1) == treeAreasAndCoordinates.end())
			{
				currentWidth = abs((itT->second.begin()->x) - (itT->second.end() - 1)->x);
				
				rightNeighborWidth = abs(((itT - 1)->second.begin()->x) - ((itT - 1)->second.end() - 1)->x); // how large the tree slice to the right of the current slice is
				rightMargin = abs(((itT - 1)->second.end() - 1)->x - itT->second.begin()->x);  // how close the current tree slice is to the adjacent right tree slice
				

				// If the width is smaller than a standard deviation under the mean, we need to merge the tree slice either left right, or make sure its a stand alone tree with incorrect width assigned 
				// Check to see which is the larger neighboor. This will be the one that absorbs the xCordinate of the current tree canopy slice
				int xCoordinate = 0;

				// If the right margins is greater than the max, smaller than a std_dev above the mean ,and the previous right coordinate was not just set to psuedo value 5000, we know we have a stand alone tree... this means we should expand its width on the right side
				if ((rightMargin > RIGHT_MAX_MARGIN) && (currentWidth < (meanWidth + std_dev_width) ) && (itT->second.begin()->y != 5000) ) 
				{
					if ( ( ((itT - 1)->second.end() - 1)->x - itT->second.begin()->x ) > meanWidth)
					{
						itT->second.begin()->x += (double)(.5*meanWidth);
					}
					else
					{
						itT->second.begin()->x = ((itT - 1)->second.end() - 1)->x - 1.1*RIGHT_MAX_MARGIN; // 1.1 for a tiny bit more buffer room than the max
					}
				}
				

				// If right margin is under max, the previous right coordinate was not just set to psuedo value 5000, and the current width is smaller than the mean - std_Dev, group right
				if ((rightMargin < RIGHT_MAX_MARGIN) && (itT->second.begin()->y != 5000) && (currentWidth < (meanWidth - std_dev_width)) && (currentWidth < (.5*rightNeighborWidth)) && (rightNeighborWidth < (meanWidth + std_dev_width) ))
				{
					xCoordinate = (itT->second.end()-1)->x; // Verdicts still out regarding whether this should be the leftmost coordinate or the end of the image itself to prevent cuttoff
					((itT - 1)->second.end()-1)->x = xCoordinate; // right tree slice gains end of xCoordinate
					eraseVec.push_back(*itT); // store current iterator contents into the eraseVec
				}

			}

			// Case 3: Both neighbors are defined
			if ( (itT != treeAreasAndCoordinates.begin()) && ((itT + 1) != treeAreasAndCoordinates.end()) )
			{
				currentWidth = abs((itT->second.begin()->x) - (itT->second.end() - 1)->x);

				leftNeighborWidth = abs(((itT + 1)->second.begin()->x) - ((itT + 1)->second.end() - 1)->x); // how large the tree slice to the left of the current slice is
				rightNeighborWidth = abs(((itT - 1)->second.begin()->x) - ((itT - 1)->second.end() - 1)->x); // how large the tree slice to the right of the current slice is

				leftMargin = abs((itT->second.end() - 1)->x - (itT + 1)->second.begin()->x); // how close the current tree slice is to the adjacent left tree slice 
				rightMargin = abs(((itT - 1)->second.end() - 1)->x - itT->second.begin()->x);  // how close the current tree slice is to the adjacent right tree slice
			
				// If the width is smaller than a standard deviation under the mean, we need to merge the tree slice either left right, or make sure its a stand alone tree with incorrect width assigned 
				// Check to see which is the larger neighboor. This will be the one that absorbs the xCordinate of the current tree canopy slice
				int xCoordinate = 0;

				// If both the left and right margins are greater than the max, AND the previous right coordinate was not just set to psuedo value 5000 , we know we have a stand alone tree... this means we should expand the current trees width its width
				if ((leftMargin > LEFT_MAX_MARGIN) && (rightMargin > RIGHT_MAX_MARGIN) && ((itT-1)->second.begin()->y != 5000) ) // then the current tree slice will get absorbed into the leftneighbor
				{
					// Find the middle between the left and right objects
					const int middleOfLeftMargin = ((itT->second.end() - 1)->x + (itT + 1)->second.begin()->x) / 2;
					const int middleofRightMargin = (itT->second.begin()->x + ((itT - 1)->second.end() - 1)->x) / 2;
					const int buffer = (double)(.1*LEFT_MAX_MARGIN);
					const int leftBackTrack = (double)(.75*LEFT_MAX_MARGIN);
					const int rightBackTrack = (double)(.75*RIGHT_MAX_MARGIN);
					const int centerBacktrack = (double)(.25*LEFT_MAX_MARGIN);
					
					// Have left and right backtrack 75 + buffer away, and middle 25 + buffer
					(itT + 1)->second.begin()->x = middleOfLeftMargin - leftBackTrack - buffer;
					((itT - 1)->second.end()-1)->x = middleofRightMargin + rightBackTrack + buffer;
					itT->second.begin()->x = middleofRightMargin - centerBacktrack - buffer;
					(itT->second.end() - 1)->x = middleOfLeftMargin + centerBacktrack + buffer;
				
				}

				// If the current width is less than a standard deviation below the mean, then we know we have to take some action since its definately not a whole tree
				if (currentWidth < (meanWidth - std_dev_width))
				{
					// If the left margin is twice over max but less than a mean width, left neighbor is not collassal (2 std's over mean) and the right margin is under max, center gains right, prepare right for removal
					if ((leftMargin >(1.5 * LEFT_MAX_MARGIN)) && (leftMargin < meanWidth) && (leftNeighborWidth < (meanWidth + 1.5 * std_dev_width)) && (rightMargin < RIGHT_MAX_MARGIN))
					{
						// Center gains right coordinate
						xCoordinate = (itT-1)->second.begin()->x;
						itT->second.begin()->x = xCoordinate;    // Center Tree slice gains the x-coordinate of the right member
						
						// Prepare right for removal
						(itT - 1)->second.begin()->x = 5000; // HR
						(itT - 1)->second.begin()->y = 5000; // HR
						((itT - 1)->second.end() - 1)->x = 5000; // LR
						((itT - 1)->second.end() - 1)->y = 5000; // LR

						eraseVec.push_back(*(itT-1)); // store right object into eraseVec for removal
					}

					// if  left margin is under max, the right margin is above max but less than a mean tree length over,the right tree is not over two standard deviations greater then mean , AND the previous right coordinate was not just set to psuedo value 5000, then we will group left with center
					if ((leftMargin < LEFT_MAX_MARGIN) && (rightMargin >(1.5 * RIGHT_MAX_MARGIN)) && (rightMargin < meanWidth) && (rightNeighborWidth < (meanWidth + 1.5*std_dev_width) ) && ((itT - 1)->second.begin()->y != 5000))
					{
						// Give left centers highest x-coordinate
						xCoordinate = itT->second.begin()->x;
						(itT + 1)->second.begin()->x = xCoordinate;
						
						// Prepare center for removal
						itT->second.begin()->x = 5000; // HC
						itT->second.begin()->y = 5000; // HC
						(itT->second.end() - 1)->x = 5000; // LC
						(itT->second.end() - 1)->y = 5000; // LC
						eraseVec.push_back(*itT); // store current iterator contents into the eraseVec
					}

					// If both margins are under max, and the width of the current tree is half the size of the surrounding trees, if the left neighbor width is larger than the right neighborwidth,  group left
					if ((leftMargin < LEFT_MAX_MARGIN) && (rightMargin < RIGHT_MAX_MARGIN) && (currentWidth < (.5*leftNeighborWidth)) && (currentWidth < (.5*rightNeighborWidth)) && (leftNeighborWidth > rightNeighborWidth))
					{
						// Give left object centers high x values
						xCoordinate = itT->second.begin()->x;
						(itT + 1)->second.begin()->x = xCoordinate; // left tree slice gains xCoordinate
						
						// Prepare center for removal
						itT->second.begin()->x = 5000; // HC
						itT->second.begin()->y = 5000; // HC
						(itT->second.end() - 1)->x = 5000; // LC
						(itT->second.end() - 1)->y = 5000; // LC
						eraseVec.push_back(*itT); // store current iterator contents into the eraseVec
					}

					// If both margins are under max, then if the right neighbor width is larger than the left neighborwidth,  group right with center
					if ((leftMargin < LEFT_MAX_MARGIN) && (rightMargin < RIGHT_MAX_MARGIN) && (leftNeighborWidth < rightNeighborWidth))
					{
						// To prevent further collision or a growth pattern, give right centers lower x coordinate, and prepare center for removal
						xCoordinate = (itT->second.end() - 1)->x;
						((itT - 1)->second.end() - 1)->x = xCoordinate;    // Right Tree slice gains the x-coordinate 
						
						// Prepare center for removal
						itT->second.begin()->x = 5000; // HC
						itT->second.begin()->y = 5000; // HC
						(itT->second.end() - 1)->x = 5000; // LC
						(itT->second.end() - 1)->y = 5000; // LC
						eraseVec.push_back(*itT); // store center into the eraseVec
					}
				}
			}	
		}

		// 11) Delete the tree areas members that were found to be too small
		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itT = eraseVec.begin(); itT != eraseVec.end(); ++itT)
		{
			treeAreasAndCoordinates.erase(remove(treeAreasAndCoordinates.begin(), treeAreasAndCoordinates.end(),*itT), treeAreasAndCoordinates.end());
		}
		
		// 12) Clear the eraseVec so we dont have a memory leak or a dangling pointer
		eraseVec.clear();
		 

		// 13) Resort the data members to make sure all remaining object center coordinate are in descending order
		sort(treeAreasAndCoordinates.begin(), treeAreasAndCoordinates.end(), treeAreasAndCoordinates_sorting_predicate_Point2i_XCord_descending());
		
		// 10) Now that the xCordinates within each vector are sorted, lets go ahead and draw lines on the BGRCroppedImg;
		//Mat _BGRImage_FirstRow;
		//BGRImage_FirstRow.copyTo(_BGRImage_FirstRow);
		for (vector <pair < pair< double, Point2i >, vector < Point> > >::iterator itT = treeAreasAndCoordinates.begin(); itT != treeAreasAndCoordinates.end(); ++itT)
		{
			rectangle(BGRImage_FirstRow, Point(itT->second.begin()->x, 100), Point((itT->second.end() - 1)->x, BGRImage_FirstRow.rows - 50), Scalar(255, 0, 0), 3);
		}
		
		// 11) Now write the Image into BlossomOrchard Folder for safe keeping  
		imwrite(FilePath, BGRImage_FirstRow);
		
 	}
	
	void findTreesTrunkMethod(string FilePath1, string FilePath2)
	{
		// 1) Declare temp Matrix and HSV_Values for the tree operation
		Mat tempFirstRowThreshold;
		HSV_Values_TreeStems Trees_HSV;

		// 2) Change BGRImage_FirstRow to thresholdImg_Blossoms_FirstRow so we may conduct HSV operations on the first row image
		tempFirstRowThreshold = convertRGB2HSV(BGRImage_FirstRow, &Trees_HSV);
		tempFirstRowThreshold.copyTo(thresholdImg_Trees_FirstRow);
		imwrite("Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\60BLOSSOMTHRESHOLD.jpg", thresholdImg_Trees_FirstRow);

		// 3) Use Moments method to find and collect areas of all threshold objects in thresholdImg_Blossoms_FirstRow
		// We will need to use the moments method to find all of the objects in the cropped image
		vector <vector <Point> >contours; // holds the contours found for the apples
		vector<Vec4i> hierarchy; // holds the hierarchial information for contours 
		Moments moment;          // moments are used to calculate area values and x-y coordinates
		double objectArea;	     // Used to store the object area 
		Point2i objectCenter;    // Used to store the x,y coordinates of the objects
		pair<double, Point2i> objectGeographicalInfo; // holds the pair of data
		pair < pair< double, Point2i >, vector < Point> > storageInfo;
		vector <pair < pair< double, Point2i >, vector < Point> > > objectInformationContainer; // Looks Cray, its not. Vector holding a pair of -> [pair of (Object Area, CenterPoint), and vector of (contours)]
		vector <pair < pair< double, Point2i >, vector < Point> > >treeAreasAndCoordinates;

		// Sorting predicates
		struct treeAreasAndCoordinates_sorting_predicate_double_descending {
			bool operator()(const pair < pair< double, Point2i >, vector < Point> > &left, const pair < pair< double, Point2i >, vector < Point> > &right)
			{
				return left.first.first > right.first.first; // first value is the largest
			}
		};
		struct treeAreasAndCoordinates_sorting_predicate_Point2i_XCord_descending {
			bool operator()(const pair < pair< double, Point2i >, vector < Point> > &left, const pair < pair< double, Point2i >, vector < Point> > &right)
			{
				return left.first.second.x > right.first.second.x; // first value is the largest
			}
		};
		struct treeAreasAndCoordinates_sorting_predicate_XCord_Contours {
			bool operator()(const Point2i &left, const Point2i &right)
			{
				return left.x > right.x; // first value is the largest
			}
		};

		// 4) Find object contours
		findContours(thresholdImg_Trees_FirstRow, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		// 5) Find all object sizes and locations
		for (vector< vector<Point> >::iterator itC = contours.begin(); itC != contours.end(); ++itC)
		{
			moment = moments((Mat)*itC);
			objectArea = moment.m00; // holds object area
			objectCenter = Point2i(moment.m10 / objectArea, moment.m01 / objectArea); // holds centerpoint
			objectGeographicalInfo = pair<double, Point2i>(objectArea, objectCenter); // holds the pair of data
			storageInfo = pair < pair< double, Point2i >, vector < Point> >(objectGeographicalInfo, *itC);
			objectInformationContainer.push_back(storageInfo); // This now holds the object information and the contours of the object
		}

		// 6) Sort all objects in object size descending order 
		sort(objectInformationContainer.begin(), objectInformationContainer.end(), treeAreasAndCoordinates_sorting_predicate_double_descending());

		
		findThresholdDensity(tempFirstRowThreshold, true);
		Histogram* treeHistogram = new Histogram(tempFirstRowThreshold, FilePath1);
		treeHistogram->drawHistogram(true);
		
		bool fillTreeTrunkContainer = false;
		findTreeCoordinates(fillTreeTrunkContainer);
		visualizeTreeCoordinates(FilePath2, fillTreeTrunkContainer);
	}

	// This Algorithm is simply used to output the tree coordinates on the cropped BGRImage. This is a great way to visualize how accurate findTreeCoordinates worked
	// Enter true for tree canopy or false for tree trunk visualization...reason being is we have two seperate containers that need to be pulled from
	void visualizeTreeCoordinates(string outputFilePath, bool TreeCanopyorTreeTrunk)
	{
		const int buffer = 50;

		switch (TreeCanopyorTreeTrunk)
		{
		case true:  // canopy

			// Draw rectangles around the canopy found
			for (vector<pair<string, array<int, 2> > >::iterator itRow = treeCanopyCoordinateContainer.begin(); itRow != treeCanopyCoordinateContainer.end(); ++itRow)
			{
				// Draws a rectangle around each and every tree identified. 
				rectangle(BGRImage_FirstRow, Point(itRow->second[0], buffer), Point(itRow->second[1], BGRImage_FirstRow.rows - buffer), Scalar(0, 0, 255), 3);
				putText(BGRImage_FirstRow, itRow->first, Point(BGRImage.cols / 2, (itRow->second[0] + itRow->second[1]) / 2), fontFace, fontScale, Scalar(255, 0, 255), thickness);
			}


			break;
		case false: // trunk
			// Draw rectangles around the trunks found
			for (vector<pair<string, array<int, 2> > >::iterator itRow = treeTrunkCoordinateContainer.begin(); itRow != treeTrunkCoordinateContainer.end(); ++itRow)
			{
				// Draws a rectangle around each and every tree identified. 
				rectangle(BGRImage_FirstRow, Point(itRow->second[0], buffer), Point(itRow->second[1], BGRImage_FirstRow.rows - buffer), Scalar(0, 0, 255), 3);
				putText(BGRImage_FirstRow, itRow->first, Point(BGRImage.cols / 2, (itRow->second[0] + itRow->second[1]) / 2), fontFace, fontScale, Scalar(255, 0, 255), thickness);
			}

			break;
		}
		
		// Output the file to the correct folder
		imwrite(outputFilePath, BGRImage_FirstRow);
	}

	// Call this Algorithm after you have identified the individual rows and wish to count the individual blossoms inside of that row
	void countBlossomsInFirstRow(string FilePath)
	{
		// 1) We need to use the row container information to crop the BGR photo to only show the first row specifications
		cropBGRtoFirstRow();
		// 2)  Assume most objects in picture are apples, this will be corrected with more ImageLabTesting Filtering Routines
		HSV_Values_Blossoms Blossom_HSV_Values;
		thresholdImg_Blossoms_FirstRow = convertRGB2HSV(BGRImage_FirstRow, &Blossom_HSV_Values); // converts cropped BGR Image to a cropped threshold image

		// 3) We will need to use the moments method to find all of the objects in the cropped image
		vector <vector <Point> >contours; // holds the contours found for the apples
		vector<Vec4i> hierarchy; // holds the hierarchial information for contours 
		Moments moment;          // moments are used to calculate area values and x-y coordinates
		double objectArea;	     // Used to store the object area 
		Point2i objectCenter;    // Used to store the x,y coordinates of the objects
		pair<double, Point2i> storeObjectData; // holds the pair of data


		findContours(thresholdImg_Blossoms_FirstRow, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		// 4) Take the contours, find areas, and store the areas, and the coordinates of the objects into a "pair vector"
		int count = -1;
		for (vector < vector<Point> >::iterator itc = contours.begin(); itc != contours.end(); ++itc)
		{
			++count;
			moment = moments((Mat)*itc);
			// 5) If an object is greater than an agreed upon small blossom size, we will count it as a blossom
			if (moment.m00 > 20) // Then it is a contender to be a row
			{
				// 6) Store all the coordinates of the apples in a "pair" vector composed of their areas and their x,y coordinates in a Point
				objectArea = moment.m00; // holds object area
				objectCenter = Point2i(moment.m10 / objectArea, moment.m01 / objectArea); // holds centerpoint
				storeObjectData = pair<double, Point2i>(objectArea, objectCenter); // holds the pair of data
				blossomCoordinateContainer.push_back(storeObjectData); // push data set into our container
				blossomCount++;
			}
		}

		// 7) Sort the data from greatest to smallest. End goal is to find the mean, median and mode
		sort(blossomCoordinateContainer.begin(), blossomCoordinateContainer.end(), sort_pred_double_descending());
		blossomCount = blossomCoordinateContainer.size();
		// 8) Find the interquartile mean and use this information to determine an average circle radius
		MeanAndStandardDeviation_Blossoms();

		// 9) Draw circles around each of the found apples, and output the total count, and mean size at the top of the Image
		for (vector< pair<double, Point2i> >::iterator itA = blossomCoordinateContainer.begin(); itA != blossomCoordinateContainer.end(); ++itA)
		{
			circle(BGRImage_FirstRow, itA->second, interquartileBlossomRadius, Scalar(0, 0, 255), 2);
		}
		// 10) Write image to a File to view
		string writeText = "Blossoms Found: " + to_string(blossomCount);
		putText(BGRImage_FirstRow, writeText, Point(50, 50), fontFace, fontScale, Scalar(255, 0, 255), thickness);
		imwrite(FilePath, BGRImage_FirstRow);
	}

	//----------------------------Utility Algorithms ----------------------------//

	// Internal Member Function to write an image to a provided FilePath
	void writeImageToFile(string FilePath, Mat ImageToWrite)
	{
		imwrite(FilePath, ImageToWrite);
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

	// Simple Blur to take away some of the edge in objects
	Mat blurImage(int ORCHARD_BLUR, Mat sourceImg)
	{
		// Simply blur the image to different values, which will be used to determine if the blurred filter Image will 
		// lead to better image recognition
		Mat blurredImage;
		medianBlur(sourceImg, blurredImage, ORCHARD_BLUR);
		return blurredImage;
	}

	// since all HSV_Value classes inheret from the innerBaseClass, this is the best way to make sure all operations can convert using the method below
	Mat convertRGB2HSV(Mat BGRImg, innerBaseClass* ORCHARD)
	{
		// HSV Values for Morphological Operations
		ORCHARD->set();
		Mat Threshold, HSVImg, Threshold_Temp;
		cvtColor(BGRImg, HSVImg, CV_BGR2HSV);
		inRange(HSVImg, Scalar(ORCHARD->H_Min, ORCHARD->S_Min, ORCHARD->V_Min), Scalar(ORCHARD->H_Max, ORCHARD->S_Max, ORCHARD->V_Max), Threshold_Temp);
		/*Threshold_Temp = dilateImage(ORCHARD->HSV_DILATE, Threshold_Temp); // dilate
		Threshold_Temp = erodeImage(ORCHARD->HSV_ERODE, Threshold_Temp);   // erode
		Threshold = blurImage(ORCHARD->HSV_BLUR, Threshold_Temp); // blur
		*/
		Threshold = dilateImage(ORCHARD->HSV_DILATE, Threshold_Temp); // dilate
		Threshold_Temp = erodeImage(ORCHARD->HSV_ERODE, Threshold);   // erode
		Threshold = blurImage(ORCHARD->HSV_BLUR, Threshold_Temp); // blur
		return Threshold;
	}




};

// Function Prototypes
void BatchProcessAppleOrchard(const int firstImgNum, const int lastImgNum);
void BatchProcessBlossomOrchard(const int firstImgNum, const int lastImgNum);

int main(void)
{
	//Batch Processing Images
	BatchProcessBlossomOrchard(70, 90);
	//BatchProcessAppleOrchard(20, 99);
}
 

void BatchProcessAppleOrchard(const int firstImgNum, const int lastImgNum)
{
	// Variable Declarations
	Mat BGRImg;
	vector< AppleOrchard*> OrchardImages;

	for (size_t i = firstImgNum; i < lastImgNum + 1; i++)
	{
		if (i < 10)
		{

			BGRImg = imread("Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\UnprocessedImages\\160901 crop load\\DJI_000" + to_string(i) + ".jpg");

			// Output File Paths
			string rowHistoFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\AppleOrchard\\" + to_string(i) + "AppleOrchard_RowHisto.jpg";
			string rowsFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\AppleOrchard\\" + to_string(i) + "AppleOrchard_RowsFound.jpg";
			string treesFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\AppleOrchard\\" + to_string(i) + "AppleOrchard_TreesFound.jpg";
			string applesFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\AppleOrchard\\" + to_string(i) + "AppleOrchard_ApplesFound.jpg";

			if (BGRImg.rows != 0) // check to make sure the read-in image is actually an image
			{
				// Batch Processing the Apple Orchard Images
				AppleOrchard* AppleImage = new AppleOrchard(BGRImg);

				AppleImage->outputRowHistogram(rowHistoFilePath);
				AppleImage->findRowLocations();
				AppleImage->visualizeRowCoordinates(rowsFoundFilePath);
				AppleImage->CountApplesInFirstRow(applesFoundFilePath);

				OrchardImages.push_back(AppleImage);
			}
		}
		else
		{
			BGRImg = imread("Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\UnprocessedImages\\160901 crop load\\DJI_00" + to_string(i) + ".jpg");
			
			// Output File Paths
			string rowHistoFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\AppleOrchard\\" + to_string(i) + "AppleOrchard_RowHisto.jpg";
			string rowsFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\AppleOrchard\\" + to_string(i) + "AppleOrchard_RowsFound.jpg";
			string treesFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\AppleOrchard\\" + to_string(i) + "AppleOrchard_TreesFound.jpg";
			string applesFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\AppleOrchard\\" + to_string(i) + "AppleOrchard_ApplesFound.jpg";

			if (BGRImg.rows != 0) // check to make sure the read-in image is actually an image
			{
				// Batch Processing the Apple Orchard Images
				AppleOrchard* AppleImage = new AppleOrchard(BGRImg);

				AppleImage->outputRowHistogram(rowHistoFilePath);
				AppleImage->findRowLocations();
				AppleImage->visualizeRowCoordinates(rowsFoundFilePath);
				AppleImage->CountApplesInFirstRow(applesFoundFilePath);

				OrchardImages.push_back(AppleImage);
			}
		}
	}
}

void BatchProcessBlossomOrchard(const int firstImgNum, const int lastImgNum)
{
	// Variable Declarations
	Mat BGRImg;
	vector< BlossomOrchard* > OrchardImages;

	for (size_t i = firstImgNum; i < lastImgNum + 1; i++)
	{
		if (i < 10)
		{
			// Read in Images from Blossom Drone Photos
			BGRImg = imread("Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\UnprocessedImages\\160503_blossoms\\DJI_000" + to_string(i) + ".jpg");

			// Output File Paths --Figure out how you want these to be organized...should have some organization scheme
			string rowHistoFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_RowHisto.jpg";
			string treeHistoFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_TreeHisto.jpg";
			string rowsFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_RowsFound.jpg";
			string treesFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_TreesFound.jpg";
			string treesFoundAreaMethodFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_TreesFound_AM.jpg";
			string treesFoundContoursMethodFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_TreesFound_AM.jpg";

			string blossomsFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_BlossomsFound.jpg";

			// Batch Processing the Blossom Orchard Images

			// Error Handling
			if (BGRImg.rows != 0) // check to make sure the read-in image is actually an image
			{
				BlossomOrchard* BlossomImage = new BlossomOrchard(BGRImg);

				BlossomImage->findRowLocations();
				BlossomImage->outputRowHistogram(rowHistoFilePath);
				BlossomImage->visualizeRowCoordinates(rowsFoundFilePath);
				BlossomImage->findTreesTrunkMethod(treeHistoFilePath, treesFoundFilePath);
				BlossomImage->findTreesAreaMethod(treesFoundAreaMethodFilePath);
				//BlossomImage->findTreesInFirstRow();
				//BlossomImage->outputTreeHistogram(treeHistoFilePath);
				//BlossomImage->visualizeTreeCoordinates(treesFoundFilePath);
				BlossomImage->countBlossomsInFirstRow(blossomsFoundFilePath);

				// Store The Pointer to the Object in the Orchard-Image Container 
				OrchardImages.push_back(BlossomImage);
			}
		}

		else
		{
			// Read in Images from Blossom Drone Photos
			BGRImg = imread("Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\UnprocessedImages\\160503_blossoms\\DJI_00" + to_string(i) + ".jpg");

			// Output File Paths --Figure out how you want these to be organized...should have some organization scheme
			string rowHistoFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_RowHisto.jpg";
			string treeHistoFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_TreeHisto.jpg";
			string rowsFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_RowsFound.jpg";
			string treesFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_TreesFound.jpg";
			string treesFoundAreaMethodFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_TreesFound_AM.jpg";
			string treesFoundContoursMethodFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_TreesFound_CM.jpg";
			string blossomsFoundFilePath = "Z:\\Desktop\\Honors_Thesis\\ImageAnalysisLab\\Photos\\ProcessedPhotos\\BlossomOrchard\\" + to_string(i) + "BlossomOrchard_BlossomsFound.jpg";

			// Batch Processing the Blossom Orchard Images

			// Error Handling
			if (BGRImg.rows != 0) // check to make sure the read-in image is actually an image
			{
				BlossomOrchard* BlossomImage = new BlossomOrchard(BGRImg);

				BlossomImage->findRowLocations();
				BlossomImage->outputRowHistogram(rowHistoFilePath);
				BlossomImage->visualizeRowCoordinates(rowsFoundFilePath);
				//BlossomImage->findTreesContoursMethod(treesFoundContoursMethodFilePath);
				BlossomImage->findTreesAreaMethod(treesFoundAreaMethodFilePath);
				BlossomImage->findTreesTrunkMethod(treeHistoFilePath, treesFoundFilePath);
				//BlossomImage->findTreesInFirstRow();
				//BlossomImage->outputTreeHistogram(treeHistoFilePath);
				//BlossomImage->visualizeTreeCoordinates(treesFoundFilePath);
				BlossomImage->countBlossomsInFirstRow(blossomsFoundFilePath);

				// Store The Object in the Orchard Image Container 
				OrchardImages.push_back(BlossomImage);
			}
		}
	}
}

