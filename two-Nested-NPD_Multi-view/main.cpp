#include <detector.h>



const std::string G_PATH_DETECTOR="../nested_cascade_Multi-view.xml";//cascade 
const int G_DEDVICE=0;//device capture


int main()
{

cv::namedWindow("DETECTION",cv::WINDOW_NORMAL);

CASCADE_CLASSIFIERS_EVALUATION detectorFace(G_PATH_DETECTOR);//detector class



/*________________________________detector configuration____________________________________*/

/*----------------configuring detection angle RIP(rotation-in-plane)------------------------*/
std::vector<double> myDegrees;

/*
myDegrees.push_back(-30);
myDegrees.push_back(-15);
*/

myDegrees.push_back(0);

/*
myDegrees.push_back(15);
myDegrees.push_back(30);
*/

/*------------------------------------------------------------------------------------------*/


detectorFace.flipFeatures(true); //applying linear transformation of reflection on each of the NPD features ROP(rotation-off-plane).

detectorFace.setDegreesDetections(myDegrees);//set angle RIP


detectorFace.setSizeBase(80); //minimum search window
detectorFace.setFactorScaleWindow(1.1); //scale factor
detectorFace.setStepWindow(0.15); //step window
detectorFace.setGroupThreshold(2); //minimum number of neighbors
detectorFace.setEps(0.5); //relative difference between sides of the rectangles to merge them into a group (see https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html)
detectorFace.setFlagActivateSkinColor(false); //skin color option
detectorFace.setLineThicknessRectangles(8); //thickness window detection
detectorFace.setColorRectangles(cv::Scalar(0,255,255)); //detection window color
detectorFace.setSizeMaxWindow(2000); //maximum size of the image(or frame)

detectorFace.initializeFeatures();// initialize NPD features

//detectorFace.setNumberClassifiersUsed(18); //number of states to use


//_________________________________________________________________________________________________//


cv::VideoCapture cap(G_DEDVICE);
cv::Mat img;


  while(true){

    cap>>img;

    detectorFace.detectObjectRectanglesRotatedGrouped(img);

    cv::imshow("DETECTION",img);
    if(cv::waitKey(1) >= 0) break;

  }


return 0;

}


