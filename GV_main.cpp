#include "kyheader.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"
#include "CmFile.h"

//using namespace cv;
//using libv4l2;

int main(int argc, char* argv[])
{
    string default_dir=CmFile::GV_pwd();

    cout << "Hello world" << default_dir<<  endl ;
	VideoCapture cap;
	cout << "Video Capture" << endl;

	if(argc >1)
	{
		cap.open(string(argv[1]));
		if(!cap.isOpened())
		{
			cout << "Can not open the file " <<string(argv[1]) << endl;
			return -1;
		}

	}
	else
	{
		cap.open(0);
	}

	if(argc >2)
	{
        default_dir=string(argv[2]);
	}
	double fps=cap.get(CV_CAP_PROP_FPS);
	cout << "Frames per second" << fps << endl ;
    DataSetVOC voc2007(default_dir);

	Mat frame; namedWindow("video",1);

	double base=2 ; int W=8; int NSS=2; int numPerSz=130;
	Objectness objNess(voc2007, base, W, NSS);
	srand((unsigned int)time(NULL));
	objNess.loadTrainedModel();

	// Initialize BB colors
	vector<Scalar> colors;
	for (int iTemp=0;iTemp< 200; iTemp++)
		colors.push_back(Scalar(rand() % 255,rand() % 255,rand() % 255));

	//ValStructVec<float, Vec4i> boxes;
	for(int iFrame=0;iFrame<10000;iFrame++)
	{
		cout << iFrame << endl;
		cap >> frame; if(!frame.data) break;
//    	voc2007.loadAnnotations();

//	**************objNess.getObjBndBoxesForTestsFast(boxesTests, numPerSz);
		const int TestNum = 1;
		vecM imgs3u(TestNum);
		imgs3u[0] = frame;

		ValStructVec<float, Vec4i> boxesTests;
		boxesTests.clear();
		boxesTests.reserve(10000);
		objNess.getObjBndBoxes(imgs3u[0], boxesTests, numPerSz);
		//boxes.clear();
		//boxes = boxesTests[0];
		int num_BB;
		num_BB=0;
		for (int j = 0; j < boxesTests.size(); j++)
		{
			if(boxesTests(j)> -0.8)
			{
				rectangle(frame,Point(boxesTests[j][0]-1,boxesTests[j][1]-1) , Point(boxesTests[j][2]-1,boxesTests[j][3]-1) , colors[num_BB],3);
				cout << "BB " <<  j << " (" << boxesTests[j][0] << "," << boxesTests[j][1] << "," << boxesTests[j][2] << "," << boxesTests[j][3] << ")" << endl;
				num_BB=num_BB+1;
			}
		}

		cout << "Frame "<< iFrame << " with " << boxesTests.size() << " BBs has " << num_BB << endl;

/*		_boxesTests[i].resize(boxesTests[i].size());
		for (int j = 0; j < boxesTests[i].size(); j++)
			_boxesTests[i][j] = boxesTests[i][j];*/

//  ***********************************
		imshow("video",frame);   if(waitKey(1) >=0) break;
	}

}
