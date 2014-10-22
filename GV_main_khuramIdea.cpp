/*
This program gets an image name as a parameter. CNN is calculated without resizing. Layer 5, 6 and predicion layer are saved in disk
*/
#include "kyheader.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"
#include "CmFile.h"

//#include "mymean.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>

//using namespace cv;
//using libv4l2;

#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#define Dtype  float
#define MMAX_PROPOSALS   200
#define WIDTH   227
#define HEIGHT  227
#define NUMCHANNELS 3
#define BINPROTO_FILE "/home/gonzalo/Downloads/caffe/caffe-master/examples/imagenet/caffe_reference_imagenet_model"
//#define FEAT_EXT_FILE "/home/gonzalo/Downloads/caffe/caffe-master/examples/imagenet/imagenet_deploy3.prototxt"
#define FEAT_EXT_FILE "/home/gonzalo/Downloads/caffe/caffe-master/examples/imagenet/imagenet_deploy_khuram.prototxt"
#define MEAN_FILE "/home/gonzalo/Downloads/caffe/caffe-master/data/ilsvrc12/imagenet_mean.binaryproto"
#define LABELS_FILE "/home/gonzalo/Downloads/caffe/caffe-master/data/ilsvrc12/imagenet_words.txt"
#define MBB	131

//#define MYDEBUG 1
#include <iostream>
#include <fstream>
//#include <fstream.h>

inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0)
{
	return ((n * NUMCHANNELS + c) * HEIGHT + h) * WIDTH + w;
}


using namespace caffe;  // NOLINT(build/namespaces)



Mat getMean()
{
    BlobProto blob_proto;
    string mean_file=MEAN_FILE;
    ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);

    int width =  blob_proto.width();
    int height = blob_proto.height();
    int channels= blob_proto.channels();

    Mat image1(Size(width, height), CV_32FC3);

    int i=0;

    float_t* pixelPtr = (float_t*)image1.data;


    for (int c=0;c<3; c++)
        for (int h=0; h<height ; h++)
        {
            for (int w=0; w<width; w++)
            {
             /*  if (c==2 && h<3)
                {
                std::cout << blob_proto.data(i) << ' ' ;

               }*/
               //image1.at<Vec3b>(w,h)[c]= blob_proto.data(i);
                //image1.at<Dtype>(h,w,c) = blob_proto.data(i);
                pixelPtr[h*width*channels + w*channels + c]= blob_proto.data(i);
                //image1.data[h*height*channels + w*channels + c]=blob_proto.data(i);
                i++;
            }
           //if (h<3)  std::cout << std::endl;
        }

    return image1;
}

/*template <typename T>
T StringToNumber ( const string &Text )//Text not by const reference so that the function can be used with a 
{                               //character array as argument
	std::stringstream ss(Text);
	T result;
	return ss >> result ? result : 0;
}
*/

void fix_format(Mat original, Dtype *destMat)
{
    #ifdef MYDEBUG
    std::cout << "reshaping figure" << std::endl;
    #endif
    int channels = original.channels();
    cv::Size s = original.size();
    int rows = s.height;
    int cols = s.width;

    float_t* pixelPtr = (float_t*)original.data;

  //  int counter=0;
    for (int k=0;k<channels ; k++)  //channels
    {
        #ifdef MYDEBUG
        std::cout << "k=" << k << std::endl;
        #endif
        for (int i=0;i<rows ; i++)   // height
        {
            for (int j=0;j<cols ; j++)   // width
            {
                (*destMat)=pixelPtr[i*original.cols*channels + j*channels + k];
                destMat++;
            }
        }
    }

    #ifdef MYDEBUG
    std::cout << "reshaping done" << std::endl;
    #endif
}

int main(int argc, char* argv[])
{

    string default_dir=CmFile::GV_pwd();    //get default directory
    string results_dir=default_dir + "/results/";
    string bb_dir=results_dir+"BB/";
    string pool5_dir=results_dir+"pool5/";
    string video_dir=default_dir+"UCFSports/Diving_Side_001.vob/";

	std::cout << "Hello world" << default_dir<<  std::endl ;


    // ******** Obtain 227x227 mean Image ********************************
    // Load the array with mean (256 x 256 x 3)
    Mat meanData = getMean();
    Dtype meanpixel=*(meanData.ptr<Dtype>(10));




    std::cout << "mean pixels is " << meanpixel << std::endl;


  // ********** Load the classification labels in memory ***************************
    std::cout << "loading file" << std::endl;
    string labels[1000];
    std::ifstream myfile (LABELS_FILE);
    if (myfile.is_open())
    {
        for (int iTemp=0;iTemp<1000;iTemp++)
        {
            getline (myfile,labels[iTemp]);
        }
        myfile.close();
     }



    // ************** Initialize Caffe: mode, networks, imagenet model, etc. ***************************

    std::string pretrained_binary_proto= BINPROTO_FILE ;
    std::string feature_extraction_proto=FEAT_EXT_FILE;
    std::string extract_feature_blob_name="prob" ;
    std::string extract_feature_blob_name1="fc6";
    std::string extract_feature_blob_name2="fc7";
    std::string extract_feature_blob_name3="pool5";


    bool GPU=false;

    if (GPU) {
        LOG(ERROR)<< "Using GPU";
        uint device_id = 0;
        CHECK_GE(device_id, 0);
        LOG(ERROR) << "Using Device_id=" << device_id;
        Caffe::SetDevice(device_id);
        Caffe::set_mode(Caffe::GPU);
      } else {
        LOG(ERROR) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
      }
      Caffe::set_phase(Caffe::TEST);
      //string pretrained_binary_proto(argv[++arg_pos]);
      //string feature_extraction_proto(argv[++arg_pos]);

std::cout << "Creating Deep Network" << std::endl;
      //create the deep network
      shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto));

      // Load pre-trained model
std::cout << "Loading pre-trained model" << std::endl;
      feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

      //string extract_feature_blob_name(argv[++arg_pos]);
    CHECK(feature_extraction_net->has_blob(extract_feature_blob_name3))
          << "Unknown feature blob name " << extract_feature_blob_name3
          << " in the network " << feature_extraction_proto;

    // ********** Initialize video capture ****************************************
//    VideoCapture cap;   //video capture device
    	Mat frame; //namedWindow("video",1);    // Mat array with current frame
	Mat frameFloat;                         // To save float version of the frame
	#ifdef MYDEBUG
	std::cout << "Video Capture" << std::endl;
	#endif
   string filename="/home/gonzalo/Downloads/bing/BingObjectnessCVPR14/Objectness-master/Src/P03.mp4";
	if(argc >1)
	{
	    filename=string(argv[1]);
	}
	std::cout << filename << std::endl;
	std::string prefix= filename.substr (filename.find_last_of("/\\")+1, filename.length()-filename.find_last_of("/\\")-5);
	std::cout << prefix << std::endl;


   // *********** Start Objectness ********************************
// Initialize BB colors
	vector<Scalar> colors;
	for (int iTemp=0;iTemp< 200; iTemp++)
		colors.push_back(Scalar(rand() % 255,rand() % 255,rand() % 255));



// ************ Memory for each sub image of the track *************
	int mem_size= MBB * MBB * 3*sizeof(Dtype);  // a 3x3 in the pool5 space
	Dtype *memchunk = (Dtype*) malloc (mem_size);

	// Open track files
	std::ifstream myfile2 (filename);
  	if (myfile2.is_open())
  	{
	int iLines=0;
	string line;
	std::ofstream pool5_out (pool5_dir + "/" + prefix  + ".bin" , std::ios::out | std::ios::binary);
      while ( getline (myfile2,line) )
    	   {
		iLines++;
		std::cout << line << '\n';
		std::istringstream iss(line);
		
		int start_frame;
		iss>>start_frame;
		
		for(int iTrack=0;iTrack<15;iTrack++)
		{
		    float x_cen, y_cen;
		    iss>>x_cen;
		    iss>>y_cen;
	    	std::cout << "line " << iLines << " : " << x_cen << " , " << y_cen << std::endl;
			
			string imgnum=std::to_string(start_frame + iTrack);
            int numchars=imgnum.size();
			string pre="";
            for (int iTemp2=0;iTemp2<(5-numchars);iTemp2++)
                pre+='0';
                
            std::string load_name= video_dir + pre + imgnum + ".jpg" ;
            std::cout<< load_name << std::endl;
            frame= imread(load_name);
            imshow("frame",frame);
            
            int x_start=x_cen- (int)(MBB/2);
            int y_start=y_cen- (int)(MBB/2); 
            
            Mat frame_process(MBB,MBB,CV_8UC3, Scalar::all(0));
            
            int delta_x=std::max(0,-1*x_start);
            int delta_y=std::max(0,-1*y_start);
            int max_x=std::max(0, x_start + MBB - frame.size().width );
            int max_y=std::max(0, y_start + MBB - frame.size().height );
            
            //std::cout << "x_start: " << x_start << " y_start: " << y_start << " delta_x: " << delta_x  << " delta_y: " << delta_y << " max_x: " << max_x << " max_y: " << max_y  << std::endl;
            std::cout << 0+delta_x << "," << 0+delta_y <<"," << MBB-1-max_x -delta_x << "," << MBB-1-max_y -delta_y<< std::endl;
            Mat tmp = frame_process(cv::Rect(0+delta_x,0+delta_y,MBB-1-max_x-delta_x,MBB-1-max_y-delta_y));
			cv::Rect centerwindow(std::max(0,x_start), std::max(0,y_start) , MBB-1-max_x -delta_x , MBB-1-max_y -delta_y );
			//std::cout << std::max(0,x_start) << "," << std::max(0,y_start) << "," << min(frame.size().width-1 , x_start + MBB) << "," << min(frame.size().height-1, y_start + MBB) << std::endl; 
			
			frame=frame(centerwindow);
			frame.copyTo(tmp);
			
			frame_process.convertTo(frameFloat, CV_32FC3);  //Convert to float

            #ifdef MYDEBUG
            std::cout << "channel correction" << std::endl;
            #endif
			std::cout << "CTCOLOR otra vez"<< std::endl;
            cvtColor(frameFloat, frameFloat, CV_BGR2RGB);   //swap color channels
            Mat myMean = Mat(frameFloat.size().height, frameFloat.size().width, CV_32FC3, Scalar(meanpixel,meanpixel,meanpixel));

           
			std::cout << "Substracting mean" << std::endl;
			std::cout << "frameFloat is " <<  frameFloat.size().width << " - " <<  frameFloat.size().height <<  " - " <<  frameFloat.channels() <<std::endl;
			std::cout << "myMean is " <<  myMean.size().width << " - " <<  myMean.size().height <<  " - " <<  myMean.channels() <<std::endl;
			frameFloat -= myMean;//GV_main5

			#ifdef MYDEBUG
			std::cout << "copy proposal to input" << std::endl;
			#endif
			//fix_format(inputsImg[j], (memchunk+offset(j)));   // re-arrange memory positions according to [chanels][width][height]
			std::cout << "copy proposal to input" << std::endl;
			fix_format(frameFloat.clone(), memchunk);   // re-arrange memory positions according to [chanels][width][height]



            // ******** HERE starts CNN of the selected windows *******************

              int num_mini_batches = 1;
              LOG(ERROR)<< "Extacting Features";
              vector<Blob<float>*> input_vec;
              feature_extraction_net->blobs()[0]->set_cpu_data(memchunk);

              //  *** Currrently only 1 minibatch
              for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
                //memcpy(feature_extraction_net->blobs()[0]->mutable_cpu_data() , memchunk,sizeof(memchunk)*sizeof(Dtype));
                //std::cout << feature_extraction_net->blobs()[0]->mutable_cpu_data() << std::endl;
                //std::cout << sizeof(feature_extraction_net->blobs()[0]->mutable_cpu_data()) << std::endl;
                #ifdef MYDEBUG
                std::cout << "copying inputs to data" << std::endl;
                #endif

                //memcpy(feature_extraction_net->blobs()[0]->mutable_cpu_data() , memchunk,mem_size);   //Copy memchunk to input data network

                #ifdef MYDEBUG
                std::cout << "network running forward" << std::endl;
                #endif

                feature_extraction_net->Forward(input_vec);

                #ifdef MYDEBUG
                std::cout << "getting output" << std::endl;
                #endif

                const shared_ptr<Blob<Dtype> > feature_blob3 = feature_extraction_net->blob_by_name(extract_feature_blob_name3);  // Extract Output layer
                int num_features3 = feature_blob3->num();
                Dtype* feature_blob_data3;

                #ifdef MYDEBUG
                std::cout << "displaying good proposals" << std::endl;
                #endif

                pool5_out.write((char *)feature_blob3->mutable_cpu_data(), feature_blob3->offset(num_features3) *sizeof(Dtype));
               }
			
			//std::cout << frame.size().width << " , " << frame.size().height << std::endl;
			//imshow("copyfrom",frame);
            //imshow("cut input" , frame_process);
            //if(waitKey(3000) >=0) break;
		}

       }// End of tracks		
    	myfile2.close();
    	pool5_out.close();
  	}
	else std::cout << "Unable to open file"; 
	
	
	LOG(ERROR)<< "Successfully extracted the features!";    
	free(memchunk);

}





