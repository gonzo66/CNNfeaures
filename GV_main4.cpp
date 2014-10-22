/*
This program gets an image name as a parameter. Use the full image to calcuate the CNN . Layer 5, 6 and predicion layer are saved in disk
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
//#define MMAX_PROPOSALS   200
#define MMAX_PROPOSALS   1
#define WIDTH   227
#define HEIGHT  227
#define NUMCHANNELS 3
#define BINPROTO_FILE "/home/gonzalo/Downloads/caffe/caffe-master/examples/imagenet/caffe_reference_imagenet_model"
//#define FEAT_EXT_FILE "/home/gonzalo/Downloads/caffe/caffe-master/examples/imagenet/imagenet_deploy2.prototxt"
#define FEAT_EXT_FILE "/home/gonzalo/Downloads/caffe/caffe-master/examples/imagenet/imagenet_deploy3.prototxt"
#define MEAN_FILE "/home/gonzalo/Downloads/caffe/caffe-master/data/ilsvrc12/imagenet_mean.binaryproto"
#define LABELS_FILE "/home/gonzalo/Downloads/caffe/caffe-master/data/ilsvrc12/imagenet_words.txt"

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
    string fc6_dir=results_dir+"fc6/";
    string fc7_dir=results_dir+"fc7/";
    string prob_dir=results_dir+"prob/";


std::cout << "Hello world" << default_dir<<  std::endl ;


    // ******** Obtain 227x227 mean Image ********************************
    // Load the array with mean (256 x 256 x 3)
    Mat meanData = getMean();

    std::cout << "pause 1" << std::endl;

    // cut to (227 x 227 x 3)
    int centerx= (int)(meanData.size().width / 2);
    int centery= (int)(meanData.size().height /2);
    cv::Rect centerwindow(centerx-(int)WIDTH/2 , centery-(int)HEIGHT/2, WIDTH , HEIGHT );

std::cout << "pause 2 "<< centerx <<","<< centery <<std::endl;
    meanData=meanData(centerwindow);
std::cout << "pause 3 "<< std::endl;
std::cout << meanData.size().width << " " << meanData.size().height << " " << meanData.channels() << std::endl;
    cvtColor(meanData, meanData, CV_BGR2RGB);
std::cout << "pause 4" << std::endl;

    std::cout << meanData.rows << std::endl;

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

      //create the deep network
      shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto));

      // Load pre-trained model
      feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

      //string extract_feature_blob_name(argv[++arg_pos]);
      CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
          << "Unknown feature blob name " << extract_feature_blob_name
          << " in the network " << feature_extraction_proto;

      CHECK(feature_extraction_net->has_blob(extract_feature_blob_name1))
          << "Unknown feature blob name " << extract_feature_blob_name1
          << " in the network " << feature_extraction_proto;

      CHECK(feature_extraction_net->has_blob(extract_feature_blob_name2))
          << "Unknown feature blob name " << extract_feature_blob_name2
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
/*    cap.open(filename);
    if(!cap.isOpened())
    {
        std::cout << "Can not open the file " <<string(argv[1]) << std::endl;
        return -1;
    }

/*	else
	{
		cap.open(0);
	}
	double fps=cap.get(CV_CAP_PROP_FPS);
	std::cout << "Frames per second" << fps << std::endl ;
*/



    // *********** Start Objectness ********************************
std::cout << "start objectness" << std::endl ;
    DataSetVOC voc2007(default_dir);    //setup voc directory
    int mem_size=MMAX_PROPOSALS*WIDTH*HEIGHT*NUMCHANNELS*sizeof(Dtype);
    Dtype *memchunk = (Dtype*) malloc (mem_size);
    //Dtype memchunk[MMAX_PROPOSALS*WIDTH*HEIGHT*NUMCHANNELS];   //Memory where the proposals are saved to feed the forward network
/*
	double base=2 ; int W=8; int NSS=2; int numPerSz=130;
	Objectness objNess(voc2007, base, W, NSS);
	srand((unsigned int)time(NULL));

std::cout << "train model" << std::endl ;
	objNess.loadTrainedModel();

std::cout << "model trained" << std::endl ;
	// Initialize BB colors
	vector<Scalar> colors;
	//Mat inputsImg[MAX_PROPOSALS];
	//Mat inputsImg(WIDTH,HEIGHT,CV_32FC3);*/
	Mat inputsImg = Mat::zeros(WIDTH,HEIGHT,CV_32FC3);

std::cout << "nputs img" << std::endl ;

/*	for (int iTemp=0;iTemp< 200; iTemp++)
		colors.push_back(Scalar(rand() % 255,rand() % 255,rand() % 255));
*/		int iFrame2=0;
		int iFrame=1;

std::cout << "reading frame" << std::endl ;
	//ValStructVec<float, Vec4i> boxes;
//	for(int iFrame=0;iFrame<100000;iFrame++)    // Loop over 10000 frames or until the end of video
	{
	    #ifdef MYDEBUG
	    std::cout << "reading frame" << std::endl;
	    #endif
            frame= imread(filename);
//            if(!frame.data) break;
        //frame=cv::imread("/home/gonzalo/Downloads/caffe/caffe-master/examples/images/cat2.jpg");
        //frame=cv::imread("/home/gonzalo/Downloads/caffe/caffe-master/examples/images/dog.jpg");

	    //if (iFrame%30==0 )  // process every XXX frames
	    {
            iFrame2++;
            string imgnum=std::to_string(iFrame2);
            int numchars=imgnum.size();
            string pre="";
            for (int iTemp2=0;iTemp2<(8-numchars);iTemp2++)
                pre+='0';

	        std::ofstream myfile;
            myfile.open (bb_dir + "/" + prefix + pre + std::to_string(iFrame) + ".txt" );

            std::ofstream prob_out (prob_dir + "/" + prefix + pre + std::to_string(iFrame) + ".bin" , std::ios::out | std::ios::binary);
            std::ofstream fc6_out (fc6_dir + "/" + prefix + pre + std::to_string(iFrame) + ".bin" , std::ios::out | std::ios::binary);
            std::ofstream fc7_out (fc7_dir + "/" + prefix + pre + std::to_string(iFrame) + ".bin" , std::ios::out | std::ios::binary);


	        #ifdef MYDEBUG
	        std::cout << "Frame: "<< iFrame << std::endl;
            std::cout << "converting to float" << std::endl;
            #endif
            frame.convertTo(frameFloat, CV_32FC3);  //Convert to float

            #ifdef MYDEBUG
            std::cout << "channel correction" << std::endl;
            #endif
std::cout << "CTCOLOR otra vez"<< std::endl;
            cvtColor(frameFloat, frameFloat, CV_BGR2RGB);   //swap color channels
            //multiply(frameFloat,frameFloat);
            //frameFloat=frameFloat/255;
            //imshow("original",frameFloat);
            //std::cout << frame.rows<< frame.cols << std::endl;


            //	************** Get Proposals. Bounding boxes ************************/
            //	************** objNess.getObjBndBoxesForTestsFast(boxesTests, numPerSz);
            #ifdef MYDEBUG
            std::cout << "getting proposals" << std::endl;
            #endif

            const int TestNum = 1;
            vecM imgs3u(TestNum);
            imgs3u[0] = frame;

       /*     ValStructVec<float, Vec4i> boxesTests;
            boxesTests.clear();
            boxesTests.reserve(10000);
std::cout << "get objecness" << std::endl;
           objNess.getObjBndBoxes(imgs3u[0], boxesTests, numPerSz);

std::cout << "objecness" << std::endl;
            #ifdef MYDEBUG
            std::cout << "proposal obtained" << std::endl;
            #endif
            //boxes.clear();
            //boxes = boxesTests[0];
*/

            // ************ Extract proposals from frame and substract the mean of the CNN dataset *********
            //for (int j = 0; j < MMAX_PROPOSALS; j++)
            int j=0;
            {
                //if(boxesTests(j)> -0.8)
                //{
                    //namedWindow("video"+ std::to_string(j),1);
/*                    #ifdef MYDEBUG
                    std::cout << "*** Extract proposal " << j <<std::endl;
                    #endif
                    cv::Rect window(boxesTests[j][0]-1,boxesTests[j][1]-1, boxesTests[j][2]-boxesTests[j][0]-1,boxesTests[j][3]-boxesTests[j][1]-1 );

                    myfile << boxesTests[j][0] << "\t" << boxesTests[j][1] << "\t" << boxesTests[j][2] << "\t" << boxesTests[j][3] << std::endl ;

                    //resize(frameFloat(window), inputsImg[j],  inputsImg[j].size() );
                    #ifdef MYDEBUG
                    std::cout << "reshaping proposal" << std::endl;
                    #endif
*/
                   // if (j==0)
                   //     resize(frameFloat, inputsImg,  inputsImg.size() );
                    //else
                        resize(frameFloat, inputsImg,  inputsImg.size() );
                    //inputsImg[j] -= meanData;
                    #ifdef MYDEBUG
                    std::cout << "substracting mean" << std::endl;
                    #endif
                    inputsImg -= meanData;
                    #ifdef MYDEBUG
                    std::cout << "copy proposal to input" << std::endl;
                    #endif
                    //fix_format(inputsImg[j], (memchunk+offset(j)));   // re-arrange memory positions according to [chanels][width][height]
                    fix_format(inputsImg.clone(), (memchunk+offset(j)));   // re-arrange memory positions according to [chanels][width][height]

                 //   imshow("video" + std::to_string(j) , inputsImg[j]/255);
                //    waitKey(30);
                   //}
            }


            // ******** HERE starts CNN of the selected windows *******************

              int num_mini_batches = 1;
              LOG(ERROR)<< "Extacting Features";
              //Datum datum;
              //leveldb::WriteBatch* batch = new leveldb::WriteBatch();
//              const int kMaxKeyStrLength = 100;
//              char key_str[kMaxKeyStrLength];
//              int num_bytes_of_binary_code = sizeof(Dtype);
              vector<Blob<float>*> input_vec;
//              int image_index = 0;

             //free(feature_extraction_net->blobs()[0]->mutable_cpu_data());
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

                const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(extract_feature_blob_name);  // Extract Output layer
                const shared_ptr<Blob<Dtype> > feature_blob1 = feature_extraction_net->blob_by_name(extract_feature_blob_name1);  // Extract Output layer
                const shared_ptr<Blob<Dtype> > feature_blob2 = feature_extraction_net->blob_by_name(extract_feature_blob_name2);  // Extract Output layer

 const shared_ptr<Blob<Dtype> > feature_blobpool5 = feature_extraction_net->blob_by_name("pool1"); 
std::cout << "pool1 is " << feature_blobpool5->num() << " - " <<feature_blobpool5->count()  << std::endl;
                int num_features = feature_blob->num();
                int dim_features = feature_blob->count() / num_features;
                Dtype* feature_blob_data;
std::cout << "fc6 is " << feature_blob1->num() << " - " <<feature_blob1->count()  << std::endl;

                int num_features1 = feature_blob1->num();
//                int dim_features1 = feature_blob1->count() / num_features1;
//                Dtype* feature_blob_data1;

                int num_features2 = feature_blob2->num();
//                int dim_features2 = feature_blob2->count() / num_features2;
//                Dtype* feature_blob_data2;

                #ifdef MYDEBUG
                std::cout << "displaying good proposals" << std::endl;
                #endif

                prob_out.write((char *)feature_blob->mutable_cpu_data(), feature_blob->offset(num_features)*sizeof(Dtype) );
                fc6_out.write((char *)feature_blob1->mutable_cpu_data(), feature_blob1->offset(num_features1)*sizeof(Dtype) );
                fc7_out.write((char *)feature_blob2->mutable_cpu_data(), feature_blob2->offset(num_features2) *sizeof(Dtype));

std::cout << "displaying good proposals: "<< num_features<< std::endl;
                for (int n = 0; n < num_features; ++n) {
                  //datum.set_height(dim_features);
                  //datum.set_width(1);
                  //datum.set_channels(1);
                  //datum.clear_data();
                  //datum.clear_float_data();
                  feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob->offset(n);
                  vector<int> objlist;
                  objlist.clear();
//                  float mymax=0;
                  int val=0;
                  for (int d = 0; d < dim_features; ++d) {
                    //datum.add_float_data(feature_blob_data[d]);
                    // Let's declare a possible detection
              /*      if(feature_blob_data[d]>mymax)
                    {
                        val=d;
                        mymax=feature_blob_data[d];
                    }*/
                    if(feature_blob_data[d]>0.8)
                    {
                        objlist.push_back(d);
                        val=d;
                    }
                  }
               //   objlist.push_back(val);
/*                  if (objlist.size()>0 )
                  {
                      rectangle(frame,Point(boxesTests[n][0]-1,boxesTests[n][1]-1) , Point(boxesTests[n][2]-1,boxesTests[n][3]-1) , colors[(int)(val/5)],3);
                      string detected;
                      for (int iDet=0;iDet < objlist.size(); iDet++)
                        detected= detected + labels[objlist[iDet]] + " - ";
                      putText(frame, detected , cvPoint(boxesTests[n][0]+10,boxesTests[n][1]+10),  FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(200,200,250), 1, CV_AA);
                  }
*/

                }  // for (int n = 0; n < num_features; ++n)
              }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)


              LOG(ERROR)<< "Successfully extracted the features!";


    //  ***********************************
            imshow("video",frame);

            imwrite(prefix + "-" + pre +std::to_string(iFrame2)+".png",frame);

        //    imshow("mymean", meanData/255);

            myfile.close();
            prob_out.close();
            fc6_out.close();
            fc7_out.close();

//            if(waitKey(10) >=0) break;

	    }
	}
	free(memchunk);

}





