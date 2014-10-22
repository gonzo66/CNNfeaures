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
#define FEAT_EXT_FILE "/home/gonzalo/Downloads/caffe/caffe-master/examples/imagenet/imagenet_deploy3.prototxt"
#define MEAN_FILE "/home/gonzalo/Downloads/caffe/caffe-master/data/ilsvrc12/imagenet_mean.binaryproto"
#define LABELS_FILE "/home/gonzalo/Downloads/caffe/caffe-master/data/ilsvrc12/imagenet_words.txt"

#define MYDEBUG 1
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

                pixelPtr[h*width*channels + w*channels + c]= blob_proto.data(i);
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
    string pool5_dir=results_dir+"pool5/";
    string fc6_dir=results_dir+"fc6/";

#ifdef MYDEBUG
std::cout << "Hello world" << default_dir<<  std::endl ;
#endif

    // ******** Obtain 227x227 mean Image ********************************
    // Load the array with mean (256 x 256 x 3)
    Mat meanData = getMean();
    Dtype meanpixel=*(meanData.ptr<Dtype>(10));
    #ifdef MYDEBUG
    std::cout << "mean pixels is " << meanpixel << std::endl;
    #endif

  // ********** Load the classification labels in memory ***************************
    #ifdef MYDEBUG
    std::cout << "loading file" << std::endl;
    #endif

    // ************** Initialize Caffe: mode, networks, imagenet model, etc. ***************************
    std::string pretrained_binary_proto= BINPROTO_FILE ;
    std::string feature_extraction_proto=FEAT_EXT_FILE;
    std::string extract_feature_blob_name="prob" ;
    std::string extract_feature_blob_name1="fc6";
    std::string extract_feature_blob_name2="fc7";
    std::string extract_feature_blob_name3="pool5";


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

	if(argc >2)
	{
	    pool5_dir=string(argv[2]);
	}

    #ifdef MYDEBUG
    std::cout << filename << std::endl;
    std::cout << pool5_dir << std::endl;
    #endif

	std::string prefix= filename.substr (filename.find_last_of("/\\")+1, filename.length()-filename.find_last_of("/\\")-5);

    #ifdef MYDEBUG
    std::cout << "prefix is: " << prefix << std::endl;
    #endif

   	int iFrame2=0;
    int iFrame=1;

    #ifdef MYDEBUG
    std::cout << "reading frame" << std::endl ;
    #endif

    frame= imread(filename);




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
      #ifdef MYDEBUG
      std::cout << "Creating Deep Network" << std::endl;
      #endif
      shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto,frame.size().height,frame.size().width));

      // Load pre-trained model
      #ifdef MYDEBUG
       std::cout << "Loading pre-trained model" << std::endl;
       #endif
      feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

      //string extract_feature_blob_name(argv[++arg_pos]);
    CHECK(feature_extraction_net->has_blob(extract_feature_blob_name3))
          << "Unknown feature blob name " << extract_feature_blob_name3
          << " in the network " << feature_extraction_proto;


    int mem_size=frame.size().width*frame.size().height*3*sizeof(Dtype);
    Dtype *memchunk = (Dtype*) malloc (mem_size);

	    //if (iFrame%30==0 )  // process every XXX frames
	    {
            iFrame2++;
            string imgnum=std::to_string(iFrame2);
            int numchars=imgnum.size();
            //string pre="";
            //for (int iTemp2=0;iTemp2<(8-numchars);iTemp2++)
            //    pre+='0';
            //std::ofstream pool5_out (pool5_dir + "/" + prefix + pre + std::to_string(iFrame) + ".bin" , std::ios::out | std::ios::binary);
	        std::ofstream pool5_out (pool5_dir + "/" + prefix  + ".bin" , std::ios::out | std::ios::binary);

	        #ifdef MYDEBUG
	        std::cout << "Frame: "<< iFrame << std::endl;
            std::cout << "converting to float" << std::endl;
            #endif
            frame.convertTo(frameFloat, CV_32FC3);  //Convert to float

            #ifdef MYDEBUG
            std::cout << "channel correction" << std::endl;
            #endif

            cvtColor(frameFloat, frameFloat, CV_BGR2RGB);   //swap color channels
            Mat myMean = Mat(frameFloat.size().height, frameFloat.size().width, CV_32FC3, Scalar(meanpixel,meanpixel,meanpixel));

            #ifdef MYDEBUG
                std::cout << "Substracting mean" << std::endl;
                std::cout << "frameFloat is " <<  frameFloat.size().width << " - " <<  frameFloat.size().height <<  " - " <<  frameFloat.channels() <<std::endl;
                std::cout << "myMean is " <<  myMean.size().width << " - " <<  myMean.size().height <<  " - " <<  myMean.channels() <<std::endl;
            #endif

            frameFloat -= myMean;//GV_main5

            #ifdef MYDEBUG
            std::cout << "copy proposal to input" << std::endl;
            #endif
            fix_format(frameFloat.clone(), memchunk);   // re-arrange memory positions according to [chanels][width][height]


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

                const shared_ptr<Blob<Dtype> > feature_blob3 = feature_extraction_net->blob_by_name(extract_feature_blob_name3);  // Extract Output layer
                int num_features3 = feature_blob3->num();
                Dtype* feature_blob_data3;

                #ifdef MYDEBUG
                std::cout << "displaying good proposals" << std::endl;
                #endif

                pool5_out.write((char *)feature_blob3->mutable_cpu_data(), feature_blob3->offset(num_features3) *sizeof(Dtype));

              }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

              LOG(ERROR)<< "Successfully extracted the features!";


    //  ***********************************
            //imshow("video",frame);
            //imwrite(prefix + "-" + pre +std::to_string(iFrame2)+".png",frame);
            pool5_out.close();

	}

	free(memchunk);

}





