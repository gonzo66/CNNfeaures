#include "kyheader.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"
#include "CmFile.h"

#include <string>
#include <vector>
//using namespace cv;
//using libv4l2;

#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#define Dtype  float


using namespace caffe;  // NOLINT(build/namespaces)



int main(int argc, char* argv[])
{
    string default_dir=CmFile::GV_pwd();

    std::cout << "Hello world" << default_dir<<  std::endl ;
	VideoCapture cap;
	std::cout << "Video Capture" << std::endl;

    //examples/imagenet/caffe_reference_imagenet_model examples/_temp/imagenet_val.prototxt fc7 examples/_temp/features 10
    std::string pretrained_binary_proto="/home/gonzalo/Downloads/caffe/caffe-master/examples/imagenet/caffe_reference_imagenet_model" ;
    std::string pretrained_binary_proto="/home/gonzalo/Downloads/caffe/caffe-master/examples/imagenet/imagenet_deploy.prototxt";"
    std::string feature_extraction_proto="/home/gonzalo/Downloads/caffe/caffe-master/examples/_temp/imagenet_val.prototxt" ;
    std::string extract_feature_blob_name="prob" ;
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

      shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto));
      feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

      //string extract_feature_blob_name(argv[++arg_pos]);
      CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
          << "Unknown feature blob name " << extract_feature_blob_name
          << " in the network " << feature_extraction_proto;


	//if(argc >1)
	{
	    string filename="/home/gonzalo/Downloads/bing/BingObjectnessCVPR14/Objectness-master/Src/P03.mp4";
	    //string filename=string(argv[1]));
		cap.open(filename);
		if(!cap.isOpened())
		{
			std::cout << "Can not open the file " <<string(argv[1]) << std::endl;
			return -1;
		}

	}
/*	else
	{
		cap.open(0);
	}

	if(argc >2)
	{
        default_dir=string(argv[2]);
	}*/
	double fps=cap.get(CV_CAP_PROP_FPS);
	std::cout << "Frames per second" << fps << std::endl ;
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
	for(int iFrame=0;iFrame<10000;iFrame++)    // Loop over 10000 frames or until the end of video
	{
		std::cout << iFrame << std::endl;
		cap >> frame; if(!frame.data) break;

          int num_mini_batches = 1;
          LOG(ERROR)<< "Extacting Features";

          Datum datum;
          //leveldb::WriteBatch* batch = new leveldb::WriteBatch();
          const int kMaxKeyStrLength = 100;
          char key_str[kMaxKeyStrLength];
          int num_bytes_of_binary_code = sizeof(Dtype);
          vector<Blob<float>*> input_vec;
          int image_index = 0;

          for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
            feature_extraction_net->Forward(input_vec);
            const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(extract_feature_blob_name);
            int num_features = feature_blob->num();
            int dim_features = feature_blob->count() / num_features;
            Dtype* feature_blob_data;
            for (int n = 0; n < num_features; ++n) {
              datum.set_height(dim_features);
              datum.set_width(1);
              datum.set_channels(1);
              datum.clear_data();
              datum.clear_float_data();
              feature_blob_data = feature_blob->mutable_cpu_data() +
                  feature_blob->offset(n);
              for (int d = 0; d < dim_features; ++d) {
                datum.add_float_data(feature_blob_data[d]);
              }
              string value;
              datum.SerializeToString(&value);
              snprintf(key_str, kMaxKeyStrLength, "%d", image_index);
              std::cout << value << std::endl;
              //batch->Put(string(key_str), value);
              ++image_index;
              if (image_index % 1000 == 0) {
              //  db->Write(leveldb::WriteOptions(), batch);
                //LOG(ERROR)<< "Extracted features of " << image_index << " query images.";
                std::cout<< "Extracted features of " << image_index <<
                    " query images.";
              //  delete batch;
              //  batch = new leveldb::WriteBatch();
              }
            }  // for (int n = 0; n < num_features; ++n)
          }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
          // write the last batch
          if (image_index % 1000 != 0) {
         //   db->Write(leveldb::WriteOptions(), batch);
            LOG(ERROR)<< "Extracted features of " << image_index <<
                " query images.";
          }

          LOG(ERROR)<< "Successfully extracted the features!";


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
			    //Rect r(boxesTests[j][0]-1,boxesTests[j][1]-1, boxesTests[j][2]-1,boxesTests[j][3]-1 );
			    //Mat candidate =dst()

				rectangle(frame,Point(boxesTests[j][0]-1,boxesTests[j][1]-1) , Point(boxesTests[j][2]-1,boxesTests[j][3]-1) , colors[num_BB],3);
				std::cout << "BB " <<  j << " (" << boxesTests[j][0] << "," << boxesTests[j][1] << "," << boxesTests[j][2] << "," << boxesTests[j][3] << ")" << std::endl;
				num_BB=num_BB+1;
			}
		}

		std::cout << "Frame "<< iFrame << " with " << boxesTests.size() << " BBs has " << num_BB << std::endl;

/*		_boxesTests[i].resize(boxesTests[i].size());
		for (int j = 0; j < boxesTests[i].size(); j++)
			_boxesTests[i][j] = boxesTests[i][j];*/

//  ***********************************
		imshow("video",frame);   if(waitKey(1) >=0) break;
	}

}





