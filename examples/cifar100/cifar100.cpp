#include"cifar100.h"
#include"layers/VoteLayer.h"
#include"config/config.h"
#include"common/cuMatrixVector.h"
#include"common/cuMatrix.h"
#include"columnNet.cuh"
#include"common/utility.cuh"
#include"readData/readCifar100.h"

#include<cudnn.h>
#include<glog/logging.h>

void runCifar100()
{
	cuMatrixVector<float>  trainSetX;
	cuMatrixVector<float>  testSetX;
	cuMatrix<int> *trainSetY, *testSetY;

	int batchSize;
	// Read the layers config
	config::instanceObjtce()->initConfig("profile/Cifar100Config.txt");
	batchSize = config::instanceObjtce()->get_batchSize();

	// Read the cifar10 data
	readCifar100Data(trainSetX, testSetX, trainSetY, testSetY);

	LOG(INFO) <<"*******************************************************";
	LOG(INFO) <<"     Train_set : " << trainSetX[0]->rows * trainSetX[0]->cols * trainSetX[0]->channels << " features and " << trainSetX.size() << " samples";
	LOG(INFO) <<"   Train_label :   " << trainSetY->cols             << "  features and " << trainSetY->rows  << " samples";
	LOG(INFO) <<"      Test_set : " << testSetX[0]->rows * testSetX[0]->cols * testSetX[0]->channels << " features and " <<  testSetX.size() << " samples";
	LOG(INFO) <<"    Test_label :   " << testSetY->cols << " * " << testSetY->rows <<"  features and "<<  testSetY->rows  <<" samples";
	LOG(INFO) <<"*******************************************************";

    VoteLayer::instance()->init( testSetY->rows, 100, testSetY );

	int version = cudnnGetVersion();
	LOG(INFO) << "cudnnGetVersion(): " << version << " CUDNN VERSION from cudnn.h: " << CUDNN_VERSION;
	// Show the device information
	showDevices();

	cout << endl << endl;
    LOG(INFO) << "Select The Way To Initial Parameter: ";
    LOG(INFO) << "1.Random   2.Read from file";
	int cmd;
	cin>> cmd;
	if(cmd == 1 || cmd == 2) creatColumnNet(cmd);
	else
	    LOG(FATAL) << "Init Way Input Error";

	// Training Network
	cuTrainNetWork(trainSetX, trainSetY, testSetX, testSetY, batchSize);
}
