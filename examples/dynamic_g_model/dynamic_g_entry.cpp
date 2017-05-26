/*************************************************************************
	> File Name: dynamic_g_entry.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2017年03月21日 星期二 20时08分52秒
 ************************************************************************/

#include<iostream>
#include<glog/logging.h>
#include<string>
#include<cudnn.h>
#include"examples/mnist/mnist.h"
#include"examples/cifar10/cifar-10.h"
#include"common/cuMatrixVector.h"
#include"common/cuMatrix.h"
#include"readData/readMnistData.h"
#include"readData/readCifar10Data.h"
#include"readData/readCifar100.h"
#include"common/utility.cuh"
#include"columnNet.cuh"

using namespace std;

void dynamic_g_entry()
{
    LOG(INFO) << "You Have Entered New Way To Generate Model, Please Select DataSet.";
    LOG(INFO) << "1. MNIST      2.CIFAR-10      3.CIFAR-100";
    const string dirPath = "profile/dynamic_g_profile/";
    const string sPhase_begin = "dynamic_begin/";
    const string sPhase_end = "dynamic_end/";

    cuMatrixVector<float> trainSetX;
    cuMatrixVector<float> testSetX;
    cuMatrix<int>* trainSetY, *testSetY;

    int batchSize;
    int normalized_width;
    int imageSize;
    // Get end profile
    config* endConfig = new config();

    int cmd;
    cin >> cmd;
    if(1 == cmd)
    {
        const string sMnist_begin_dir = dirPath + sPhase_begin + "MnistConfig.txt";
        const string sMnist_end_dir = dirPath + sPhase_end + "MnistConfig.txt";
        // read the begin profile
        config::instanceObjtce()->initConfig(sMnist_begin_dir);
        normalized_width = config::instanceObjtce()->get_normalizedWidth();
        imageSize = config::instanceObjtce()->get_imageSize();

        //read the end profile
        endConfig->initConfig(sMnist_end_dir);
        // Read Mnist dataSet
	    readMnistData(trainSetX, trainSetY, "data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", normalized_width, imageSize);
        readMnistData(testSetX, testSetY, "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", normalized_width, imageSize);
        LOG(INFO) << "*******************************************************";
        LOG(INFO) << "     Train_set : " << trainSetX[0]->rows << " x " << trainSetX[0]->cols << " features and " << trainSetX.size() << " samples";
        LOG(INFO) << "   Train_label :   " << trainSetY->cols << " x " << trainSetY->cols << " features and " << trainSetY->rows << " samples";
        LOG(INFO) << "      Test_set : " << testSetX[0]->rows << " x " << testSetX[0]->cols << " features and " <<  testSetX.size() << " samples";
        LOG(INFO) << "    Test_label :   " << testSetY->cols << " x " << testSetY->cols << " features and " <<  testSetY->rows  << " samples";
        LOG(INFO) << "*******************************************************";
    
    }
    else if(2 == cmd)
    {
        const string sCifar10_begin_dir = dirPath + sPhase_begin + "Cifar10Config.txt";
        const string sCifar10_end_dir = dirPath + sPhase_end + "Cifar10Config.txt";
        //read the begin profile
        config::instanceObjtce()->initConfig(sCifar10_begin_dir);
        //read the end profile
        endConfig->initConfig(sCifar10_end_dir);
        //read Cifar-10 DataSet
        read_Cifar10_Data(trainSetX, testSetX, trainSetY, testSetY);

	    LOG(INFO) << "*******************************************************";;
        LOG(INFO) << "     Train_set : " << trainSetX[0]->rows << " x " << trainSetX[0]->cols << " features and " << trainSetX.size() << " samples";
        LOG(INFO) << "   Train_label :   " << trainSetY->cols << " x " << trainSetY->cols << " features and " << trainSetY->rows << " samples";
        LOG(INFO) << "      Test_set : " << testSetX[0]->rows << " x " << testSetX[0]->cols << " features and " <<  testSetX.size() << " samples";
        LOG(INFO) << "    Test_label :   " << testSetY->cols << " x " << testSetY->cols << " features and " <<  testSetY->rows  << " samples";
	    LOG(INFO) << "*******************************************************";
    }
    else if(3 == cmd)
    {
        const string sCifar10_begin_dir = dirPath + sPhase_begin + "Cifar100Config.txt";
        const string sCifar10_end_dir = dirPath + sPhase_end + "Cifar100Config.txt";
        //read the begin profile
        config::instanceObjtce()->initConfig(sCifar10_begin_dir);
        //read the end profile
        endConfig->initConfig(sCifar10_end_dir);
        //read Cifar-10 DataSet
        readCifar100Data(trainSetX, testSetX, trainSetY, testSetY);

	    LOG(INFO) << "*******************************************************";;
        LOG(INFO) << "     Train_set : " << trainSetX[0]->rows << " x " << trainSetX[0]->cols << " features and " << trainSetX.size() << " samples";
        LOG(INFO) << "   Train_label :   " << trainSetY->cols << " x " << trainSetY->cols << " features and " << trainSetY->rows << " samples";
        LOG(INFO) << "      Test_set : " << testSetX[0]->rows << " x " << testSetX[0]->cols << " features and " <<  testSetX.size() << " samples";
        LOG(INFO) << "    Test_label :   " << testSetY->cols << " x " << testSetY->cols << " features and " <<  testSetY->rows  << " samples";
	    LOG(INFO) << "*******************************************************";
    }else
    LOG(FATAL) << "DataSet Select Error.";
	    
    int cuda_version = cudnnGetVersion();
	LOG(INFO) << "cudnnGetVersion(): " << cuda_version << " CUDNN VERSION from cudnn.h: "<< CUDNN_VERSION;
	/*show the device information*/
	showDevices();
	// Create network 
    creatColumnNet(1);
    //Training Network
	dynamic_g_trainNet(trainSetX, trainSetY, testSetX, testSetY, endConfig);

}
