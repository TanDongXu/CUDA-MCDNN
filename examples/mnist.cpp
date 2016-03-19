#include"mnist.h"



void runMnist()
{
	cuMatrixVector<float> trainSetX;
	cuMatrixVector<float> testSetX;

	cuMatrix<int>* trainSetY, *testSetY;

	int batchSize;
	int normalized_width;
	int imageSize;


	/*read the layers configure*/
	config::instanceObjtce()->initConfig("profile/MnistConfig.txt");
	batchSize = config::instanceObjtce()->get_batchSize();
	normalized_width = config::instanceObjtce()->get_normalizedWidth();
	imageSize = config::instanceObjtce()->get_imageSize();

	/*read Mnist dataSet*/
	readMnistData(trainSetX, trainSetY, "data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", normalized_width, imageSize);
    readMnistData(testSetX, testSetY, "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", normalized_width, imageSize);
    cout<<"*******************************************************"<<endl;
    cout<<"     Train_set : "<< trainSetX[0]->rows * trainSetX[0]->cols <<" features and "<< trainSetX.size() <<" samples"<<endl;
    cout<<"   Train_label :   "<< trainSetY->cols                       <<" features and "<< trainSetY->rows  <<" samples"<<endl;
    cout<<"      Test_set : "<< testSetX[0]->rows * testSetX[0]->cols   <<" features and "<<  testSetX.size() <<" samples"<<endl;
    cout<<"    Test_label :   "<< testSetY->cols                        <<" features and "<<  testSetY->rows  <<" samples"<<endl;
    cout<<"*******************************************************"<<endl;


    int version = cudnnGetVersion();
    cout<<"cudnnGetVersion(): "<<version<<" CUDNN VERSION from cudnn.h: "<<CUDNN_VERSION<<endl;
    /*show the device information*/
    showDevices();

    cout<<endl<<endl<<"Select the way to initial Parameter: "<<endl<<"1.random   2.read from file"<<endl;
    int cmd;
    cin>> cmd;
    if(cmd == 1 || cmd == 2)
    	creatColumnNet(cmd);
    else
    {
    	cout<<"Init way input Error"<<endl;
        exit(0);
    }

    /*training Network*/
    cuTrainNetWork(trainSetX, trainSetY, testSetX, testSetY, batchSize);

}
