#include"cifar-10.h"

void runCifar10()
{
	cuMatrixVector<float>  trainSetX;
	cuMatrixVector<float>  testSetX;
	cuMatrix<int> *trainSetY, *testSetY;

	int batchSize;
	/*read the layers config*/
	config::instanceObjtce()->initConfig("profile/Cifar10Config.txt");
	batchSize = config::instanceObjtce()->get_batchSize();

	/*read the cifar10 data*/
	read_Cifar10_Data(trainSetX, testSetX, trainSetY, testSetY);

	 cout<<"*******************************************************"<<endl;
	 cout<<"     Train_set : "<< trainSetX[0]->rows * trainSetX[0]->cols * trainSetX[0]->channels
			                  <<" features and "<< trainSetX.size() <<" samples"<<endl;
	 cout<<"   Train_label :   "<< trainSetY->cols                       <<"  features and "<< trainSetY->rows  <<" samples"<<endl;
	 cout<<"      Test_set : "<< testSetX[0]->rows * testSetX[0]->cols  * testSetX[0]->channels
			                  <<" features and "<<  testSetX.size() <<" samples"<<endl;
	 cout<<"    Test_label :   "<< testSetY->cols                        <<"  features and "<<  testSetY->rows  <<" samples"<<endl;
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
