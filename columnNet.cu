#include"columnNet.cuh"
#include"config/config.h"
#include"./readData/readNetWork.h"
#include"./layers/dataLayer.h"
#include"./layers/convLayer.h"
#include"./layers/poolLayer.h"
#include"./layers/InceptionLayer.h"
#include"./layers/hiddenLayer.h"
#include"./layers/activationLayer.h"
#include"./layers/LRNLayer.h"
#include"./layers/softMaxLayer.h"
#include"./common/cuMatrixVector.h"
#include"./common/cuMatrix.h"
#include"./common/utility.h"


#include<iostream>
#include<time.h>
#include"math.h"

using namespace std;

const double FLAGS_lr_gamma = 0.0001;   //learning rate policy
const double FLAGS_lr_power = 0.75;     //Learing rate policy power

void creatColumnNet(int sign)
{
	layersBase* baseLayer;
	/*get the number of layers*/
	int layerNum = config::instanceObjtce()->getLayersNum();

	/*get the first layer*/
	configBase *layer = config::instanceObjtce()->getFirstLayers();

	for(int i=0; i<layerNum; i++)
	{
		//cout<<layer->_name<<endl;
		if((layer->_type) == "DATA")
		{
			//configData* data = (configData*) layer;
			baseLayer = new dataLayer(layer->_name);

		}else if((layer->_type) == "CONV")
		{
			//configConv* conv = (configConv*)layer;
			baseLayer = new convLayer (layer->_name, sign);

		}else if((layer->_type == "POOLING"))
		{
			//configPooling* pooling = (configPooling*)layer;
			baseLayer = new poolLayer (layer->_name);

		}else if((layer->_type) == "HIDDEN")
		{
			//configHidden* hidden = (configHidden*)layer;
			baseLayer = new hiddenLayer(layer->_name, sign);

		}else if((layer->_type) == "SOFTMAX")
		{
			//configSoftMax* softmax = (configSoftMax*) layer;
			baseLayer = new softMaxLayer(layer->_name);

		}else if((layer->_type) == "ACTIVATION")
		{
			//configActivation* activation = (configActivation*)layer;
			baseLayer = new activationLayer(layer->_name);

		}else if((layer->_type) == "LRN")
		{
			//configLRN* LRN = (configLRN*)layer;
			baseLayer = new LRNLayer(layer->_name);

		}else if((layer->_type) == "INCEPTION")
		{
			baseLayer = new InceptionLayer(layer->_name, sign);
		}

		Layers::instanceObject()->storLayers(layer->_name, baseLayer);
		layer=layer->_next;
	}

	if(sign == READ_FROM_FILE) readNetWork();
}




/*predict the result*/
void resultPredict(string train_or_test)
{
	int size = Layers::instanceObject()->getLayersNum();
	configBase * config = (configBase*) config::instanceObjtce()->getFirstLayers();
	for(int i=0; i<size;i++)
	{
		layersBase * layer = (layersBase*)Layers::instanceObject()->getLayer(config->_name);
		layer->forwardPropagation(train_or_test);
		//cout<<layer->_name<<endl;
		if(train_or_test == "test")
		{
			layer->Forward_cudaFree();
		}
		config = config->_next;
	}
}

/*test netWork*/
void predictTestData(cuMatrixVector<float>&testData, cuMatrix<int>* &testLabel, int batchSize)
{
	dataLayer* datalayer = static_cast<dataLayer*>( Layers::instanceObject()->getLayer("data"));

	for(int i=0;i<(testData.size()+batchSize)/batchSize;i++)
    {
		datalayer->getBatch_Images_Label(i , testData, testLabel);
		resultPredict("test");

    }

}



void getNetWorkCost(float&Momentum)
{
	resultPredict("train");

	int layerNum = Layers::instanceObject()->getLayersNum();
	configBase * config = (configBase*) config::instanceObjtce()->getLastLayer();
	for(int num=0; num<layerNum;num++)
	{
		layersBase * layer = (layersBase*)Layers::instanceObject()->getLayer(config->_name);
		layer->backwardPropagation(Momentum);
		layer->Backward_cudaFree();
		config = config->_prev;

	}

}


/*training netWork*/
void cuTrainNetWork(cuMatrixVector<float> &trainData, cuMatrix<int>* &trainLabel, cuMatrixVector<float> &testData, cuMatrix<int>*&testLabel,
		           int batchSize, int imageSize, int normalized_width)
{

	cout<<"TestData Forecast The Result..."<<endl;
	predictTestData(testData, testLabel, batchSize);
	cout<<endl;


	cout<<"NetWork training......"<<endl;
	int epochs = config::instanceObjtce()->get_trainEpochs();
	int iter_per_epo = config::instanceObjtce()->get_iterPerEpo();
	int layerNum = Layers::instanceObject()->getLayersNum();
	double nMomentum[]={0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99};//调整动量
	int epoCount[]={80,80,80,80,80,80,80,80,80,80};
	float Momentum = 0.9;
	int id = 0;

	clock_t start, stop;
	double runtime;

	start = clock();
	for(int epo = 0; epo < epochs; epo++)
	{
		dataLayer* datalayer = static_cast<dataLayer*>(Layers::instanceObject()->getLayer("data"));

		Momentum = nMomentum[id];

		clock_t inStart, inEnd;
		inStart = clock();
		/*train network*/
		for(int iter = 0 ; iter < iter_per_epo; iter++)
		{
			datalayer->RandomBatch_Images_Label(trainData,trainLabel);
			getNetWorkCost(Momentum);
		}

        inEnd = clock();

        //adjust learning rate
		configBase * config = (configBase*) config::instanceObjtce()->getFirstLayers();
		for(int i=0; i<layerNum;i++)
		{
			layersBase * layer = (layersBase*)Layers::instanceObject()->getLayer(config->_name);
			layer->adjust_learnRate(epo, FLAGS_lr_gamma, FLAGS_lr_power);
			config = config->_next;
		}

		if(epo && epo % epoCount[id] == 0)
		{
			id++;
			if(id>9) id=9;
		}

		/*test network*/
		cout<<"epochs: "<<epo<<" ,Time: "<<(inEnd - inStart)/CLOCKS_PER_SEC<<"s,";
		predictTestData(testData, testLabel, batchSize);
		cout<<" ,Momentum: "<<Momentum<<endl;

	}

	stop = clock();
	runtime = stop - start;
	cout<< epochs <<" epochs total rumtime is: "<<runtime /CLOCKS_PER_SEC<<" Seconds"<<endl;

}
