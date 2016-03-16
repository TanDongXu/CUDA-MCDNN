#include"columnNet.cuh"
#include"config/config.h"
#include"./readData/readNetWork.h"
#include"./layers/dataLayer.h"
#include"./layers/convLayer.h"
#include"./layers/poolLayer.h"
#include"./layers/InceptionLayer.h"
#include"./layers/hiddenLayer.h"
#include"./layers/dropOutLayer.h"
#include"./layers/activationLayer.h"
#include"./layers/LRNLayer.h"
#include"./layers/softMaxLayer.h"
#include"./common/cuMatrixVector.h"
#include"./common/cuMatrix.h"
#include"./common/utility.cuh"
#include<iostream>
#include<time.h>
#include"math.h"

using namespace std;

/*create netWork*/
void creatColumnNet(int sign)
{
	layersBase* baseLayer;
	int layerNum = config::instanceObjtce()->getLayersNum();
	configBase *layer = config::instanceObjtce()->getFirstLayers();

	for(int i=0; i<layerNum; i++)
	{
		if((layer->_type) == "DATA")
		{
			baseLayer = new dataLayer(layer->_name);

		}else if((layer->_type) == "CONV")
		{
			baseLayer = new convLayer (layer->_name, sign);

		}else if((layer->_type == "POOLING"))
		{
			baseLayer = new poolLayer (layer->_name);

		}else if((layer->_type) == "HIDDEN")
		{
			baseLayer = new hiddenLayer(layer->_name, sign);

		}else if((layer->_type) == "SOFTMAX")
		{
			baseLayer = new softMaxLayer(layer->_name);

		}else if((layer->_type) == "ACTIVATION")
		{
			baseLayer = new activationLayer(layer->_name);

		}else if((layer->_type) == "LRN")
		{
			baseLayer = new LRNLayer(layer->_name);

		}else if((layer->_type) == "INCEPTION")
		{
			baseLayer = new InceptionLayer(layer->_name, sign);

		}else if((layer->_type) == "DROPOUT")
		{
			baseLayer = new dropOutLayer(layer->_name);
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


/*train netWork*/
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
void cuTrainNetWork(cuMatrixVector<float> &trainData, 
                    cuMatrix<int>* &trainLabel, 
                    cuMatrixVector<float> &testData,
                    cuMatrix<int>*&testLabel,
		            int batchSize, 
                    int imageSize, 
                    int normalized_width)
{

	cout<<"TestData Forecast The Result..."<<endl;
	predictTestData(testData, testLabel, batchSize);
	cout<<endl;


	cout<<"NetWork training......"<<endl;
	int epochs = config::instanceObjtce()->get_trainEpochs();
	int iter_per_epo = config::instanceObjtce()->get_iterPerEpo();
	int layerNum = Layers::instanceObject()->getLayersNum();
	double nMomentum[]={0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99};
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
