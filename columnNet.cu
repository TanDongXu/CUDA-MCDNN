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
#include "./layers/voteLayer.h"
#include"./common/cuMatrixVector.h"
#include"./common/cuMatrix.h"
#include"./common/utility.cuh"
#include"./composite/NodeFission.h"
#include<iostream>
#include<time.h>
#include <queue>
#include <set>
#include"math.h"
#include<algorithm>

const bool DFS_TRAINING = true;
const bool DFS_TEST = true;
const bool FISS_TRAINING = false;

using namespace std;

/*create netWork*/
void creatColumnNet(int sign)
{
    layersBase* baseLayer;
    int layerNum = config::instanceObjtce()->getLayersNum();
    configBase* layer = config::instanceObjtce()->getFirstLayers();
    queue<configBase*>que;
    que.push(layer);
    set<configBase*>hash;
    hash.insert( layer );
    while(!que.empty()){
        layer = que.front();
        que.pop();
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

        Layers::instanceObject()->storLayers(layer->_input, layer->_name, baseLayer);
        for(int i = 0; i < layer->_next.size(); i++){
            if( hash.find( layer->_next[i] ) == hash.end()){
                hash.insert( layer->_next[i] );
                que.push( layer->_next[i]);
            }
        }
    }

    if(sign == READ_FROM_FILE) readNetWork();
}

/*predict the result*/
void resultPredict(string train_or_test)
{
    int size = Layers::instanceObject()->getLayersNum();
    configBase* config = (configBase*) config::instanceObjtce()->getFirstLayers();
    queue<configBase*>que;
    que.push(config);
    set<configBase*>hash;
    hash.insert(config);
    while(!que.empty()){
        config = que.front();
        que.pop();
        layersBase* layer = (layersBase*)Layers::instanceObject()->getLayer(config->_name);
        layer->forwardPropagation(train_or_test);
        for(int i = 0; i < config->_next.size(); i++){
            if( hash.find( config->_next[i] ) == hash.end()){
                hash.insert( config->_next[i] );
                que.push( config->_next[i] );
            }
        }
    }
}

float dfsGetLearningRateReduce(configBase* config){
    layersBase* layer = (layersBase*)Layers::instanceObject()->getLayer(config->_name);
    if(config->_next.size() == 0){
        layer->setRateReduce( 1 );
        return 1;
    }

    float fRateReduce = 0;
    for(int i = 0; i < config->_next.size(); i++){
        fRateReduce += dfsGetLearningRateReduce( config->_next[i] );
    }

    layer->setRateReduce( fRateReduce );
    printf("rate %f\n", layer->getRateReduce()); 

    return fRateReduce;
}

/*test netWork*/
void predictTestData(cuMatrixVector<float>&testData, cuMatrix<int>* &testLabel, int batchSize)
{
    dataLayer* datalayer = static_cast<dataLayer*>( Layers::instanceObject()->getLayer("data"));

    for(int i=0;i<( testData.size() + batchSize - 1)/batchSize;i++)
    {
        datalayer->getBatch_Images_Label(i , testData, testLabel);
        resultPredict("test");
    }
}

/*linear structure training network*/
void getNetWorkCost(float&Momentum)
{
    resultPredict("train");

    configBase* config = (configBase*) config::instanceObjtce()->getLastLayer();
    queue<configBase*>que;
    que.push(config);
    set<configBase*>hash;
    hash.insert(config);
    while(!que.empty()){
        config = que.front();
        que.pop();
        layersBase* layer = (layersBase*)Layers::instanceObject()->getLayer(config->_name);
        layer->backwardPropagation(Momentum);

        for(int i = 0; i < config->_prev.size(); i++){
            if( hash.find( config->_prev[i] ) == hash.end()){
                hash.insert(config->_prev[i]);
                que.push(config->_prev[i]);
            }
        }
    }
}

std::vector<configBase*> g_vQue;
//std::map<layersBase*, size_t> g_vFissNode;
std::vector<softMaxLayer*> g_vBranchResult;
//int g_nMinCorrSize;

/* voting */
void dfsResultPredict( configBase* config, cuMatrixVector<float>& testData, cuMatrix<int>*& testLabel, int nBatchSize)
{
    g_vQue.push_back( config );
    if( config->_next.size() == 0 ){
        //printf("%s\n", config->_name.c_str());

        dataLayer* datalayer = static_cast<dataLayer*>( Layers::instanceObject()->getLayer("data"));

        for(int i = 0; i < (testData.size() + nBatchSize - 1) / nBatchSize; i++)
        {
            datalayer->getBatch_Images_Label(i , testData, testLabel);
            for(int j = 0; j < g_vQue.size(); j++)
            {
                layersBase* layer = (layersBase*)Layers::instanceObject()->getLayer(g_vQue[j]->_name);
                layer->forwardPropagation("test");
//                if(i == 0)
//                {
//                	cout<<layer->_name<<endl;
//                }
                // is softmax, then vote
                if( j == g_vQue.size() - 1 ){
                    VoteLayer::instance()->vote( i , nBatchSize, layer->dstData );
                }
            }
        }
    }

    for(int i = 0; i < config->_next.size(); i++){
        configBase* tmpConfig = config->_next[i];
        layersBase* layer = (layersBase*)Layers::instanceObject()->getLayer( config->_name );
        layer->setCurBranchIndex(i);
        dfsResultPredict( tmpConfig, testData, testLabel, nBatchSize );
    }
    g_vQue.pop_back();
}

void dfsTraining(configBase* config, float nMomentum, cuMatrixVector<float>& trainData, cuMatrix<int>* &trainLabel, int& iter)
{
    g_vQue.push_back(config);

    /*如果是一个叶子节点*/
    if (config->_next.size() == 0){
        dataLayer* datalayer = static_cast<dataLayer*>(Layers::instanceObject()->getLayer("data"));
        datalayer->RandomBatch_Images_Label(trainData, trainLabel);

        for(int i = 0; i < g_vQue.size(); i++){
            //printf("f %d %s\n", i, g_vQue[i]->_name.c_str());
            layersBase* layer = (layersBase*)Layers::instanceObject()->getLayer(g_vQue[i]->_name);
            layer->forwardPropagation( "train" );
        }

        for( int i = g_vQue.size() - 1; i>= 0; i--){
            layersBase* layer = (layersBase*)Layers::instanceObject()->getLayer(g_vQue[i]->_name);
           // if(layer->getRateReduce() > 1e-4){
              layer->backwardPropagation( nMomentum );
          //  }
           // else{
          //      break;
         //   }
        }
    }
    /*如果不是叶子节点*/
    for(int i = 0; i < config->_next.size(); i++){
        configBase* tmpConfig = config->_next[i];
        layersBase* layer = (layersBase*)Layers::instanceObject()->getLayer( config->_name );
        layer->setCurBranchIndex(i);
        dfsTraining( tmpConfig, nMomentum, trainData, trainLabel, iter);
    }
    g_vQue.pop_back();
}

//ascend order
bool cmp_ascend_Order( softMaxLayer* a, softMaxLayer* b)
{
	return (a->getCorrectNum()) < (b->getCorrectNum());
}

/*get min result branch*/
void getBranchResult(layersBase*curLayer)
{
	//叶子节点
	if (curLayer->nextLayer.size() == 0)
	{
		softMaxLayer* tmp = (softMaxLayer*) curLayer;
		g_vBranchResult.push_back(tmp);
	}

	for (int i = 0; i < curLayer->nextLayer.size(); i++) {
		layersBase* tmpLayer = curLayer->nextLayer[i];
		getBranchResult(tmpLayer);
	}
//	//叶子节点
//	if (curLayer->nextLayer.size() == 0)
//	{
//		softMaxLayer* tmp = (softMaxLayer*)curLayer;
//		if(tmp->getCorrectNum() < g_nMinCorrSize)
//		{
//			g_nMinCorrSize = tmp->getCorrectNum();
//			g_vMinBranch.push_back(tmp);
//		}
//	}
//
//	for(int i = 0; i < curLayer->nextLayer.size(); i++)
//	{
//		layersBase* tmpLayer =  curLayer->nextLayer[i];
//		ascend_OrderBranch(tmpLayer);
//	}
}


/*get Fissnode and Fission*/
void performFiss()
{
	for(int i = 0; i < g_vBranchResult.size(); i++)
	{
		layersBase* tmpCur = (layersBase*)g_vBranchResult[i];

		while (tmpCur->prevLayer[0]->_name != string("data") && tmpCur->prevLayer[0]->nextLayer.size() == 1)
		{
			tmpCur = tmpCur->prevLayer[0];
		}
		//if curBranch is Fiss to data layer, then fiss another
		if (tmpCur->prevLayer[0]->_name== "data" && (i != g_vBranchResult.size() - 1))
			continue;
		else if(i == g_vBranchResult.size() - 1)
		{
			softmaxFission(g_vBranchResult[0]);
			break;
		}
		else
		{
			//Fission one Node every time
			//cout<<tmpCur->prevLayer[0]->_name<<endl;
			NodeFission(tmpCur->prevLayer[0], tmpCur);
			break;
		}
	}
}

/*training netWork*/
void cuTrainNetWork(cuMatrixVector<float> &trainData, 
        cuMatrix<int>* &trainLabel, 
        cuMatrixVector<float> &testData,
        cuMatrix<int>*&testLabel,
        int batchSize
        )
{
    //configBase* config = (configBase*) config::instanceObjtce()->getFirstLayers();
    //dfsGetLearningRateReduce( config );
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
        configBase* config = (configBase*) config::instanceObjtce()->getFirstLayers();
        if( DFS_TRAINING == false ){
            /*train network*/
            for(int iter = 0 ; iter < iter_per_epo; iter++)
            {
                datalayer->RandomBatch_Images_Label(trainData, trainLabel);
                getNetWorkCost(Momentum);
            }
        }
        else{
            //printf("error\n");
            int iter = 0;
            g_vQue.clear();
            while(iter < iter_per_epo){
                dfsTraining(config, Momentum, trainData, trainLabel, iter);
                iter++;
            }
        }

        inEnd = clock();

        config = (configBase*) config::instanceObjtce()->getFirstLayers();
        //adjust learning rate
        queue<configBase*> que;
        set<configBase*> hash;
        hash.insert(config);
        que.push(config);
        while( !que.empty() ){
            config = que.front();
            que.pop();
            layersBase * layer = (layersBase*)Layers::instanceObject()->getLayer(config->_name);
            layer->adjust_learnRate(epo, FLAGS_lr_gamma, FLAGS_lr_power);

            for(int i = 0; i < config->_next.size(); i++){
                if( hash.find(config->_next[i]) == hash.end()){
                    hash.insert(config->_next[i]);
                    que.push(config->_next[i]);
                }
            }
        }
//        if( epo% 50 == 0 && epo != 0 ){
//            config = (configBase*) config::instanceObjtce()->getFirstLayers();
//            //adjust learning rate
//            queue<configBase*> que;
//            set<configBase*> hash;
//            hash.insert(config);
//            que.push(config);
//            while( !que.empty() ){
//                config = que.front();
//                que.pop();
//                layersBase * layer = (layersBase*)Layers::instanceObject()->getLayer(config->_name);
//                layer->rateReduce();
//                /*
//                if( layer->lrate >= 1e-4 && layer->lrate <= 1){
//                    printf("lRate %s %f\n", layer->_name.c_str(), layer->lrate);
//                }
//                */
//
//                for(int i = 0; i < config->_next.size(); i++){
//                    if( hash.find(config->_next[i]) == hash.end()){
//                        hash.insert(config->_next[i]);
//                        que.push(config->_next[i]);
//                    }
//                }
//            }
//        }

        if(epo && epo % epoCount[id] == 0)
        {
            id++;
            if(id>9) id=9;
        }

        /*test network*/
       cout<<"epochs: "<<epo<<" ,Time: "<<(inEnd - inStart)/CLOCKS_PER_SEC<<"s,";
        if( DFS_TEST == false){
            predictTestData( testData, testLabel, batchSize );
        }
        else{
            VoteLayer::instance()->clear();
            static float fMax = 0;
            configBase* config = (configBase*) config::instanceObjtce()->getFirstLayers();
            dfsResultPredict(config, testData, testLabel, batchSize);
            float fTest = VoteLayer::instance()->result();
            if ( fMax < fTest ) fMax = fTest;
            printf(" test_result %f/%f ", fTest, fMax);
        }
        cout<<" ,Momentum: "<<Momentum<<endl;

        if (DFS_TRAINING == true && FISS_TRAINING == true )
        {
			if ((epo < 30 && ((epo + 1) % 15) == 0) || (epo >= 30 && ((epo + 1) % 10) == 0)) {
				//g_vFissNode.clear();
				g_vBranchResult.clear();
				layersBase* curLayer = Layers::instanceObject()->getLayer("data");
				//dataLayer* tmpLayer = (dataLayer*) curLayer;
				//g_nMinCorrSize = tmpLayer->getDataSize();
				getBranchResult(curLayer);
				sort(g_vBranchResult.begin(), g_vBranchResult.end(), cmp_ascend_Order);
//				vector<softMaxLayer*> ::iterator it;
//				for(it  = g_vBranchResult.begin(); it != g_vBranchResult.end(); it++)
//				{
//					cout<<(*it)->_name<<endl;
//				}
				//ascending order softmax result

				performFiss();
			}
		}

    }

    stop = clock();
    runtime = stop - start;
    cout<< epochs <<" epochs total rumtime is: "<<runtime /CLOCKS_PER_SEC<<" Seconds"<<endl;
    }
