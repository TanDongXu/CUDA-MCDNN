#include"columnNet.cuh"
#include"config/config.h"
#include"./readData/readNetWork.h"
#include"./layers/DataLayer.h"
#include"./layers/ConvLayer.h"
#include"./layers/PoolLayer.h"
#include"./layers/InceptionLayer.h"
#include"./layers/HiddenLayer.h"
#include"./layers/DropOutLayer.h"
#include"./layers/ActivationLayer.h"
#include"./layers/LRNLayer.h"
#include"./layers/SoftMaxLayer.h"
#include "./layers/VoteLayer.h"
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
#include<glog/logging.h>

const bool DFS_TRAINING = false;
const bool DFS_TEST = false;
const bool FISS_TRAINING = false;

using namespace std;

void globalRateAdjust(int epoch);
/*create netWork*/
void creatColumnNet(int sign)
{
    LOG(INFO) << "All Layers Initialization:";
    LOG(INFO) << "LR_POLICY: " << config::instanceObjtce()->get_lrPolicy();
    LayersBase* baseLayer;
    int layerNum = config::instanceObjtce()->getLayersNum();
    configBase* layer = config::instanceObjtce()->getFirstLayers();
    queue<configBase*>que;
    que.push(layer);
    set<configBase*>hash;
    hash.insert( layer );
    while(!que.empty()){
        layer = que.front();
        que.pop();
        if(string("DATA") == (layer->_type))
        {
            baseLayer = new DataLayer(layer->_name);
        }else if(string("CONV") == (layer->_type))
        {
            baseLayer = new ConvLayer (layer->_name, sign);
        }else if(string("POOLING") == (layer->_type))
        {
            baseLayer = new PoolLayer (layer->_name);
        }else if(string("HIDDEN") == (layer->_type))
        {
            baseLayer = new HiddenLayer(layer->_name, sign);
        }else if(string("SOFTMAX") == (layer->_type))
        {
            baseLayer = new SoftMaxLayer(layer->_name);
        }else if(string("ACTIVATION") == (layer->_type))
        {
            baseLayer = new ActivationLayer(layer->_name);
        }else if(string("LRN") == (layer->_type))
        {
            baseLayer = new LRNLayer(layer->_name);
        }else if(string("INCEPTION") == (layer->_type))
        {
            baseLayer = new InceptionLayer(layer->_name, sign);
        }else if(string("DROPOUT") == (layer->_type))
        {
            baseLayer = new DropOutLayer(layer->_name);
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
        LayersBase* layer = (LayersBase*)Layers::instanceObject()->getLayer(config->_name);
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
    LayersBase* layer = (LayersBase*)Layers::instanceObject()->getLayer(config->_name);
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
    DataLayer* datalayer = static_cast<DataLayer*>( Layers::instanceObject()->getLayer("data"));

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
        LayersBase* layer = (LayersBase*)Layers::instanceObject()->getLayer(config->_name);
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
//std::map<LayersBase*, size_t> g_vFissNode;
std::vector<SoftMaxLayer*> g_vBranchResult;
//int g_nMinCorrSize;
/*多少次迭代进行分裂*/
int g_nSplitIndex = 15;
int g_nCount = 0;

/* voting */
void dfsResultPredict( configBase* config, cuMatrixVector<float>& testData, cuMatrix<int>*& testLabel, int nBatchSize)
{
    g_vQue.push_back( config );
    if( config->_next.size() == 0 ){

        DataLayer* datalayer = static_cast<DataLayer*>( Layers::instanceObject()->getLayer("data"));

        for(int i = 0; i < (testData.size() + nBatchSize - 1) / nBatchSize; i++)
        {
            datalayer->getBatch_Images_Label(i , testData, testLabel);
            for(int j = 0; j < g_vQue.size(); j++)
            {
                LayersBase* layer = (LayersBase*)Layers::instanceObject()->getLayer(g_vQue[j]->_name);
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
        LayersBase* layer = (LayersBase*)Layers::instanceObject()->getLayer( config->_name );
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
        DataLayer* datalayer = static_cast<DataLayer*>(Layers::instanceObject()->getLayer("data"));
        datalayer->RandomBatch_Images_Label(trainData, trainLabel);

        for(int i = 0; i < g_vQue.size(); i++){
            //printf("f %d %s\n", i, g_vQue[i]->_name.c_str());
            LayersBase* layer = (LayersBase*)Layers::instanceObject()->getLayer(g_vQue[i]->_name);
            layer->forwardPropagation( "train" );
        }

        for( int i = g_vQue.size() - 1; i>= 0; i--){
            LayersBase* layer = (LayersBase*)Layers::instanceObject()->getLayer(g_vQue[i]->_name);
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
        LayersBase* layer = (LayersBase*)Layers::instanceObject()->getLayer( config->_name );
        layer->setCurBranchIndex(i);
        dfsTraining( tmpConfig, nMomentum, trainData, trainLabel, iter);
    }
    g_vQue.pop_back();
}

/*
 *ascend order
 */
bool cmp_ascend_Order( SoftMaxLayer* a, SoftMaxLayer* b)
{
    return (a->getCorrectNum()) < (b->getCorrectNum());
}

/*
 *Get min result branch
 */
void getBranchResult(LayersBase*curLayer)
{
    //叶子节点
    if (curLayer->nextLayer.size() == 0)
    {
        SoftMaxLayer* tmp = (SoftMaxLayer*) curLayer;
        g_vBranchResult.push_back(tmp);
    }

    for (int i = 0; i < curLayer->nextLayer.size(); i++) {
        LayersBase* tmpLayer = curLayer->nextLayer[i];
        getBranchResult(tmpLayer);
    }
    //	//叶子节点
    //	if (curLayer->nextLayer.size() == 0)
    //	{
    //		SoftMaxLayer* tmp = (SoftMaxLayer*)curLayer;
    //		if(tmp->getCorrectNum() < g_nMinCorrSize)
    //		{
    //			g_nMinCorrSize = tmp->getCorrectNum();
    //			g_vMinBranch.push_back(tmp);
    //		}
    //	}
    //
    //	for(int i = 0; i < curLayer->nextLayer.size(); i++)
    //	{
    //		LayersBase* tmpLayer =  curLayer->nextLayer[i];
    //		ascend_OrderBranch(tmpLayer);
    //	}
}


/*
 *Get Fissnode and Fission
 */
void performFiss()
{
    for(int i = 0; i < g_vBranchResult.size(); i++)
    {
        LayersBase* tmpCur = (LayersBase*)g_vBranchResult[i];

        /*Find fissable node*/
        while (tmpCur->prevLayer[0]->_name != string("data") && tmpCur->prevLayer[0]->nextLayer.size() == 1)
        {
            tmpCur = tmpCur->prevLayer[0];
        }
        /*if curBranch is Fiss to data layer, then fiss another 如果分裂到数据层,那么就不再分裂,进行下一个分支的裂变*/
        if (tmpCur->prevLayer[0]->_name== "data" && (i != g_vBranchResult.size() - 1))
        continue;
        else if(i == g_vBranchResult.size() - 1)
        {
            /*softmax layer fission, add a classifier*/
            softmaxFission(g_vBranchResult[0]);
            break;
        }
        else
        {
            string layerType = config::instanceObjtce()->getLayersByName(tmpCur->prevLayer[0]->_name)->_type;
            /*全连接层和卷积层需要经过g_nSplitIndex迭代之后才裂变,其他层直接裂变*/
            if(layerType == string("HIDDEN") || layerType == string("CONV"))
            {
                //cifar-10 set 15
                g_nSplitIndex = 10;
            }else
            {
                g_nSplitIndex = 1;
            }
            //Fission one Node every time
            cout<< tmpCur->prevLayer[0]->_name<<endl;
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
    configBase* config = (configBase*) config::instanceObjtce()->getFirstLayers();
    /*裂变学习率的初始设定*/
    //dfsGetLearningRateReduce( config );
    
    cout<<"TestData Forecast The Result..."<<endl;
    predictTestData(testData, testLabel, batchSize);
    cout<<endl;

    cout<<"NetWork training......"<<endl;
    int epochs = config::instanceObjtce()->get_trainEpochs();
    int iter_per_epo = config::instanceObjtce()->get_iterPerEpo();
    int layerNum = Layers::instanceObject()->getLayersNum();
    double nMomentum[]={0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99};
    //int epoCount[]={80,80,80,80,80,80,80,80,80,80};
    int epoCount[]={100,300,500,700,900,1100,1300,1500,1700,80};
    float Momentum = 0.9;
    int id = 0;

    clock_t start, stop;
    double runtime;

    start = clock();
    for(int epo = 0; epo < epochs; epo++)
    {
        DataLayer* datalayer = static_cast<DataLayer*>(Layers::instanceObject()->getLayer("data"));
        //Momentum = nMomentum[id];
        Momentum = 0.9;

        clock_t inStart, inEnd;
        inStart = clock();
        configBase* config = (configBase*) config::instanceObjtce()->getFirstLayers();
        if( false == DFS_TRAINING ){
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
        // adjust learn rate
        globalRateAdjust(epo);
        
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

        /*在进入下一次训练之前进行裂变*/
        if (DFS_TRAINING == true && FISS_TRAINING == true )
        {
            g_nCount ++;
            if (epo >= 40 && (epo % g_nSplitIndex) == 0 && (g_nCount >= g_nSplitIndex)) {
                g_vBranchResult.clear();
                LayersBase* curLayer = Layers::instanceObject()->getLayer("data");
                getBranchResult(curLayer);
                sort(g_vBranchResult.begin(), g_vBranchResult.end(), cmp_ascend_Order);
                performFiss();
                g_nCount = 0;
            }
        }
    }

    stop = clock();
    runtime = stop - start;
    cout<< epochs <<" epochs total rumtime is: "<<runtime /CLOCKS_PER_SEC<<" Seconds"<<endl;
}

// all learn rate adjust
void globalRateAdjust(int epoch)
{
    string sMethod = config::instanceObjtce()->get_lrPolicy();
    configBase* config = (configBase*) config::instanceObjtce()->getFirstLayers();
    if("INV" == sMethod)
    {
        queue<configBase*> que;
        set<configBase*> hash;
        hash.insert(config);
        que.push(config);
        while( !que.empty() ){
            config = que.front();
            que.pop();
            LayersBase * layer = (LayersBase*)Layers::instanceObject()->getLayer(config->_name);
            layer->adjust_learnRate(epoch, FLAGS_lr_gamma, FLAGS_lr_power);

            for(int i = 0; i < config->_next.size(); i++){
                if( hash.find(config->_next[i]) == hash.end()){
                    hash.insert(config->_next[i]);
                    que.push(config->_next[i]);
                }
            }
        }

    }else if("STEP" == sMethod)
    {
        if( epoch == 150 || epoch == 450 || epoch == 750 )
        {
            queue<configBase*> que;
            set<configBase*> hash;
            hash.insert(config);
            que.push(config);
            while(!que.empty())
            {
                config = que.front();
                que.pop();
                LayersBase * layer = (LayersBase*)Layers::instanceObject()->getLayer(config->_name);
                if(epoch == 150)
                    layer->lrate /= 10.0f;
                else if(epoch == 450)
                    layer->lrate /= 5.0f;
                else layer->lrate /= 2.0f;
               for(int i = 0; i < config->_next.size(); i++){
                    if( hash.find(config->_next[i]) == hash.end()){
                        hash.insert(config->_next[i]);
                        que.push(config->_next[i]);
                    }
                }
            }
        }
    }else
    {
       //fixed 
    }
}

// 动态生成全局参数
int g_nGenerateIndex = 25;

// Train dynamic generation model
void dynamic_g_trainNet(cuMatrixVector<float> &trainData, 
                        cuMatrix<int>* &trainLabel, 
                        cuMatrixVector<float> &testData,
                        cuMatrix<int>*&testLabel,
                        config* endConfig 
                       )
{
    int epochs = config::instanceObjtce()->get_trainEpochs();
    int iter_per_epo = config::instanceObjtce()->get_iterPerEpo();
    int layerNum = Layers::instanceObject()->getLayersNum();
    int batchSize = config::instanceObjtce()->get_batchSize();
    double nMomentum[]={0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99};
    int epoCount[]={100,300,500,700,900,1100,1300,1500,1700,80};
    float Momentum = 0.9;
    int id = 0;
    clock_t start, stop;
    double runtime;

    configBase* config = (configBase*) config::instanceObjtce()->getFirstLayers();
    cout<<"TestData Forecast The Result..."<<endl;
    predictTestData(testData, testLabel, batchSize);
    cout<<endl;

    cout<<"NetWork training......"<<endl;
    start = clock();
    for(int epo = 0; epo < epochs; epo++)
    {
        clock_t inStart, inEnd;
        DataLayer* datalayer = static_cast<DataLayer*>(Layers::instanceObject()->getLayer("data"));
        //Momentum = nMomentum[id];
        Momentum = 0.9;

        inStart = clock();
        configBase* config = (configBase*) config::instanceObjtce()->getFirstLayers();
        // train one epoch
        for(int iter = 0 ; iter < iter_per_epo; iter++)
        {
            datalayer->RandomBatch_Images_Label(trainData, trainLabel);
            getNetWorkCost(Momentum);
        }
        inEnd = clock();

        // adjust learn rate
        globalRateAdjust(epo);

        if(epo && epo % epoCount[id] == 0)
        {
            id++;
            if(id > 9) id = 9;
        }

        // Test network
        cout<< "epochs: " << epo << " ,Time: " << (inEnd - inStart)/CLOCKS_PER_SEC << "s,";
        predictTestData( testData, testLabel, batchSize );
        cout<< " ,Momentum: " << Momentum << endl;

        
        /*在进入下一次训练之前动态生成下一层*/
        while(((epo + 1) % g_nGenerateIndex) == 0)
        {
            ++g_nCount;
            configBase* templateConfig = endConfig->getFirstLayers();
            configBase* curConfig = config::instanceObjtce()->getFirstLayers();
            configBase* newConfig_prev = NULL;
            configBase* newConfig_next = NULL;
            // find the first different node 
            while(0 != templateConfig->_next.size() && 0 != curConfig->_next.size())
            {
                if(templateConfig->_type == string("SOFTMAX") || curConfig->_type == string("SOFTMAX"))
                {
                    LOG(FATAL) << "Config Table Error, SoftMax not finally layer." ;
                }

                if((templateConfig->_name == curConfig->_name) && (templateConfig->_type == curConfig->_type))
                {
                    newConfig_prev = curConfig;
                    templateConfig = templateConfig->_next[0];
                    curConfig = curConfig->_next[0];
                }else
                {
                    newConfig_next = curConfig;
                    break;
                }
            }

            if(NULL != newConfig_prev && NULL != newConfig_next)
            {
                static int nIndex = 0;
                nodeGenerate(templateConfig, newConfig_prev, newConfig_next);
                if((templateConfig->_type == string("CONV") || templateConfig->_type == string("HIDDEN")) && templateConfig->_next[0]->_type != string("CONV"))
                {
                    continue;
                }else if(2 == g_nCount)
                {
                    //++nIndex;
                    //if(nIndex == 1)g_nGenerateIndex = 25;
                    //if(nIndex == 2)g_nGenerateIndex = 20;
                    //if(nIndex == 3)g_nGenerateIndex = 10;
                    g_nCount = 0;
                    break;
                }
            }
            break;
        }
        
    }
    stop = clock();
    runtime = stop - start;
    cout<< epochs << " epochs total rumtime is: " << runtime /CLOCKS_PER_SEC << " Seconds" <<endl;
}
