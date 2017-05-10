/*
* SoftMaxLayer.h
*
*  Created on: Nov 28, 2015
*      Author: tdx
*/

#ifndef SOFTMAXLAYER_H_
#define SOFTMAXLAYER_H_

#include<cudnn.h>
#include"LayersBase.h"
#include"../common/cuMatrix.h"
#include"./DataLayer.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"
#include"../saveData/saveNetWork.h"
#include"../common/utility.cuh"

/*
 *Class Softmax layer
 * */
class SoftMaxLayer : public LayersBase
{
    public:
    SoftMaxLayer(string name);
    SoftMaxLayer(const SoftMaxLayer* layer);
    SoftMaxLayer(const configBase* templateConfig){}
    ~SoftMaxLayer();
    void initRandom();
    void ReShape();
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file){}
    void readWeight(FILE*file){}
    void ClassificationResults();
    void getBackPropDiffData();
    void GetDataSize_BatchLabel();
    void createHandles();
    void destroyHandles();
    int getOutputSize();
    int getCorrectNum();
    void compute_cost();

    private:
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    int inputSize;
    int outputSize;
    int nclasses;
    int batchSize;
    int dataSize;
    int* srcLabel;
    int cur_correctSize;
    int CorrectSize;
    int flag;
    int* devLabel;
    float lambda;
    float* srcDiff;
    float* host_result;
    float* m_devGroundTruth;
    float* m_hostGroundTruth;
    float* dev_logArray;
};
#endif /* SOFTMAXLAYER_H_ */
