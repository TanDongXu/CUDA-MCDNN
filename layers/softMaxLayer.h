/*
* softMaxLayer.h
*
*  Created on: Nov 28, 2015
*      Author: tdx
*/

#ifndef SOFTMAXLAYER_H_
#define SOFTMAXLAYER_H_

#include<cudnn.h>
#include"layersBase.h"
#include"../common/cuMatrix.h"
#include"dataLayer.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"
#include"../saveData/saveNetWork.h"
#include"../common/utility.cuh"

class softMaxLayer : public layersBase
{
    public:
    softMaxLayer(string name);
    softMaxLayer(softMaxLayer* layer);
    void initRandom();
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file){}
    void readWeight(FILE*file){}
    void ClassificationResults();
    void getBackPropDiffData();
    void GetDataSize_BatchLabel();

    ~softMaxLayer()
    {
        MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
        MemoryMonitor::instanceObject()->freeGpuMemory(diffData);
        MemoryMonitor::instanceObject()->freeGpuMemory(srcDiff);
        MemoryMonitor::instanceObject()->freeGpuMemory(devLabel);
        MemoryMonitor::instanceObject()->freeCpuMemory(host_result);
        destroyHandles();
    }

    void createHandles();
    void destroyHandles();

    int getOutputSize()
    {
        return 0;
    }

    int getCorrectNum()
    {
        return CorrectSize;
    }


    private:
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

private:
    cudnnTensorDescriptor_t srcTensorDesc = NULL;
    cudnnTensorDescriptor_t dstTensorDesc = NULL;
    cudnnTensorDescriptor_t srcDiffTensorDesc = NULL;
    cudnnTensorDescriptor_t dstDiffTensorDesc = NULL;

};
#endif /* SOFTMAXLAYER_H_ */
