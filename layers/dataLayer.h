/*
* dataLayer.h
*
*  Created on: Nov 29, 2015
*      Author: tdx
*/

#ifndef DATALAYER_H_
#define DATALAYER_H_

#include"layersBase.h"
#include"../common/cuMatrixVector.h"
#include"../tests/test_layer.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../common/utility.cuh"
#include<cstring>
#include<cuda_runtime_api.h>


class dataLayer: public layersBase
{
    public:
    dataLayer(string name);
    dataLayer(dataLayer* layer);
    void getBatch_Images_Label(int index, cuMatrixVector<float> &inputData, cuMatrix<int>* &inputLabel);
    void RandomBatch_Images_Label(cuMatrixVector<float> &inputData, cuMatrix<int>* &inputLabel);
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file){}
    void readWeight(FILE*file){}

    ~dataLayer()
    {
        MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
        MemoryMonitor::instanceObject()->freeCpuMemory(srcLabel);
        MemoryMonitor::instanceObject()->freeCpuMemory(batchImage);
    }
    int getOutputSize()
    {
        return channels * height * width;
    }

    int getDataSize()
    {
        return dataSize;
    }

    int* getDataLabel()
    {
        return srcLabel;
    }

    private:

    int dataSize;
    int batchSize;
    int *srcLabel;
    float* batchImage;

};

#endif /* DATALAYER_H_ */
