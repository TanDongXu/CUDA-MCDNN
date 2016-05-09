/*
* dropOutLayer.h
*
*  Created on: Mar 15, 2016
*      Author: tdx
*/

#ifndef DROPOUTLAYER_H_
#define DROPOUTLAYER_H_

#include"../config/config.h"
#include"../tests/test_layer.h"
#include"../common/utility.cuh"
#include"layersBase.h"
#include<curand.h>


class dropOutLayer : public layersBase
{
    public:
    dropOutLayer(string name);
    dropOutLayer(dropOutLayer* layer);
    void CreateUniform(int size);
    void Dropout_TrainSet(float* data, int size, float dropout_rate);
    void Dropout_TestSet(float* data, int size, float dropout_rate);
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momemtum);
    void saveWeight(FILE* file){}
    void readWeight(FILE* file){}
    void createHandles();
    void destroyHandles();

    ~dropOutLayer()
    {
        MemoryMonitor::instanceObject()->freeGpuMemory(outputPtr);
        destroyHandles();
    }

    int getOutputSize()
    {
        return outputSize;
    }

    private:
    int outputSize;
    float DropOut_rate;
    float* outputPtr;

    private:
    curandGenerator_t curandGenerator_DropOut;
};



#endif /* DROPOUTLAYER_H_ */
