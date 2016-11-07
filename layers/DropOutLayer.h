/*
* DropOutLayer.h
*
*  Created on: Mar 15, 2016
*      Author: tdx
*/

#ifndef DROPOUTLAYER_H_
#define DROPOUTLAYER_H_

#include"../config/config.h"
#include"../tests/test_layer.h"
#include"../common/utility.cuh"
#include"LayersBase.h"
#include<curand.h>

/*
 * Class DropOut layer
 * */
class DropOutLayer : public LayersBase
{
    public:
    DropOutLayer(string name);
    DropOutLayer(DropOutLayer* layer);
    ~DropOutLayer();
    void CreateUniform(int size);
    void Dropout_TrainSet(float* data, int size, float dropout_rate);
    void Dropout_TestSet(float* data, int size, float dropout_rate);
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momemtum);
    void saveWeight(FILE* file){}
    void readWeight(FILE* file){}
    void createHandles();
    void destroyHandles();
    int getOutputSize();

    private:
    int outputSize;
    float DropOut_rate;
    float* outputPtr;
    curandGenerator_t curandGenerator_DropOut;
};


#endif /* DROPOUTLAYER_H_ */
