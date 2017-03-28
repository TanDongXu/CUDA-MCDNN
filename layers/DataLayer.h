/*
* DataLayer.h
*
*  Created on: Nov 29, 2015
*      Author: tdx
*/

#ifndef DATALAYER_H_
#define DATALAYER_H_

#include"LayersBase.h"
#include"../common/cuMatrixVector.h"
#include"../tests/test_layer.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../common/utility.cuh"
#include<cstring>
#include<cuda_runtime_api.h>

/*
 * Class Data Layer
 * */
class DataLayer: public LayersBase
{
    public:
    DataLayer(string name);
    DataLayer(const DataLayer* layer);
    DataLayer(const configBase* templateConfig){}
    ~DataLayer();
    void getBatch_Images_Label(int index, cuMatrixVector<float> &inputData, cuMatrix<int>* &inputLabel);
    void RandomBatch_Images_Label(cuMatrixVector<float> &inputData, cuMatrix<int>* &inputLabel);
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file){}
    void readWeight(FILE*file){}
    int getOutputSize();
    int getDataSize();
    int* getDataLabel();

    private:
    int dataSize;
    int batchSize;
    int *srcLabel;
    float* batchImage;
};

#endif /* DATALAYER_H_ */
