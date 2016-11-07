/*
* LRNLayer.h
*
*  Created on: Dec 31, 2015
*      Author: tdx
*/

#ifndef LRNLAYER_H_
#define LRNLAYER_H_

#include<string>
#include<cudnn.h>
#include"LayersBase.h"
#include"../common/utility.cuh"

using namespace std;

/*
 * Class LRN layer
 * */
class LRNLayer : public LayersBase
{
    public:
    LRNLayer(string name);
    LRNLayer(LRNLayer* layer);
    ~LRNLayer();
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file){}
    void readWeight(FILE*file){}
    void createHandles();
    void destroyHandles();
    int getOutputSize();

    private:
    cudnnLRNDescriptor_t normDesc;
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnTensorDescriptor_t srcDiffTensorDesc;
    cudnnTensorDescriptor_t dstDiffTensorDesc;
    int outputSize;
    int inputSize;
    unsigned lrnN ;
    double lrnAlpha;
    double lrnBeta;
    double lrnK;
};

#endif /* LRNLAYER_H_ */
