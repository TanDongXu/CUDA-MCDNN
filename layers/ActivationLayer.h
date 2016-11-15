/*
* ActivationLayer.h
*
*  Created on: Dec 13, 2015
*      Author: tdx
*/

#ifndef ACTIVATIONLAYER_H_
#define ACTIVATIONLAYER_H_

#include"LayersBase.h"
#include<cudnn.h>

/*
 * Class activation layer
 * */
class ActivationLayer: public LayersBase
{
    public:
    ActivationLayer(string name);
    ActivationLayer(const ActivationLayer* layer);
    ~ActivationLayer();
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file){}
    void readWeight(FILE*file){}
    void createHandles();
    void destroyHandles();
    int getOutputSize();

    private:
    int outputSize;
    int ActivationMode;
    cudnnActivationMode_t cudnnActivationMode;
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnTensorDescriptor_t srcDiffTensorDesc;
    cudnnTensorDescriptor_t dstDiffTensorDesc;
    cudnnActivationDescriptor_t activDesc;
};

#endif /* ACTIVATIONLAYER_H_ */
