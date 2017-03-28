/*
* ActivationLayer.h
*
*  Created on: Dec 13, 2015
*      Author: tdx
*/

#ifndef ACTIVATIONLAYER_H_
#define ACTIVATIONLAYER_H_

#include<cudnn.h>
#include"LayersBase.h"
#include"config/config.h"

/*
 * Class activation layer
 * */
class ActivationLayer: public LayersBase
{
    public:
    ActivationLayer(string name);
    ActivationLayer(const ActivationLayer* layer);
    ActivationLayer(const configBase* templateConfig);
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
    cudnnActivationDescriptor_t activDesc;
};

#endif /* ACTIVATIONLAYER_H_ */
