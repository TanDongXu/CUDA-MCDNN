/*
 * activationLayer.h
 *
 *  Created on: Dec 13, 2015
 *      Author: tdx
 */

#ifndef ACTIVATIONLAYER_H_
#define ACTIVATIONLAYER_H_

#include"layersBase.h"
#include<cudnn.h>

class activationLayer: public layersBase
{
public:
	activationLayer(string name);
	activationLayer(activationLayer* layer);
	void forwardPropagation(string train_or_test);
	void backwardPropagation(float Momentum);
	void saveWeight(FILE*file){}
	void readWeight(FILE*file){}
	~activationLayer()
	{
		MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
		MemoryMonitor::instanceObject()->freeGpuMemory(diffData);
		destroyHandles();
	}

	void createHandles();
	void destroyHandles();

	int getOutputSize()
	{
		return outputSize;
	}

private:
	int outputSize;
    int ActivationMode;
    cudnnActivationMode_t cudnnActivationMode;
private:
	cudnnTensorDescriptor_t srcTensorDesc = NULL;
	cudnnTensorDescriptor_t dstTensorDesc = NULL;
	cudnnTensorDescriptor_t srcDiffTensorDesc = NULL;
	cudnnTensorDescriptor_t dstDiffTensorDesc = NULL;
};



#endif /* ACTIVATIONLAYER_H_ */
