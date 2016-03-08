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

class activationLayer: public convLayerBase
{
public:
	activationLayer(string name);
	void forwardPropagation(string train_or_test);
	void backwardPropagation(float Momentum);
	void saveWeight(FILE*file){}
	void readWeight(FILE*file){}
	void Forward_cudaFree();
	void Backward_cudaFree();

	~activationLayer()
	{
		destroyHandles();
	}

	void createHandles();
	void destroyHandles();

	int getOutputSize()
	{
		return outputSize;
	}

private:
	int inputSize;
	int outputSize;



private:
	cudnnTensorDescriptor_t srcTensorDesc = NULL;
	cudnnTensorDescriptor_t dstTensorDesc = NULL;
	cudnnTensorDescriptor_t srcDiffTensorDesc = NULL;
	cudnnTensorDescriptor_t dstDiffTensorDesc = NULL;

};



#endif /* ACTIVATIONLAYER_H_ */
