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
#include"layersBase.h"

using namespace std;

class LRNLayer : public layersBase
{
public:

	LRNLayer(string name);
	void forwardPropagation(string train_or_test);
	void backwardPropagation(float Momentum);
	void saveWeight(FILE*file){}
	void readWeight(FILE*file){}
	void Forward_cudaFree();
	void Backward_cudaFree();

	~LRNLayer()
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
	int outputSize;
	int inputSize;
	unsigned lrnN ;
	double lrnAlpha;
	double lrnBeta;
	double lrnK;


private:
	cudnnTensorDescriptor_t srcTensorDesc = NULL;
	cudnnTensorDescriptor_t dstTensorDesc = NULL;
	cudnnLRNDescriptor_t normDesc;

	cudnnTensorDescriptor_t srcDiffTensorDesc = NULL;
	cudnnTensorDescriptor_t dstDiffTensorDesc = NULL;


};



#endif /* LRNLAYER_H_ */
