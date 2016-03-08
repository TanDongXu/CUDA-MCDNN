/*
 * softMaxLayer.h
 *
 *  Created on: Nov 28, 2015
 *      Author: tdx
 */

#ifndef SOFTMAXLAYER_H_
#define SOFTMAXLAYER_H_

#include<cudnn.h>
#include"layersBase.h"
#include"../common/cuMatrix.h"

class softMaxLayer : public layersBase
{
public:

	softMaxLayer(string name);
	void initRandom();
	void forwardPropagation(string train_or_test);
	void backwardPropagation(float Momentum);
	void saveWeight(FILE*file){}
	void readWeight(FILE*file){}
	void Forward_cudaFree();
	void Backward_cudaFree();
	void ClassificationResults();
	void getBackPropDiffData();
	void GetDataSize_BatchLabel();

	~softMaxLayer()
	{
		destroyHandles();
	}


	void createHandles();
	void destroyHandles();

	int getOutputSize()
	{
		return NULL;
	}

public:


private:
	int inputSize;
	int outputSize;
	int nclasses;
	int batchSize;
	int dataSize;
	int *srcLabel;
	int cur_correctSize;
	int CorrectSize;
	int flag;

	//计算cost用
	float lambda;
	float *srcDiff;


private:
    cudnnTensorDescriptor_t srcTensorDesc = NULL;
    cudnnTensorDescriptor_t dstTensorDesc = NULL;
    cudnnTensorDescriptor_t srcDiffTensorDesc = NULL;
    cudnnTensorDescriptor_t dstDiffTensorDesc = NULL;

};



#endif /* SOFTMAXLAYER_H_ */
