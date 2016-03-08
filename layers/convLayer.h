/*
 * convLayer.h
 *
 *  Created on: Nov 23, 2015
 *      Author: tdx
 */

#ifndef CONVLAYER_H_
#define CONVLAYER_H_

#include"./layersBase.h"
#include"../common/cuMatrixVector.h"
#include"../common/cuMatrix.h"
#include"../common/utility.h"
#include "../cuDNN_netWork.h"

#include<tuple>
#include<cudnn.h>
#include<curand.h>

using namespace std;


class convLayer:public convLayerBase
{
public:
	typedef tuple<int, int, int, int, int, int, int, int, float, float, float> param_tuple;

	convLayer(string name, int sign);
	convLayer(string name, int sign, const param_tuple& args);
	void initRandom();
	void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file);
    void readWeight(FILE*file);
    void Forward_cudaFree();
    void Backward_cudaFree();
	void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, int c, float *data );

	~convLayer()
	{
		MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
		MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Weight);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Bias);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Wgrad);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Bgrad);
		destroyHandles();
	}

	void createHandles();
	void destroyHandles();

	int getOutputSize()
	{
		return outputSize;
	}


private:

	float *host_Weight, *dev_Weight;
	float *host_Bias, *dev_Bias;
	float *dev_Wgrad, *dev_Bgrad;
	float lambda;
	float epsilon;
	int kernelSize;
	int pad_h;
	int pad_w;
	int stride_h;
	int stride_w;
	int kernelAmount;
	int non_linearity;
	int outputSize;
	int batchSize;



private:
	cudnnTensorDescriptor_t srcTensorDesc = NULL;
	cudnnTensorDescriptor_t dstTensorDesc = NULL;
	cudnnTensorDescriptor_t biasTensorDesc = NULL;
	cudnnTensorDescriptor_t srcDiffTensorDesc = NULL;
	cudnnTensorDescriptor_t dstDiffTensorDesc = NULL;
	cudnnFilterDescriptor_t filterDesc = NULL;
	cudnnConvolutionDescriptor_t convDesc = NULL;
	cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)-1;

private:
	curandGenerator_t curandGenerator_W;
	curandGenerator_t curandGenerator_B;



};



#endif /* CONVLAYER_H_ */
