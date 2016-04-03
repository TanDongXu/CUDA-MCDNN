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
#include"../common/utility.cuh"
#include "../cuDNN_netWork.h"
#include"../config/config.h"
#include"../common/MemoryMonitor.h"
#include"../common/checkError.h"
#include<time.h>
#include<tuple>
#include<cudnn.h>
#include<curand.h>

using namespace std;


class convLayer:public layersBase
{
public:
	typedef tuple<int, int, int, int, int, int, int, int, float, float, float> param_tuple;

	convLayer(string name, int sign);
	convLayer(string name, int sign, const param_tuple& args);
	convLayer(convLayer* layer);
	void initRandom();
	void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file);
    void readWeight(FILE*file);
	void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, int c, float *data );

	~convLayer()
	{
		MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
		MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Weight);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Bias);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Wgrad);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Bgrad);
		MemoryMonitor::instanceObject()->freeGpuMemory(tmp_Wgrad);
	    MemoryMonitor::instanceObject()->freeGpuMemory(tmp_Bgrad);
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

	float *host_Weight, *dev_Weight;
	float *tmp_Wgrad, *tmp_Bgrad;
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
	int outputSize;
	int batchSize;
	int prev_num;
	int prev_channels;
	int prev_height;
	int prev_width;

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

};



#endif /* CONVLAYER_H_ */
