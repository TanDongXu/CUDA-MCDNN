/*
 * hiddenLayer.h
 *
 *  Created on: Nov 28, 2015
 *      Author: tdx
 */

#ifndef HIDDENLAYER_H_
#define HIDDENLAYER_H_

#include"layersBase.h"
#include"../common/cuMatrix.h"
#include"../common/cuMatrixVector.h"
#include"../common/utility.cuh"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"
#include<cuda_runtime.h>
#include<math.h>
#include "curand.h"

class hiddenLayer: public layersBase
{
public:

	hiddenLayer(string name, int sign);
	void initRandom();
	void forwardPropagation(string train_or_test);
	void backwardPropagation(float Momentum);
	void saveWeight(FILE*file);
	void readWeight(FILE*file);
	void Forward_cudaFree();
	void Backward_cudaFree();
	void dropOut(float *data, int size, float dropout_rate);


	~hiddenLayer()
	{
		MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
		MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Weight);
		MemoryMonitor::instanceObject()->freeGpuMemory(dev_Bias);
		MemoryMonitor::instanceObject()->freeGpuMemory(VectorOnes);
		destroyHandles();
	}

	void createHandles();
	void destroyHandles();

    int getOutputSize()
   {
	    return outputSize;
   }



private:

	float* dev_Weight, *host_Weight;
	float* dev_Bias, *host_Bias;
	float* dev_Wgrad,*dev_Bgrad;
	float epsilon;
	float* VectorOnes;
	int inputSize;
	int outputSize;
	int batchSize;
	int nonLinearity;
	float lambda;


private:
	curandGenerator_t curandGenerator_W;
	curandGenerator_t curandGenerator_B;

};



#endif /* HIDDENLAYER_H_ */
