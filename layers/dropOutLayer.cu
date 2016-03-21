#include"dropOutLayer.h"

dropOutLayer::dropOutLayer(string name)
{
	_name = name;
	_inputName = " ";
	lrate = 0.0f;
	srcData = NULL;
	dstData = NULL;
    nextLayer.clear();
    prevLayer.clear();
	outputPtr = NULL;

	configDropOut* curConfig = (configDropOut*) config::instanceObjtce()->getLayersByName(_name);
	string prevLayerName = curConfig->_input;
	layersBase* prev_Layer = (layersBase*) Layers::instanceObject()->getLayer(prevLayerName);

	inputAmount = prev_Layer->channels;
	inputImageDim = prev_Layer->height;
	number = prev_Layer->number;
	channels = prev_Layer->channels;
	height = prev_Layer->height;
	width = prev_Layer->width;
	outputSize = channels * height * width;
	DropOut_rate = curConfig->dropOut_rate;

	this->createHandles();
}

void dropOutLayer::createHandles()
{
	curandCreateGenerator(&curandGenerator_DropOut, CURAND_RNG_PSEUDO_MTGP32);
}

__global__ void dropout_train(float* data, float* outputPtr, int size, float probability)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < size)
	{
		if(outputPtr[idx] < probability)
			data[idx] = 0;
	}
}

__global__ void dropout_test(float* data, int size, float probability)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < size)
	{
		data[idx] = data[idx] * probability;
	}
}

void dropOutLayer::CreateUniform(int size)
{
	curandSetPseudoRandomGeneratorSeed(curandGenerator_DropOut, time(NULL));
	outputPtr = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&outputPtr, size * sizeof(float));
	curandGenerateUniform(curandGenerator_DropOut, outputPtr, size);
}

void dropOutLayer::Dropout_TrainSet(float* data, int size, float dropout_rate)
{
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	dropout_train<<<blocksPerGrid, threadsPerBlock>>>(data, outputPtr, size, dropout_rate);

}

void dropOutLayer::Dropout_TestSet(float* data, int size, float dropout_rate)
{
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	dropout_test<<<blocksPerGrid, threadsPerBlock>>>(srcData, number * channels * height * width, DropOut_rate);
}

void dropOutLayer::forwardPropagation(string train_or_test)
{
	srcData =prevLayer[0]->dstData;
	dstData = srcData;

	/*use dropout in training, when testing multiply probability*/
	if(train_or_test == "train")
	{
		CreateUniform(number * channels * height * width);
		Dropout_TrainSet(srcData, number * channels * height * width, DropOut_rate);
	}
	else
		Dropout_TestSet(srcData, number * channels * height * width, DropOut_rate);

}


void dropOutLayer::backwardPropagation(float Momemtum)
{
	diffData = nextLayer[0]->diffData;
	Dropout_TrainSet(diffData, number * channels * height * width, DropOut_rate);
	MemoryMonitor::instanceObject()->freeGpuMemory(outputPtr);
}

void dropOutLayer::destroyHandles()
{
	curandDestroyGenerator(curandGenerator_DropOut);

}
