#include"activationLayer.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"
#include"../common/utility.cuh"

void activationLayer::createHandles()
{
	checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}

/*activation layer constructor*/
activationLayer::activationLayer(string name)
{
	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
    prevLayer.clear();
    nextLayer.clear();

	configActivation * curConfig = (configActivation*) config::instanceObjtce()->getLayersByName(_name);
	string preLayerName = curConfig->_input;
	layersBase* prev_Layer = (layersBase*) Layers::instanceObject()->getLayer(preLayerName);

	inputAmount = prev_Layer->channels;
	inputImageDim = prev_Layer->height;
	number = prev_Layer->number;
	channels = prev_Layer->channels;
	height = prev_Layer->height;
	width = prev_Layer->width;
    outputSize = channels * height * width;

    ActivationMode = (cudnnActivationMode_t)curConfig->_non_linearity;
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, number * channels * height * width * sizeof(float));

    this->createHandles();
}

//deep copy constructor
activationLayer::activationLayer(activationLayer* layer)
{
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
	prevLayer.clear();
	nextLayer.clear();

	static int idx = 0;
	_name = layer->_name + int_to_string(idx);
	idx ++;
	_inputName = layer->_inputName;

	inputAmount = layer->inputAmount;
	inputImageDim = layer->inputImageDim;
	number = layer->number;
	channels =  layer->channels;
	height = layer->height;
	width = layer->width;
	outputSize = layer->outputSize;
	ActivationMode = layer->ActivationMode;

	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpu2gpu(dstData, layer->dstData, number * channels * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpu2gpu(diffData, layer->diffData, number * channels * height * width * sizeof(float));

	cout<<"act deep copy"<<endl;
	this->createHandles();
}

void activationLayer::forwardPropagation(string train_or_test)
{
	srcData = prevLayer[0]->dstData;

    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
		                                 cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
		                                 cuDNN_netWork<float>::instanceObject()->GetDataType(),
		                                 number,
		                                 channels,
		                                 height,
		                                 width));

	checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
			                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
			                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
			                              number,
			                              channels,
			                              height,
			                              width));

	MemoryMonitor::instanceObject()->gpuMemoryMemset(dstData,number*channels*height*width*sizeof(float));
	float alpha = 1.0f;
	float beta = 0.0f;
	checkCUDNN(cudnnActivationForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                          ActivationMode,
			                          &alpha,
			                          srcTensorDesc,
			                          srcData,
			                          &beta,
			                          dstTensorDesc,
			                          dstData));
}


void activationLayer::backwardPropagation(float Momentum)
{
	checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
				                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
				                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
				                          number,
				                          channels,
				                          height,
				                          width));

	checkCUDNN(cudnnSetTensor4dDescriptor(srcDiffTensorDesc,
				                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
				                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
				                          number,
				                          channels,
				                          height,
				                          width));

	checkCUDNN(cudnnSetTensor4dDescriptor(dstDiffTensorDesc,
				                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
				                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
				                          number,
				                          channels,
				                          height,
				                          width));

	checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
			                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
			                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
			                              number,
			                              channels,
			                              height,
			                              width));

	float alpha = 1.0f;
	float beta = 0.0f;
	int nIndex = m_nCurBranchIndex;
	checkCUDNN(cudnnActivationBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                           ActivationMode,
			                           &alpha,
			                           dstTensorDesc,
			                           dstData,
			                           srcDiffTensorDesc,
			                           nextLayer[nIndex]->diffData,
			                           srcTensorDesc,
			                           srcData,
			                           &beta,
			                           dstDiffTensorDesc,
			                           diffData));

}

void activationLayer::destroyHandles()
{
	checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));

}
