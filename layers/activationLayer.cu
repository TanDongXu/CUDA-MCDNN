#include"activationLayer.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"

#include"opencv2/imgproc/imgproc.hpp"
#include"opencv2/highgui/highgui.hpp"

using namespace cv;

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
	number = 0;
	channels =0;
	height = 0;
	width = 0;
	lrate = 0.0f;
    prevLayer.clear();
    nextLayer.clear();

	configActivation * curConfig = (configActivation*) config::instanceObjtce()->getLayersByName(_name);
	string preLayerName = curConfig->_input;

	convLayerBase* prev_Layer = (convLayerBase*) Layers::instanceObject()->getLayer(preLayerName);

	_inputAmount = prev_Layer->_outputAmount;
    _outputAmount = _inputAmount;
	_inputImageDim = prev_Layer->_outputImageDim;
	_outputImageDim = _inputImageDim;
	inputSize = prev_Layer->getOutputSize();
    outputSize =inputSize;

    ActivationMode = (cudnnActivationMode_t)curConfig->_non_linearity;

    this->createHandles();
}


void activationLayer::forwardPropagation(string train_or_test)
{
	number = prevLayer[0]->number;
	channels = prevLayer[0]->channels;
	height = prevLayer[0]->height;
	width = prevLayer[0]->width;
	srcData = prevLayer[0]->dstData;

	dstData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));


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


/*free forwardPropagation memory*/
void activationLayer::Forward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(srcData);
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





	diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, number * channels * height * width * sizeof(float));

	float alpha = 1.0f;
	float beta = 0.0f;
	checkCUDNN(cudnnActivationBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                           ActivationMode,
			                           &alpha,
			                           dstTensorDesc,
			                           dstData,
			                           srcDiffTensorDesc,
			                           nextLayer[0]->diffData,
			                           srcTensorDesc,
			                           srcData,
			                           &beta,
			                           dstDiffTensorDesc,
			                           diffData));

}


/*free backwardPropagation memory*/
void activationLayer::Backward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer[0]->diffData);
}


void activationLayer::destroyHandles()
{
	checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));

}
