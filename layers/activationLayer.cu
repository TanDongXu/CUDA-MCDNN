#include"activationLayer.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"

#include"opencv2/imgproc/imgproc.hpp"
#include"opencv2/highgui/highgui.hpp"

using namespace cv;

void activationLayer::createHandles()
	{
		/*cudnnCreateTensorDescriptor创建一个tensor对象（并没有初始化）*/
		checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
	}


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
	prevLayer = NULL;
	nextLayer = NULL;

	configActivation * curConfig = (configActivation*) config::instanceObjtce()->getLayersByName(_name);
	string preLayerName = curConfig->_input;
	//layersBase* prelayer = (layersBase*) Layers::instanceObject()->getLayer(preLayerName);

	convLayerBase* prev_Layer = (convLayerBase*) Layers::instanceObject()->getLayer(preLayerName);

	_inputAmount = prev_Layer->_outputAmount;
    _outputAmount = _inputAmount;
	_inputImageDim = prev_Layer->_outputImageDim;
	_outputImageDim = _inputImageDim;

	inputSize = prev_Layer->getOutputSize();
    outputSize =inputSize;

    this->createHandles();
}


void activationLayer::forwardPropagation(string train_or_test)
{
	number = prevLayer->number;
	channels = prevLayer->channels;
	height = prevLayer->height;
	width = prevLayer->width;
	srcData = prevLayer->dstData;

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
			                          CUDNN_ACTIVATION_RELU,
			                          &alpha,
			                          srcTensorDesc,
			                          srcData,
			                          &beta,
			                          dstTensorDesc,
			                          dstData));

//	/*如果是测试集的，就释放源输入，因为没有反向传导*/
//	if(train_or_test == "test")
//		MemoryMonitor::instanceObject()->freeGpuMemory(srcData);

}

void activationLayer::Forward_cudaFree()
{
	/*如果是测试集的，就释放源输入，因为没有反向传导*/
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
	/*计算神经元激活函数的梯度*/
	checkCUDNN(cudnnActivationBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                           CUDNN_ACTIVATION_RELU,
			                           &alpha,
			                           dstTensorDesc,
			                           dstData,
			                           srcDiffTensorDesc,
			                           nextLayer->diffData,
			                           srcTensorDesc,
			                           srcData,
			                           &beta,
			                           dstDiffTensorDesc,
			                           diffData));

//	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
//	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer->diffData);
}



void activationLayer::Backward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer->diffData);
}


void activationLayer::destroyHandles()
{
	/*销毁创建的描述符  逆向销毁*/
	checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));

}