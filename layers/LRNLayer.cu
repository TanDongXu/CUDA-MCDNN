#include"LRNLayer.h"
#include"../common/checkError.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"

void LRNLayer::createHandles()
{
	checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
	checkCUDNN(cudnnCreateLRNDescriptor(&normDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}

LRNLayer::LRNLayer(string name)
{
	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
	lrate = 0.0f;
	prevLayer.clear();
	nextLayer.clear();

	configLRN* curConfig = (configLRN*)config::instanceObjtce()->getLayersByName(_name);
	string prevLayerName = curConfig->_input;
	layersBase* prev_Layer = (layersBase*)Layers::instanceObject()->getLayer(prevLayerName);

	lrnN = curConfig->_lrnN;
	lrnAlpha = curConfig->_lrnAlpha;
	lrnBeta = curConfig->_lrnBeta;
	lrnK = 1.0;

	inputAmount = prev_Layer->channels;
	inputImageDim = prev_Layer->height;
	number = prev_Layer->number;
	channels = prev_Layer->channels;
	height = prev_Layer->height;
	width = prev_Layer->width;
	inputSize = prev_Layer->getOutputSize();
	outputSize =inputSize;

	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, number * channels * height* width * sizeof(float));

	this->createHandles();
}

void LRNLayer::forwardPropagation(string train_or_test)
{
	srcData = prevLayer[0]->dstData;

	checkCUDNN(cudnnSetLRNDescriptor(normDesc,
			                         lrnN,
			                         lrnAlpha,
			                         lrnBeta,
			                         lrnK));

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
	checkCUDNN(cudnnLRNCrossChannelForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                               normDesc,
			                               CUDNN_LRN_CROSS_CHANNEL_DIM1,
			                               &alpha,
			                               srcTensorDesc,
			                               srcData,
			                               &beta,
			                               dstTensorDesc,
			                               dstData));

}


void LRNLayer::backwardPropagation(float Momentum)
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
	checkCUDNN(cudnnLRNCrossChannelBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                                normDesc,
			                                CUDNN_LRN_CROSS_CHANNEL_DIM1,
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


void LRNLayer::destroyHandles()
{
	checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc))
	checkCUDNN(cudnnDestroyLRNDescriptor(normDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));
}
