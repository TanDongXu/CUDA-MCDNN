#include"poolLayer.h"


void poolLayer:: createHandles()
{
	checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
	checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}


/*constructor*/
poolLayer::poolLayer(string name)
{
	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
	number = 0;
	channels = 0;
	height =0;
	width =0;
	lrate =  0.0f;
	prevLayer = NULL;
	nextLayer = NULL;

	configPooling* curConfig = (configPooling*) config::instanceObjtce()->getLayersByName(_name);
	string prevLayerName = curConfig->_input;
	convLayerBase* prev_Layer = (convLayerBase*) Layers::instanceObject()->getLayer(prevLayerName);

	poolType = curConfig->_poolType;
	poolDim = curConfig->_size;
	pad_h = curConfig->_pad_h;
	pad_w = curConfig->_pad_w;
	stride_h =  curConfig->_stride_h;
	stride_w = curConfig->_stride_w;
    _inputImageDim = prev_Layer->_outputImageDim;
	/*the size after pool*/
	_outputImageDim = static_cast<int>(ceil(static_cast<float>(_inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
	_inputAmount = prev_Layer->_outputAmount;
	_outputAmount = _inputAmount;
	outputSize = _outputAmount * _outputImageDim * _outputImageDim;

	this->createHandles();
}

/*constructor overload*/
poolLayer::poolLayer(string name, const param_tuple& args)
{
	std::tie(poolType, poolDim, pad_h, pad_w, stride_h,
			stride_w, _inputImageDim, _inputAmount) = args;

	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
	number = 0;
	channels = 0;
	height = 0;
	width = 0;
	lrate = 0.0f;
	prevLayer = NULL;
	nextLayer = NULL;

	_outputImageDim = static_cast<int>(ceil(static_cast<float>(_inputImageDim + 2 * pad_h - poolDim) / stride_h)) + 1;
	_outputAmount = _inputAmount;
	outputSize = _outputAmount * _outputImageDim * _outputImageDim;

	this->createHandles();
}


void poolLayer::forwardPropagation(string train_or_test)
{
	srcData = NULL;
	number = prevLayer->number;
	channels = prevLayer->channels;
	height = prevLayer->height;
	width = prevLayer->width;
	srcData = prevLayer->dstData;

	checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc,
			                               CUDNN_POOLING_MAX,
			                               poolDim,
			                               poolDim,//window
			                               pad_h,
			                               pad_w,//pading
			                               stride_h,
			                               stride_w));//stride

	checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
			                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
			                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
			                              number,
			                              channels,
			                              height,
			                              width));

	height = static_cast<int>(ceil(static_cast<float>(height + 2 * pad_h - poolDim) / stride_h)) + 1;
	width = static_cast<int>(ceil(static_cast<float>(width + 2 * pad_w - poolDim) / stride_w)) + 1;


    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
    		                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
    		                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
    		                              number,
    		                              channels,
    		                              height,
    		                              width));


	dstData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));

	float alpha = 1.0;
	float beta = 0.0;
	checkCUDNN(cudnnPoolingForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                       poolingDesc,
			                       &alpha,
			                       srcTensorDesc,
			                       srcData,
			                       &beta,
			                       dstTensorDesc,
			                       dstData));

}

/*free forwardPropagation memory*/
void poolLayer::Forward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(srcData);
}



void poolLayer::backwardPropagation(float Momentum)
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

   int prevlayer_n, prevlayer_c, prevlayer_h,prevlayer_w;
   prevlayer_n = prevLayer->number;
   prevlayer_c = prevLayer->channels;
   prevlayer_h = prevLayer->height;
   prevlayer_w = prevLayer->width;

   checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
		                               cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
		                               cuDNN_netWork<float>::instanceObject()->GetDataType(),
		                               prevlayer_n,
		                               prevlayer_c,
		                               prevlayer_h,
		                               prevlayer_w));

   checkCUDNN(cudnnSetTensor4dDescriptor(dstDiffTensorDesc,
		                                 cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
		                                 cuDNN_netWork<float>::instanceObject()->GetDataType(),
		                                 prevlayer_n,
		                                 prevlayer_c,
		                                 prevlayer_h,
		                                 prevlayer_w));


   diffData = NULL;
   MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, 
                                                    prevlayer_n * prevlayer_c * prevlayer_h * prevlayer_w * sizeof(float));

   float alpha = 1.0f;
   float beta = 0.0;
   checkCUDNN(cudnnPoolingBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
		                           poolingDesc,
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

}



void poolLayer::Backward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer->diffData);
}

void poolLayer:: destroyHandles()
{
	checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc))
	checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));
}
