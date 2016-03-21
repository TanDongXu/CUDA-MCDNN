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
	lrate =  0.0f;
    prevLayer.clear();
    nextLayer.clear();

	configPooling* curConfig = (configPooling*) config::instanceObjtce()->getLayersByName(_name);
	string prevLayerName = curConfig->_input;
	layersBase* prev_Layer = (layersBase*) Layers::instanceObject()->getLayer(prevLayerName);

	poolType = curConfig->_poolType;
	poolDim = curConfig->_size;
	pad_h = curConfig->_pad_h;
	pad_w = curConfig->_pad_w;
	stride_h =  curConfig->_stride_h;
	stride_w = curConfig->_stride_w;

	prev_num = prev_Layer->number;
	prev_channels = prev_Layer->channels;
	prev_height = prev_Layer->height;
	prev_width = prev_Layer->width;

    inputImageDim = prev_Layer->height;
	inputAmount = prev_Layer->channels;
	number = prev_num;
	channels = prev_channels;
	height = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
	width = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
	outputSize = channels * height * width;

	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));

	this->createHandles();
}

/*constructor overload*/
poolLayer::poolLayer(string name, const param_tuple& args)
{
	std::tie(poolType, poolDim, pad_h, pad_w, stride_h,
			stride_w, inputImageDim, inputAmount) = args;

	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
	lrate = 0.0f;
    prevLayer.clear();
    nextLayer.clear();

    prev_num = config::instanceObjtce()->get_batchSize();
    prev_channels = inputAmount;
    prev_height = inputImageDim;
    prev_width = inputImageDim;
    number = prev_num;
    channels = prev_channels;
    height = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    width = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    outputSize = channels * height * width;
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));

    this->createHandles();
}


void poolLayer::forwardPropagation(string train_or_test)
{
	srcData = prevLayer[0]->dstData;

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
			                              prev_num,
			                              prev_channels,
			                              prev_height,
			                              prev_width));

    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
    		                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
    		                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
    		                              number,
    		                              channels,
    		                              height,
    		                              width));

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

   checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
		                                 cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
		                                 cuDNN_netWork<float>::instanceObject()->GetDataType(),
		                                 prev_num,
		                                 prev_channels,
		                                 prev_height,
		                                 prev_width));

   checkCUDNN(cudnnSetTensor4dDescriptor(dstDiffTensorDesc,
		                                 cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
		                                 cuDNN_netWork<float>::instanceObject()->GetDataType(),
		                                 prev_num,
		                                 prev_channels,
		                                 prev_height,
		                                 prev_width));


   float alpha = 1.0f;
   float beta = 0.0;
   checkCUDNN(cudnnPoolingBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
		                           poolingDesc,
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


void poolLayer:: destroyHandles()
{
	checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc))
	checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));
}
