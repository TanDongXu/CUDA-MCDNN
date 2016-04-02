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
	m_poolMethod = NULL;
    prevLayer.clear();
    nextLayer.clear();

	configPooling* curConfig = (configPooling*) config::instanceObjtce()->getLayersByName(_name);
	string prevLayerName = curConfig->_input;
	layersBase* prev_Layer = (layersBase*) Layers::instanceObject()->getLayer(prevLayerName);

	PoolingMode = (cudnnPoolingMode_t)curConfig->_poolType;
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
    printf("height %d width %d\n", prev_height, prev_width);
	height = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
	width = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    printf("height %d width %d\n", height, width);
	outputSize = channels * height * width;

	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));

	this->createHandles();
}

/*constructor overload*/
poolLayer::poolLayer(string name, const param_tuple& args)
{
	std::tie(pool_Type, poolDim, pad_h, pad_w, stride_h,
			stride_w, inputImageDim, inputAmount) = args;

	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
	lrate = 0.0f;
    prevLayer.clear();
    nextLayer.clear();

    m_poolMethod = new ConfigPoolMethod(pool_Type);
    PoolingMode = (cudnnPoolingMode_t)m_poolMethod->getValue();
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


//deep copy constructor
poolLayer::poolLayer(poolLayer* layer)
{
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
	m_poolMethod = NULL;
	prevLayer.clear();
	nextLayer.clear();

	static int idx = 0;
	_name = layer->_name + string("_") + int_to_string(idx);
	idx ++;
	_inputName = layer->_inputName;
	PoolingMode = layer->PoolingMode;
	poolDim = layer->poolDim;
	pad_h = layer->pad_h;
	pad_w = layer->pad_w;
	stride_h = layer->stride_h;
	stride_w = layer->stride_w;

	prev_num = layer->prev_num;
	prev_channels = layer->prev_channels;
	prev_height = layer->prev_height;
	prev_width = layer->prev_width;

	inputImageDim = layer->inputImageDim;
	inputAmount = layer->inputAmount;
	number = layer->number;
	channels = layer->channels;
	height = layer->height;
	width = layer->width;
	outputSize = layer->outputSize;

	MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));
	MemoryMonitor::instanceObject()->gpu2gpu(dstData, layer->dstData, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpu2gpu(diffData, layer->diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));

	this->createHandles();
	//cout<<"pool deep copy"<<endl;
}


void poolLayer::forwardPropagation(string train_or_test)
{
	srcData = prevLayer[0]->dstData;

	checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc,
										   PoolingMode,
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
   int nIndex = m_nCurBranchIndex;
   checkCUDNN(cudnnPoolingBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
		                           poolingDesc,
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


void poolLayer:: destroyHandles()
{
	checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc))
	checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));
}
