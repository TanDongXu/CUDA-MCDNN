#include"convLayer.h"



void convLayer::createHandles()
	{
		checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
		checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

		curandCreateGenerator(&curandGenerator_W, CURAND_RNG_PSEUDO_MTGP32);
		curandCreateGenerator(&curandGenerator_B, CURAND_RNG_PSEUDO_MTGP32);
	}

void convLayer::initRandom()
{
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Weight, 
                                                     kernelAmount * _inputAmount * kernelSize * kernelSize * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bias, kernelAmount * 1 * 1 * 1 * sizeof(float));
	//set seed
	curandSetPseudoRandomGeneratorSeed(curandGenerator_W, time(NULL));
	curandSetPseudoRandomGeneratorSeed(curandGenerator_B, time(NULL));
	curandGenerateNormal(curandGenerator_W, dev_Weight, kernelAmount *_inputAmount * kernelSize * kernelSize, 0, epsilon);
	curandGenerateNormal(curandGenerator_B, dev_Bias, kernelAmount, 0, epsilon);
}


/*convLayer constructor*/
convLayer::convLayer(string name, int sign)
{
	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	host_Weight = NULL;
	dev_Weight = NULL;
	host_Bias = NULL;
	dev_Bias = NULL;
	dev_Wgrad = NULL;
	dev_Bgrad = NULL;
	diffData = NULL;
	number = 0;
	channels = 0;
	height = 0;
	width = 0;
    prevLayer.clear();
    nextLayer.clear();

    /*can use class member prevLayer, because it has not assignment*/
	configConv* curConfig = (configConv*) config::instanceObjtce()->getLayersByName(_name);
    string prevLayerName = curConfig->_input;
    convLayerBase* prev_Layer = (convLayerBase*) Layers::instanceObject()->getLayer(prevLayerName);

    epsilon = curConfig->_init_w;
    lrate = curConfig->_lrate;
    batchSize = config::instanceObjtce()->get_batchSize();
    kernelAmount = curConfig->_kernelAmount;
    kernelSize = curConfig->_kernelSize;
    pad_h = curConfig->_pad_h;
    pad_w = curConfig->_pad_w;
    stride_h = curConfig->_stride_h;
    stride_w = curConfig->_stride_w;
    lambda = curConfig->_weight_decay;

    _inputAmount = prev_Layer->_outputAmount;
    _outputAmount = kernelAmount;
    _inputImageDim = prev_Layer->_outputImageDim;
    _outputImageDim = (_inputImageDim + 2 * pad_h - kernelSize)/stride_h + 1;

    outputSize = _outputAmount * _outputImageDim * _outputImageDim;


    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Wgrad, 
                                                     kernelAmount * _inputAmount * 1 * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Wgrad, 
                                                     kernelAmount * _inputAmount * 1 * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));

    this->createHandles();
    if(sign == RANDOM)
    	this->initRandom();
}


/*conv constructor overload*/
convLayer::convLayer(string name, int sign, const param_tuple& args)
{
	std::tie(pad_h, pad_w, stride_h, stride_w, kernelSize,
			kernelAmount, _inputAmount, _inputImageDim,
			epsilon, lrate, lambda) = args;

	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	host_Weight = NULL;
	host_Bias = NULL;
	dev_Weight = NULL;
	dev_Bias = NULL;
	dev_Wgrad = NULL;
	dev_Bgrad = NULL;
	diffData = NULL;
	number = 0;
	channels = 0;
	height = 0;
	width = 0;
    prevLayer.clear();
    nextLayer.clear();

	_outputAmount = kernelAmount;
	_outputImageDim = (_inputImageDim + 2 * pad_h - kernelSize)/stride_h + 1;
	outputSize = _outputAmount * _outputImageDim * _outputImageDim;

	 MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Wgrad, 
                                                      kernelAmount * _inputAmount * kernelSize * kernelSize * sizeof(float));
	 MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
	 MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Wgrad, 
                                                      kernelAmount * _inputAmount * kernelSize * kernelSize * sizeof(float));
	 MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));

	 this->createHandles();
	 if(sign == RANDOM)
	    	this->initRandom();
}


void convLayer::addBias(const cudnnTensorDescriptor_t& dstTensorDesc, int c, float *data )
{
    
    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
			                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
				                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
				                          1,
                                          c,
				                          1,
				                          1));


	float alpha = 1.0;
	float beta = 1.0;
	checkCUDNN(cudnnAddTensor(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                  CUDNN_ADD_SAME_C,
			                  &alpha,
			                  biasTensorDesc,
			                  dev_Bias,
			                  &beta,
			                  dstTensorDesc,
			                  data));
}



void convLayer::forwardPropagation(string train_or_test)
{
	srcData = NULL;
	number = prevLayer[0]->number;
	channels = prevLayer[0]->channels;
	height = prevLayer[0]->height;
	width = prevLayer[0]->width;
	srcData = prevLayer[0]->dstData;

	checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
			                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
			                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
			                              number,
			                              channels,
			                              height,
			                              width));


	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
			                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
			                              kernelAmount,
			                              _inputAmount,
			                              kernelSize,
			                              kernelSize));

	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
			                                   pad_h,
			                                   pad_w,//pading
			                                   stride_h,
			                                   stride_w,//stride
			                                   1,1,//upscale
			                                   CUDNN_CROSS_CORRELATION));


	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
			                                         srcTensorDesc,
			                                         filterDesc,
			                                         &number,
			                                         &channels,
			                                         &height,
			                                         &width));

	checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
			                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
			                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
			                              number,
			                              channels,
			                              height,
			                              width));

	if (cuDNN_netWork<float>::instanceObject()->GetconvAlgorithm() < 0)
	{

	    //std::cout<< "Testing cudnnGetConvolutionForwardAlgorithm ..."<<std::endl;

	    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
						                               srcTensorDesc,
						                               filterDesc,
						                               convDesc,
						                               dstTensorDesc,
						                               CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
						                               0,
						                               &algo));

	    cuDNN_netWork<float>::instanceObject()->setConvolutionAlgorithm(algo);
//		std::cout<<"Fastest algorithm is Algo: "<<algo<<std::endl;
//		std::cout<<"Testing cudnnFindConvolutionForwardAlgorithm ..."<<std::endl;
//		/*测试算法的总数：5个算法*/
//		int requestedAlgoCount = 5;
//		/*返回算法的个数*/
//		int returnedAlgoCount[1];
//
//		/*result 保存的是cudnnFindConvolutionForwardAlgorithm()返回包含结构性能结果：包含5个算法的测试结果*/
//		cudnnConvolutionFwdAlgoPerf_t *results =(cudnnConvolutionFwdAlgoPerf_t*) malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount);
//
//		/*该函数尝试cudnn所有算法，结果输出到用户分配的数组result中，
//		 * 这些指标是按照排序的方式编写，其中第一个元素（算法）具有最低的计算时间
//		 * */
//
//		checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
//						                                srcTensorDesc,
//						                                filterDesc,
//						                                convDesc,
//						                                dstTensorDesc,
//						                                requestedAlgoCount,
//						                                returnedAlgoCount,
//						                first_ShareLayer = new ShareLayer("share1", prevLayer);
	    //
	    //    /*the first layer is share layer*/
	    //	for(int i = 0; i < 4; i++)
	    //	{
	    //		sprintf(branch, "branch_%d", i);
	    //		InnerLayers[i].storLayers(branch, new ShareLayer(branch, prevLayer));
	    //	}                 results));
//
//
//		for (int algoIndex = 0; algoIndex < *returnedAlgoCount; ++algoIndex) {
//			printf(".... %s for Algo %d: %f time requiring %llu memory\n",
//					cudnnGetErrorString(results[algoIndex].status),
//					results[algoIndex].algo,
//					results[algoIndex].time,
//					(unsigned long long) results[algoIndex].memory);
//		}
//
//		free(results);

	}else
	{
		algo =(cudnnConvolutionFwdAlgo_t)cuDNN_netWork<float>::instanceObject()->GetconvAlgorithm();
	}


	this->dstData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width *sizeof(float));

	size_t sizeInBytes = 0;
	void* workSpace =NULL;


	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
					                                   srcTensorDesc,
					                                   filterDesc,
					                                   convDesc,
					                                   dstTensorDesc,
					                                   algo,
					                                   &sizeInBytes));

	if (sizeInBytes != 0)
	{
		checkCudaErrors(cudaMalloc(&workSpace, sizeInBytes));
	}



	float alpha = 1.0f;
	float beta = 0.0f;
	checkCUDNN(cudnnConvolutionForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
					                   &alpha,
					                   srcTensorDesc,
					                   srcData,
					                   filterDesc,
					                   dev_Weight,
					                   convDesc,
					                   algo,
					                   workSpace,
					                   sizeInBytes,
					                   &beta,
					                   dstTensorDesc,
					                   dstData));


	/*add bias*/
	addBias(dstTensorDesc, channels, dstData);

	if (sizeInBytes != 0)
	{
		checkCudaErrors(cudaFree(workSpace));
	}

}

/*free forward memory*/
void convLayer::Forward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(srcData);
}


void convLayer::backwardPropagation(float Momentum)
{
	float *tmp_Wgrad = NULL , *tmp_Bgrad = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tmp_Wgrad, 
                                                     kernelAmount * _inputAmount * 1 * kernelSize * kernelSize * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tmp_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));

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


	int prevlayer_n, prevlayer_c, prevlayer_h, prevlayer_w;
	prevlayer_n = prevLayer[0]->number;
	prevlayer_c = prevLayer[0]->channels;
	prevlayer_h = prevLayer[0]->height;
	prevlayer_w = prevLayer[0]->width;


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


	float alpha = 1.0f;
	float beta = 0.0f;
	checkCUDNN(cudnnConvolutionBackwardBias(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                                &alpha,
			                                srcDiffTensorDesc,
			                                nextLayer[0]->diffData,
			                                &beta,
			                                biasTensorDesc,
			                                tmp_Bgrad
			                                ));


	checkCUDNN(cudnnConvolutionBackwardFilter(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                                  &alpha,
			                                  srcTensorDesc,
			                                  srcData,
			                                  srcDiffTensorDesc,
			                                  nextLayer[0]->diffData,
			                                  convDesc,
			                                  &beta,
			                                  filterDesc,
			                                  tmp_Wgrad));

	alpha = lambda * batchSize;
	int size =  kernelAmount * _inputAmount * kernelSize * kernelSize;
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
						          size,
						          &alpha,
						          dev_Weight,
						          1,
						          tmp_Wgrad,
						          1));



	/*compute diff*/
	diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, 
                                                     prevlayer_n * prevlayer_c * prevlayer_h * prevlayer_w * sizeof(float));

	alpha = 1.0f;
	beta = 0.0f;
	checkCUDNN(cudnnConvolutionBackwardData(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                                &alpha,
			                                filterDesc,
			                                dev_Weight,
			                                srcDiffTensorDesc,
			                                nextLayer[0]->diffData,
			                                convDesc,
			                                &beta,
			                                dstDiffTensorDesc,
			                                diffData));

	float scalVal = Momentum;
	size =  kernelAmount * _inputAmount * kernelSize * kernelSize;
	checkCublasErrors(cublasSscal(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
						          size,
						          &scalVal,
						          dev_Wgrad,
						          1));

	size = kernelAmount * 1 * 1 * 1;
	checkCublasErrors(cublasSscal(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
							      size,
							      &scalVal,
							      dev_Bgrad,
							      1));

	scalVal =lrate * 1.0f / batchSize;
	size =  kernelAmount * _inputAmount * kernelSize * kernelSize;
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                  size,
				                  &scalVal,
				                  tmp_Wgrad,
				                  1,
				                  dev_Wgrad,
				                  1));

	size = kernelAmount * 1 * 1 * 1;
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
					              size,
					              &scalVal,
					              tmp_Bgrad,
					              1,
					              dev_Bgrad,
					              1));
	
    alpha = -1.0f;
	size =  kernelAmount * _inputAmount * kernelSize * kernelSize;
    checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
					                  size,
					                  &alpha,
					                  dev_Wgrad,
					                  1,
					                  dev_Weight,
					                  1));

	size = kernelAmount * 1 * 1 * 1;
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
					                  size,
					                  &alpha,
					                  dev_Bgrad,
					                  1,
					                  dev_Bias,
					                  1));


	MemoryMonitor::instanceObject()->freeGpuMemory(tmp_Wgrad);
    MemoryMonitor::instanceObject()->freeGpuMemory(tmp_Bgrad);
}


/*free backwardPropagation memory*/
void convLayer::Backward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer[0]->diffData);
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
}


void convLayer::saveWeight(FILE*file)
{
	host_Weight = NULL; host_Bias = NULL;
	copy_DeviceToHost(dev_Weight, host_Weight, kernelAmount, _inputAmount, kernelSize, kernelSize);
	copy_DeviceToHost(dev_Bias, host_Bias, 1, 1, 1, kernelAmount);

	for(int n = 0 ; n < kernelAmount; n++)
	{
		for(int c = 0; c < _inputAmount; c++)
		{
			for(int h = 0; h < kernelSize; h++)
			{
				for(int w = 0; w < kernelSize; w++)
				{
				   fprintf(file, "%f ", host_Weight[w + h * kernelSize + kernelSize * kernelSize * c + kernelSize * kernelSize * _inputAmount * n]);
				}
			}
		}
	}

	for(int n = 0; n < kernelAmount; n++)
	{
		fprintf(file, "%f ", host_Bias[n]);
	}

	MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
	MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);
}



void convLayer::readWeight(FILE*file)
{
	host_Weight = NULL; host_Bias = NULL;
	dev_Weight = NULL; dev_Bias = NULL;
	host_Weight = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(kernelAmount * _inputAmount * kernelSize * kernelSize * sizeof(float));
    host_Bias = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(kernelAmount * 1 * 1 * 1 * sizeof(float));

    for(int n = 0 ; n < kernelAmount; n++)
    	{
    		for(int c = 0; c < _inputAmount; c++)
    		{
    			for(int h = 0; h < kernelSize; h++)
    			{
    				for(int w = 0; w < kernelSize; w++)
    				{
    				   fscanf(file, "%f", &host_Weight[w + h * kernelSize + kernelSize * kernelSize * c + kernelSize * kernelSize * _inputAmount * n]);
    				}
    			}
    		}
    	}

	for (int n = 0; n < kernelAmount; n++)
	{
		fscanf(file, "%f", &host_Bias[n]);
	}


	copy_HostToDevice(host_Weight, dev_Weight, kernelAmount, _inputAmount, kernelSize, kernelSize);
    copy_HostToDevice(host_Bias, dev_Bias, 1, 1, 1, kernelAmount);

    MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
    MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);
}


void convLayer:: destroyHandles()
{
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));
	curandDestroyGenerator(curandGenerator_W);
	curandDestroyGenerator(curandGenerator_B);
}






