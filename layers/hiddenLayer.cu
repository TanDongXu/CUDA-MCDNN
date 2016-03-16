#include"hiddenLayer.h"



void hiddenLayer::createHandles()
{
	curandCreateGenerator(&curandGenerator_W, CURAND_RNG_PSEUDO_MTGP32);
	curandCreateGenerator(&curandGenerator_B, CURAND_RNG_PSEUDO_MTGP32);
}

void hiddenLayer::destroyHandles()
{
	curandDestroyGenerator(curandGenerator_W);
	curandDestroyGenerator(curandGenerator_B);
}



void hiddenLayer::initRandom()
{
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Weight, outputSize * inputSize * 1 * 1 * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bias, outputSize * 1 * 1 * 1 * sizeof(float));
	/*initial weight*/
	curandSetPseudoRandomGeneratorSeed(curandGenerator_W, time(NULL));
	curandSetPseudoRandomGeneratorSeed(curandGenerator_B, time(NULL));
	curandGenerateNormal(curandGenerator_W, dev_Weight, outputSize * inputSize, 0, epsilon);
	curandGenerateNormal(curandGenerator_B, dev_Bias, outputSize, 0, epsilon);

}


/*fill a float-point array with one*/
__global__ void FillOnes(float* vec, int value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx > value) return ;

	vec[idx] = 1.0f;
}

/*constructor*/
hiddenLayer::hiddenLayer(string name, int sign)
{
	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
	host_Weight = NULL;
	dev_Weight = NULL;
	host_Bias = NULL;
	dev_Bias = NULL;
	dev_Wgrad = NULL;
	dev_Bgrad = NULL;
	VectorOnes = NULL;
	number = 0;
	channels = 0;
	height = 0;
	width = 0;
	prevLayer = NULL;
	nextLayer = NULL;

	configHidden * curConfig = (configHidden*) config::instanceObjtce()->getLayersByName(_name);
	string preLayerName = curConfig->_input;
	layersBase* prev_Layer = (layersBase*) Layers::instanceObject()->getLayer(preLayerName);

	epsilon = curConfig->_init_w;
	lrate = curConfig->_lrate;
	inputSize = prev_Layer->getOutputSize();
	outputSize = curConfig->_NumHiddenNeurons;
	nonLinearity = curConfig->_non_linearity;
	batchSize = config::instanceObjtce()->get_batchSize();
	lambda = curConfig->_weight_decay;

	//1*batchSize
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&VectorOnes, 1 * 1 * 1 * batchSize* sizeof(float));
	FillOnes<<<1, batchSize>>>(VectorOnes, batchSize);

	MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dev_Wgrad,1 * 1 * outputSize * inputSize * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dev_Bgrad,1 * 1 * outputSize * 1 * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Wgrad, 1 * 1 * outputSize * inputSize * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bgrad, 1 * 1 * outputSize * 1 * sizeof(float));

	this->createHandles();
	if(sign == RANDOM)
		this->initRandom();
}


void hiddenLayer::forwardPropagation(string train_or_test)
{
	number = prevLayer->number;
	channels = prevLayer->channels;
	height = prevLayer->height;
	width = prevLayer->width;
	srcData = prevLayer->dstData;

	int dim_x = channels * height * width ;
	int dim_y = outputSize ;

	dstData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, dim_y * batchSize * sizeof(float));

	float alpha = 1.0f;
	float beta = 0.0f;

	checkCublasErrors(cublasSgemm(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                  CUBLAS_OP_T,
				                  CUBLAS_OP_N,
				                  dim_y,
				                  batchSize,
				                  dim_x,
				                  &alpha,
				                  dev_Weight,
				                  dim_x,
				                  srcData,
				                  dim_x,
				                  &beta,
				                  dstData,
				                  dim_y));


    //add bias
	alpha = 1.0f;
	beta = 1.0f;
	checkCublasErrors(cublasSgemm(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                  CUBLAS_OP_N,
				                  CUBLAS_OP_N,
				                  dim_y,
				                  batchSize,
				                  1,
				                  &alpha,
				                  dev_Bias,
				                  dim_y,
				                  VectorOnes,
				                  1,
				                  &beta,
				                  dstData,
				                  dim_y));

	height = 1; width = 1; channels = dim_y;

}


/*free forwardPropagation memory*/
void hiddenLayer::Forward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(srcData);
}


void hiddenLayer::backwardPropagation(float Momentum)
{
	int prevlayer_c, prevlayer_h, prevlayer_w;
	prevlayer_c = prevLayer->channels;
	prevlayer_h = prevLayer->height;
	prevlayer_w = prevLayer->width;


	int dim_x = prevlayer_c * prevlayer_h * prevlayer_w;
	int dim_y = outputSize;


	float*tmp_Wgrad = NULL, *tmp_Bgrad = NULL;

	MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &tmp_Wgrad,1 * 1 * outputSize * inputSize * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &tmp_Bgrad,1 * 1 * outputSize * 1 * sizeof(float));

	checkCudaErrors(cudaMemcpy(tmp_Wgrad, dev_Weight, 1 * 1 * outputSize * inputSize * sizeof(float), cudaMemcpyDeviceToDevice));

	float alpha = 1.0f /(float)batchSize;
	float beta = lambda;
	checkCublasErrors(cublasSgemm(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                  CUBLAS_OP_N,
				                  CUBLAS_OP_T,
				                  dim_x,
				                  dim_y,
				                  batchSize,
				                  &alpha,
				                  srcData,
				                  dim_x,
				                  nextLayer->diffData,
				                  dim_y,
				                  &beta,
				                  tmp_Wgrad,
				                  dim_x));


	beta = 0.0f;
	checkCublasErrors(cublasSgemv(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
			                      CUBLAS_OP_N,
				                  outputSize,
				                  batchSize,
				                  &alpha,
				                  nextLayer->diffData,
				                  outputSize,
				                  VectorOnes,
				                  1,
				                  &beta,
				                  tmp_Bgrad,
				                  1));

	diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData,  inputSize * batchSize* sizeof(float));

	alpha = 1.0f;
	beta = 0.0f;
	checkCublasErrors(cublasSgemm(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
			                      CUBLAS_OP_N,
			                      CUBLAS_OP_N,
			                      dim_x,
				                  batchSize,
				                  outputSize,
				                  &alpha,
				                  dev_Weight,
				                  dim_x,
				                  nextLayer->diffData,
				                  outputSize,
				                  &beta,
				                  diffData,
				                  dim_x));




	float scalVal = Momentum;
	int size = 1 * 1 * outputSize * inputSize * 1;
	checkCublasErrors(cublasSscal(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
							      size,
							      &scalVal,
							      dev_Wgrad,
							      1));


	size = 1 * 1 * outputSize * 1 * 1;
	checkCublasErrors(cublasSscal(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
								  size,
								  &scalVal,
								  dev_Bgrad,
								  1));


	scalVal = lrate;
	size = 1 * 1 * outputSize * inputSize * 1;
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
					                  size,
					                  &scalVal,
					                  tmp_Wgrad,
					                  1,
					                  dev_Wgrad,
					                  1));

	size = outputSize * 1 * 1 * 1;
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
						          size,
						          &scalVal,
						          tmp_Bgrad,
						          1,
						          dev_Bgrad,
						          1));



	/*updata weightt*/
	alpha = -1.0f;
	size = outputSize * inputSize;

	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                  size,
				                  &alpha,
				                  dev_Wgrad,
				                  1,
				                  dev_Weight,
				                  1));

	size = outputSize;
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                  size,
				                  &alpha,
				                  dev_Bgrad,
				                  1,
				                  dev_Bias,
				                  1));


	MemoryMonitor::instanceObject()->freeGpuMemory(tmp_Bgrad);
	MemoryMonitor::instanceObject()->freeGpuMemory(tmp_Wgrad);
}

/*free backwardPropagation memory*/
void hiddenLayer::Backward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer->diffData);
}

void hiddenLayer::saveWeight(FILE*file)
{
	host_Weight = NULL, host_Bias = NULL;
	copy_DeviceToHost(dev_Weight, host_Weight, 1, 1,outputSize, inputSize);
	copy_DeviceToHost(dev_Bias, host_Bias, 1, 1, 1, outputSize);

	for(int h = 0; h < outputSize; h++)
	{
		for(int w = 0; w < inputSize; w++)
		{
			fprintf(file, "%f ", host_Weight[w + inputSize * h]);
		}
	}

	for(int h = 0; h < outputSize; h++)
	{
		fprintf(file, "%f ", host_Bias[h]);
	}

	MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
	MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);

}


void hiddenLayer::readWeight(FILE*file)
{
	host_Weight = NULL; host_Bias = NULL;
	dev_Weight = NULL; dev_Bias = NULL;

	host_Weight = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(outputSize * inputSize * sizeof(float));
	host_Bias = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(outputSize * 1 * 1 * 1 * sizeof(float));

	for (int h = 0; h < outputSize; h++) {
		for (int w = 0; w < inputSize; w++) {
			fscanf(file, "%f", &host_Weight[w + inputSize * h]);
		}
	}

	for (int h = 0; h < outputSize; h++) {
		fscanf(file, "%f", &host_Bias[h]);
	}

	copy_HostToDevice(host_Weight, dev_Weight, 1, 1, outputSize, inputSize);
	copy_HostToDevice(host_Bias, dev_Bias, 1, 1, 1, outputSize);

	MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
	MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);
}
