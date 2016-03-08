#include<iostream>
#include"convLayer.h"
#include"../config/config.h"
#include"../common/cuMatrix.h"
#include"../common/MemoryMonitor.h"
#include"../common/checkError.h"
#include<time.h>

#include"opencv2/highgui.hpp"
#include"opencv2/core/core.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include"../tests/test_layer.h"

using namespace cv;

void convLayer::createHandles()
	{
		/*cudnnCreateTensorDescriptor创建一个tensor对象（并没有初始化）*/
		checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
		/*cudnnFilterDescriptor创建一个过滤器对象（没有初始化）*/
		checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
		/*cudnnCreateConvolutionDescriptor创建一个卷积层对象（没初始化）*/
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

		//创建一个新的目标类型（参考 触发器类型）触发器
		curandCreateGenerator(&curandGenerator_W, CURAND_RNG_PSEUDO_MTGP32);
		curandCreateGenerator(&curandGenerator_B, CURAND_RNG_PSEUDO_MTGP32);
	}

void convLayer::initRandom()
{
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Weight, kernelAmount * _inputAmount * kernelSize * kernelSize * sizeof(float));
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
	prevLayer = NULL;
	nextLayer = NULL;

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

    /*inputAmount和outputAmount和本曾的channels相同*/
    _inputAmount = prev_Layer->_outputAmount;
    _outputAmount = kernelAmount;
    _inputImageDim = prev_Layer->_outputImageDim;
    _outputImageDim = _inputImageDim - kernelSize + 1;

    outputSize = _outputAmount * _outputImageDim * _outputImageDim;
    non_linearity = curConfig->_non_linearity;


    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Wgrad, kernelAmount * _inputAmount * 1 * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Wgrad, kernelAmount * _inputAmount * 1 * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));

    this->createHandles();
    if(sign == RANDOM)
    	this->initRandom();
}

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
	prevLayer = NULL;
	nextLayer = NULL;

	_outputAmount = kernelAmount;
	_outputImageDim = _inputImageDim - kernelSize + 1;
	outputSize = _outputAmount * _outputImageDim * _outputImageDim;

	 MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Wgrad, kernelAmount * _inputAmount * kernelSize * kernelSize * sizeof(float));
	 MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
	 MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Wgrad, kernelAmount * _inputAmount * kernelSize * kernelSize * sizeof(float));
	 MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));

	 this->createHandles();
	 if(sign == RANDOM)
	    	this->initRandom();
}


/*加上偏置*/
void convLayer::addBias(const cudnnTensorDescriptor_t& dstTensorDesc, int c, float *data )
{
		/*设置偏置的维度*/
		checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
				                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
				                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
				                              1,c,
				                              1,
				                              1));

		/*scaling参数*/
		float alpha = 1.0;
		/*指数*/
		float beta = 1.0;
		/*增加一个tensor的缩放值到另一个tensor,意思就是卷积后以某种方式加上偏置
		 * 第二个参数是添加的模式
		 * 第四个参数是偏置的tensor,第五个参数是偏置在内存中的位置
		 * 第七个参数是偏置要添加到的tensor，第八个是偏置要添加到的数据
		 * */
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
	number = prevLayer->number;
	channels = prevLayer->channels;
	height = prevLayer->height;
	width = prevLayer->width;
	srcData = prevLayer->dstData;

    /*设置srcTensorDesc的维度*/
	checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
			                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
			                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
			                              number,
			                              channels,
			                              height,
			                              width));


	/*卷积核的配置*/
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


	/*给dstDataGPU分配内存,用来存储结果,分配GPU是一定确保指向为空*/
	this->dstData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width *sizeof(float));

	size_t sizeInBytes = 0;
	void* workSpace =NULL;

	/*该函数返回用户根据指定算法需要的分配的GPU内存工作区
	 * 最后一个参数存储算法所需要GPU内存的数量,
	 * */

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
	/*执行卷积运算，根据指定的src数据和srctensor以及卷积核进行运算
	 * 结果返回存储在dst中，alpha通常用来缩放输入tensor，
	 * beta通常用来缩放输出tensor
	 * 第三个参数开始依次是：原数据tensor，原数据在GPU地址，卷积核tensor,卷积核权重正在GPU地址
	 * 卷积描述符，指定的卷积算法，执行指定算法所需要的内存，workSpace大小，beta，
	 * 目的tensor，卷积后的结果
	 * */
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


	/*加上偏置*/
	addBias(dstTensorDesc, channels, dstData);

	if (sizeInBytes != 0)
	{
		checkCudaErrors(cudaFree(workSpace));
	}

//	if(train_or_test == "test")
//	    MemoryMonitor::instanceObject()->freeGpuMemory(srcData);

}


void convLayer::Forward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(srcData);
}


void convLayer::backwardPropagation(float Momentum)
{
	/*使用Moment用*/
	float *tmp_Wgrad = NULL , *tmp_Bgrad = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tmp_Wgrad, kernelAmount * _inputAmount * 1 * kernelSize * kernelSize * sizeof(float));
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

	/*计算偏置的梯度*/
	float alpha = 1.0f;
	float beta = 0.0f;
	checkCUDNN(cudnnConvolutionBackwardBias(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                                &alpha,
			                                srcDiffTensorDesc,
			                                nextLayer->diffData,
			                                &beta,
			                                biasTensorDesc,
			                                tmp_Bgrad
			                                ));


	/*计算权重梯度*/
	checkCUDNN(cudnnConvolutionBackwardFilter(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                                  &alpha,
			                                  srcTensorDesc,
			                                  srcData,
			                                  srcDiffTensorDesc,
			                                  nextLayer->diffData,
			                                  convDesc,
			                                  &beta,
			                                  filterDesc,
			                                  tmp_Wgrad));

	/*加上权重衰减项:这是根据公式做的变化，因为下面跟新参数又除以batchSize,所以这里要乘*/
	alpha = lambda * batchSize;
	int size =  kernelAmount * _inputAmount * kernelSize * kernelSize;
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
						          size,
						          &alpha,
						          dev_Weight,
						          1,
						          tmp_Wgrad,
						          1));



	/*计算本曾残差*/
	diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, prevlayer_n * prevlayer_c * prevlayer_h * prevlayer_w * sizeof(float));

	alpha = 1.0f;
	beta = 0.0f;
	checkCUDNN(cudnnConvolutionBackwardData(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                                &alpha,
			                                filterDesc,
			                                dev_Weight,
			                                srcDiffTensorDesc,
			                                nextLayer->diffData,
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

	/*权重更新，新W= 旧W-rate*(1/m)*偏导数*/
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
	/*更新w = w - wgrad*/
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
//	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer->diffData);
//	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
}



void convLayer::Backward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer->diffData);
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
	/*销毁创建的描述符  逆向销毁*/
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






