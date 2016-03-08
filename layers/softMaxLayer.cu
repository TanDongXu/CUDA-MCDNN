#include"softMaxLayer.h"
#include"dataLayer.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"
#include"../saveData/saveNetWork.h"

#include"opencv2/imgproc/imgproc.hpp"
#include"opencv2/highgui/highgui.hpp"

using namespace cv;


void softMaxLayer::createHandles()
{
	/*cudnnCreateTensorDescriptor创建一个tensor对象（并没有初始化）*/
	checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}



void softMaxLayer::GetDataSize_BatchLabel()
{
	dataLayer* data_Layer = (dataLayer*) Layers::instanceObject()->getLayer("data");
	dataSize = data_Layer->getDataSize();
	srcLabel = data_Layer->getDataLabel();

}

softMaxLayer::softMaxLayer(string name)
{
	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	srcDiff = NULL;
	diffData = NULL;
	number = 0;
	channels =0;
	height = 0;
	width =0 ;
	dataSize = 0;
	srcLabel = NULL;
	nextLayer = NULL;
	prevLayer = NULL;
	flag = 1;
	lrate = 0.0f;
	CorrectSize = 0;
	cur_correctSize = 0;

	configSoftMax* curConfig = (configSoftMax*) config::instanceObjtce()->getLayersByName(_name);
	string prevLayerName = curConfig->_input;
	layersBase* prev_Layer =(layersBase*) Layers::instanceObject()->getLayer(prevLayerName);

	batchSize = config::instanceObjtce()->get_batchSize();
	inputSize = prev_Layer->getOutputSize();
	nclasses = curConfig->_nclasses;
	lambda = curConfig->_weight_decay;
	outputSize = nclasses;

	this->createHandles();

}


void softMaxLayer::ClassificationResults()
{
	if(flag == 1)
	{
		cur_correctSize = dataSize;
	}

	const int max_digit = nclasses;
	float *result;
	result = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height * width *sizeof(float));
	checkCudaErrors(cudaMemcpy(result, dstData, number * channels * height * width * sizeof(float),cudaMemcpyDeviceToHost));

	int temp = ((number < dataSize - flag) ? number : dataSize-flag);
	for(int i=0; i< temp; i++)
	{
		float max = result[i * max_digit];
		int labelIndex =0;
		/*选出10个中最大的一个代表该数字的输出*/
		for(int j=1; j<max_digit;j++)
		{
			if(max < result[i * max_digit + j])
			{
				max = result[i * max_digit + j];
				labelIndex = j;

			}
		}
		flag++;
		if(srcLabel[i] != labelIndex) --cur_correctSize;

	}

	if(flag == dataSize)
	{
		cout<<"Correct_Sizes: "<<cur_correctSize<<"/"<<CorrectSize;
		if(cur_correctSize > CorrectSize)
		{
			CorrectSize = cur_correctSize;
            saveNetWork();
		}
		flag = 1;
	}

	/*测试集的话把源输入释放，因为没有反向传导*/
	//MemoryMonitor::instanceObject()->freeGpuMemory(srcData);
	//MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeCpuMemory(result);
	//MemoryMonitor::instanceObject()->freeCpuMemory(srcLabel);
}


void softMaxLayer::forwardPropagation(string train_or_test)
{
	GetDataSize_BatchLabel();
	number = prevLayer->number;
	channels = prevLayer->channels;
	height = prevLayer->height;
	width = prevLayer->width;
	srcData = NULL;
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

	float alpha = 1.0;
	float beta = 0.0;

	checkCUDNN(cudnnSoftmaxForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                       CUDNN_SOFTMAX_FAST,
			                       CUDNN_SOFTMAX_MODE_CHANNEL,
			                       &alpha,
			                       srcTensorDesc,
			                       srcData,
			                       &beta,
			                       dstTensorDesc,
			                       dstData));


	if(train_or_test == "test" )
		ClassificationResults();

}


void softMaxLayer::Forward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(srcData);
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeCpuMemory(srcLabel);
}

__global__ void SoftmaxLossBackprop(const int* label, int num_labels, int batch_size, float* diffData)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size) return;
	const int label_value = label[idx];
	// For each item in the batch, decrease the result of the label's value by 1
	diffData[idx * num_labels + label_value] -= 1.0f;
}




/*求残差*/
void softMaxLayer::getBackPropDiffData()
{
	int *devLabel;
	devLabel = NULL;
	srcDiff = NULL;
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&srcDiff, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&devLabel, batchSize * 1 * 1 * 1 * sizeof(int));
	checkCudaErrors(cudaMemcpy(devLabel, srcLabel, batchSize * 1 * 1 * 1 * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(srcDiff, dstData, number * channels * height * width * sizeof(float), cudaMemcpyDeviceToDevice));

	SoftmaxLossBackprop<<< (batchSize + 127)/128, 128>>>(devLabel, nclasses, batchSize, srcDiff);
	cudaThreadSynchronize();
	MemoryMonitor::instanceObject()->freeGpuMemory(devLabel);
}



void softMaxLayer::backwardPropagation(float Momentum)
{
	/*获取残差*/
	getBackPropDiffData();
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

	diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, number * channels * height * width * sizeof(float));

	float alpha = 1.0f;
	float beta = 0.0f;

	/*computes the gridient of the softmax*/
	checkCUDNN(cudnnSoftmaxBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
			                        CUDNN_SOFTMAX_FAST,
			                        CUDNN_SOFTMAX_MODE_CHANNEL,
			                        &alpha,
			                        dstTensorDesc,
			                        dstData,
			                        srcDiffTensorDesc,
			                        srcDiff,
			                        &beta,
			                        dstDiffTensorDesc,
			                        diffData));

	//MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	//MemoryMonitor::instanceObject()->freeGpuMemory(srcDiff);
}



void softMaxLayer::Backward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeGpuMemory(srcDiff);
}

void softMaxLayer:: destroyHandles()
{
	/*销毁创建的描述符  逆向销毁*/
	checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));
}
