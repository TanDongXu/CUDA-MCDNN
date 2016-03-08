#include"dataLayer.h"
#include"../config/config.h"
#include"layersBase.h"
#include"../cuDNN_netWork.h"

#include"opencv2/highgui/highgui.hpp"
#include<cstring>

using namespace cv;
dataLayer::dataLayer(string name)
{
	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	srcLabel = NULL;
	dataSize = 0;
	lrate = 0.0f;
	prevLayer = NULL;
	nextLayer = NULL;

	batchSize = config::instanceObjtce()->get_batchSize();
	_inputAmount = config::instanceObjtce()->getChannels();
	_outputAmount = _inputAmount;
	_inputImageDim = config::instanceObjtce()->get_imageSize();
	_outputImageDim = _inputImageDim;


	number = config::instanceObjtce()->get_batchSize();
	channels = config::instanceObjtce()->getChannels();
	height = config::instanceObjtce()->get_imageSize();
	width = config::instanceObjtce()->get_imageSize();

	//Layers::instanceObject()->storLayers(_name, this);
}

/*数据层的前向传导*/
void dataLayer::forwardPropagation(string train_or_test)
{
    //nothing
}



/*get batch size image*/
void dataLayer::getBatch_Images_Label(int index, cuMatrixVector<float>&inputData, cuMatrix<int>* &inputLabel)
{

	dataSize = inputData.size();
	int start = index * batchSize;

	float* batchImage;
	dstData = NULL;
	srcLabel = NULL;
	batchImage = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
	srcLabel = (int*) MemoryMonitor::instanceObject()->cpuMallocMemory(batchSize * 1 * 1 * 1 * sizeof(int));

	/*k保证batchImage大小都在一个batch大小内*/
	int k = 0;
	for (int i = start;i< (start + batchSize > inputData.size() ? inputData.size() : (start + batchSize)); i++) {
		for (int c = 0; c < channels; c++) {
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					batchImage[h * width + w + height * width * c + height * width * channels * k] =
							inputData[i]->getValue(h, w, 0);
				}
			}
		}

		srcLabel[k] = inputLabel->getValue(i, 0, 0);
		k++;
	}

	checkCudaErrors(cudaMemcpy(dstData, batchImage, number * channels * height * width * sizeof(float),cudaMemcpyHostToDevice));

	MemoryMonitor::instanceObject()->freeCpuMemory(batchImage);

}


/*get batchSize image in random format*/
void dataLayer::RandomBatch_Images_Label(cuMatrixVector<float>&inputData, cuMatrix<int>*&inputLabel)
{
	dataSize = inputData.size();
	int randomNum = ((long)rand() + (long)rand()) % (inputData.size() - batchSize);

	float * randomBatch;

	dstData = NULL;
	srcLabel = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
	srcLabel = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(batchSize * 1 * 1 * 1 * sizeof(int));
    randomBatch = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height * width * sizeof(float));

	for(int i = 0; i< batchSize; i++)
	{
		for(int c = 0; c < channels; c++)
		{
			for(int h = 0; h < height; h++)
			{
				for(int w = 0; w < width; w++)
				{
					randomBatch[h * width + w + height * width * c + channels * height * width * i] =
							inputData[i + randomNum]->getValue(h, w, 0);
				}
			}
		}

		srcLabel[i] = inputLabel->getValue(i + randomNum, 0, 0);

	}

	checkCudaErrors(cudaMemcpy(dstData, randomBatch, number *channels *height * width * sizeof(float), cudaMemcpyHostToDevice));
	MemoryMonitor::instanceObject()->freeCpuMemory(randomBatch);
}


void dataLayer::backwardPropagation(float Momentum)
{
//	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
//	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer->diffData);
}


void dataLayer::Backward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer->diffData);
}
