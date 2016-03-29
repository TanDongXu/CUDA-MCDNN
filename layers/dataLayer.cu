#include"dataLayer.h"


/*constructor*/
dataLayer::dataLayer(string name)
{
	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	srcLabel = NULL;
	batchImage = NULL;
	dataSize = 0;
    prevLayer.clear();
    nextLayer.clear();

	batchSize = config::instanceObjtce()->get_batchSize();
	inputAmount = config::instanceObjtce()->getChannels();
	inputImageDim = config::instanceObjtce()->get_imageSize();

	number = config::instanceObjtce()->get_batchSize();
	channels = config::instanceObjtce()->getChannels();
	height = config::instanceObjtce()->get_imageSize();
	width = config::instanceObjtce()->get_imageSize();

	batchImage = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
	srcLabel = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(batchSize * 1 * 1 * 1 * sizeof(int));

}

//deep copy constructor
dataLayer::dataLayer(dataLayer* layer)
{
	srcData = NULL;
	dstData = NULL;
	srcLabel = NULL;
	batchImage = NULL;
	dataSize = 0;
	prevLayer.clear();
	nextLayer.clear();

	static int idx = 0;
	_name = layer->_name + int_to_string(idx);;
	idx ++;
	_inputName = layer->_inputName;
	batchSize = layer->batchSize;
	inputAmount = layer->inputAmount;
	inputImageDim = layer->inputImageDim;

	number = layer->number;
	channels = layer->channels;
	height = layer->height;
	width = layer->width;

	batchImage = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
	srcLabel = (int*) MemoryMonitor::instanceObject()->cpuMallocMemory(batchSize * 1 * 1 * 1 * sizeof(int));

	MemoryMonitor::instanceObject()->cpu2cpu(batchImage, layer->batchImage, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->cpu2cpu(srcLabel, layer->srcLabel, batchSize * 1 * 1 * 1 * sizeof(int));
	MemoryMonitor::instanceObject()->gpu2gpu(dstData, layer->dstData,  number * channels * height * width * sizeof(float));

	srcData = layer->dstData;
	//cout<<"data deep copy"<<endl;
}


void dataLayer::forwardPropagation(string train_or_test)
{
    //nothing
}

/*get batch size image*/
void dataLayer::getBatch_Images_Label(int index, 
                                      cuMatrixVector<float>&inputData, 
                                      cuMatrix<int>* &inputLabel)
{
	dataSize = inputData.size();
	int start = index * batchSize;

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
}


/*get batchSize image in random format*/
void dataLayer::RandomBatch_Images_Label(cuMatrixVector<float>&inputData, cuMatrix<int>*&inputLabel)
{
	dataSize = inputData.size();
	int randomNum = ((long)rand() + (long)rand()) % (inputData.size() - batchSize);

	for(int i = 0; i< batchSize; i++)
	{
		for(int c = 0; c < channels; c++)
		{
			for(int h = 0; h < height; h++)
			{
				for(int w = 0; w < width; w++)
				{
					batchImage[h * width + w + height * width * c + channels * height * width * i] =
							inputData[i + randomNum]->getValue(h, w, 0);
				}
			}
		}

		srcLabel[i] = inputLabel->getValue(i + randomNum, 0, 0);

	}

	checkCudaErrors(cudaMemcpy(dstData, batchImage, number *channels *height * width * sizeof(float), cudaMemcpyHostToDevice));
}


void dataLayer::backwardPropagation(float Momentum)
{
	//nothing
}
