#include"utility.cuh"


/*展示GPU信息*/
void showDevices()
{
	int totalDevices;
	cudaGetDeviceCount(&totalDevices);
	std::cout<<"There are "<<totalDevices<<" CUDA capable devices on your machine: "<<std::endl;

	for(int i=0; i< totalDevices; i++)
	{
		struct cudaDeviceProp prop;
		checkCudaErrors(cudaGetDeviceProperties(&prop, i));
		/*multiProcessorCount设备上多处理器的数量*/
		printf( "device %d : sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, boardGroupID=%d\n",i, prop.multiProcessorCount, prop.major, prop.minor,
                (float)prop.clockRate*1e-3,
                (int)(prop.totalGlobalMem/(1024*1024)),
                (float)prop.memoryClockRate*1e-3,
                prop.ECCEnabled,
                prop.multiGpuBoardGroupID);
	}

	printf("\n");

}

/*
 * 多通道数组合并
 * blockId.x表示batch的ID(样本id)
 * blockId.y表示分支个数（合并数组的个数）
 * offset 表示通道的下标
 * curChannel 表示当前分支的通道数
 * */


__global__ void MultiChannelsMerge(float** inputs, float* outputs, int* channels, int* indexs, int row, int outChannels)
{
	int batchId  = blockIdx.x;
	int index = blockIdx.y;
	int offset = indexs[index];
	int curChannels = channels[index];

	/*得出某个分支输入*/
	float *input = inputs[index];
	/*计算输出的位置*/
	float* output = outputs + batchId * outChannels * row * row + offset;
	/*每个block的任务是负责每个样本某个分支的拷贝*/
	int blockDo = curChannels * row * row;
	for(int i = 0; i < blockDo; i += blockDim.x)
	{
		int j = i + threadIdx.x;
		if (j < blockDo)
		{
			/*计算该分支中某样本的起始位置*/
			int pos = batchId * curChannels * row * row;
			output[j] = input[pos + j];
		}
	}
}


/*多通道的划分*/
__global__ void MultiChannelsSplit(float* inputs, float**outputs, int* channels, int* indexs, int row, int inChannels)
{
	int batchId  = blockIdx.x;
	int index = blockIdx.y;
	int offset = indexs[index];
	int curChannels = channels[index];

	float* output = outputs[index];
	float* input = inputs + batchId * inChannels * row * row + offset;

	int blockDo = curChannels * row * row;
	for(int i = 0; i < blockDo; i += blockDim.x)
	{
		int j = i + threadIdx.x;
		if(j < blockDo)
		{
			int pos = batchId * curChannels * row * row;
			output[pos + j] = input[j];
		}
	}

}


/*从多通道中分出一个多通道分支*/
__global__ void MultiChannelsSplit(float* inputs, float* outputs, int outChannels, int offset, int row, int inChannels)
{
	int  batchId = blockIdx.x;
	float* input = inputs + batchId * inChannels * row * row + offset;

	int blockDo  = outChannels * row * row;
	for(int i = 0; i < blockDo; i += blockDim.x)
	{
		int j = i + threadIdx.x;
		if(j < blockDo)
		{
			int pos = batchId * outChannels * row * row;
			outputs[pos + j] = input[j];
		}
	}
}


/*多个数组相加*/
__global__ void MultiArrayAdd(float** inputs, float* outputs, int number,int channels, int height, int width)
{
	int blockDo = number * channels * height * width;
	for(int j = 0; j < 4; j++){
		float* input = inputs[j];
		for(int i = 0; i < blockDo; i += blockDim.x)
		{
			int idx = i + threadIdx.x;
			if(idx < blockDo)
			{
				outputs[idx] = outputs[idx] + input[idx];
			}
		}
	}
}
