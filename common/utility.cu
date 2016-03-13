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



__global__ void MultiArrayAdd(float* array1, float* array2, float* array3, float* array4, float* outputs)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	outputs[idx] = array1[idx] + array2[idx] + array3[idx];
}
