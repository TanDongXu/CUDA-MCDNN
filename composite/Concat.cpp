#include"Concat.h"

Concat::Concat(Layers*& Inner_Layers, const param_tuple& args)
{
	std::tie(one, three, five, pool_proj) = args;
	number = 0;
	channels = 0;
	height = 0;
	width = 0;
	prev_number = 0;
	prev_channels = 0;
	prev_height = 0;
	prev_width = 0;
	oneDim = 0;
	threeDim = 0;
	fiveDim = 0;
	pool_projDim = 0;
	one_outResult = NULL;
	three_outResult = NULL;
	five_outResult = NULL;
	pool_proj_outResult = NULL;
	outputResult = NULL;
	one_diff = NULL;
	three_diff = NULL;
	five_diff = NULL;
	pool_proj_diff = NULL;
	diffData = NULL;

	channels = one + three + five + pool_proj;
	InnerLayers = Inner_Layers;

}


float* Concat::forwardSetup()
{
	number = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->number;
	height = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->height;
	width = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->width;


	oneDim = number * one * height * width;
	threeDim = number * three * height * width;
	fiveDim = number * five * height * width;
	pool_projDim = number * pool_proj * height * width;

	one_outResult = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(InnerLayers[0].getLayersNum() -1))->dstData;
	three_outResult = InnerLayers[1].getLayer(InnerLayers[1].getLayersName(InnerLayers[1].getLayersNum() -1))->dstData;
	five_outResult = InnerLayers[2].getLayer(InnerLayers[2].getLayersName(InnerLayers[2].getLayersNum() -1))->dstData;
	pool_proj_outResult = InnerLayers[3].getLayer(InnerLayers[3].getLayersName(InnerLayers[3].getLayersNum() -1))->dstData;

	outputResult = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&outputResult, number * channels * height * width * sizeof(float));



	checkCudaErrors(cudaMemcpy(outputResult, one_outResult, oneDim * sizeof(float), cudaMemcpyDeviceToDevice ));
	checkCudaErrors(cudaMemcpy(outputResult + oneDim, three_outResult, threeDim * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(outputResult + oneDim + threeDim, five_outResult, fiveDim * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(outputResult + oneDim + threeDim + fiveDim, pool_proj_outResult, pool_projDim * sizeof(float), cudaMemcpyDeviceToDevice));



	return outputResult;
}


float* Concat::backwardSetup()
{
	/*the first share layer no need compute here*/
	one_diff = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->diffData;
	three_diff = InnerLayers[1].getLayer(InnerLayers[1].getLayersName(0))->diffData;
	five_diff = InnerLayers[2].getLayer(InnerLayers[2].getLayersName(0))->diffData;
	pool_proj_diff = InnerLayers[3].getLayer(InnerLayers[3].getLayersName(0))->diffData;

	prev_number = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->number;
	prev_channels = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->channels;
	prev_height = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->height;
	prev_width = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->width;

	int size = prev_number * prev_channels * prev_height * prev_width;

	diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, size * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(diffData, size * sizeof(float));

	float alpha = 1.0;

	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
			                      size,
			                      &alpha,
			                      one_diff,
			                      1,
			                      diffData,
			                      1));

	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                      size,
				                      &alpha,
				                      three_diff,
				                      1,
				                      diffData,
				                      1));
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                      size,
				                      &alpha,
				                      five_diff,
				                      1,
				                      diffData,
				                      1));
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                      size,
				                      &alpha,
				                      pool_proj_diff,
				                      1,
				                      diffData,
				                      1));



	return diffData;

}
