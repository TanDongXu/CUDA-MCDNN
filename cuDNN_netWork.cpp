#include"./cuDNN_netWork.h"
#include<cuda_runtime.h>
#include<cudnn.h>
#include<cublas.h>
#include<cublas_v2.h>
#include"./common/checkError.h"



//#define ND_TENSOR_DESCRIPTOR

/*cudnnTensorDescriptor表示数据集描述
 * cudnnTesorFormat被cudnnSetTensor4dDescriptor所用来创建一个tensor
 * */
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, cudnnTensorFormat_t& tensorFormat, cudnnDataType_t& dataType,
		           int n,
		           int c,
		           int h,
		           int w){

/*条件编译指令，满足if 后面的条件就执行if后面代码*/
#if SIMPLE_TENSOR_DESCRIPTOR
	/*cudnnSetTensor4dDescriptor初始化一个先前创建的通用描述符为一个四维tensor
	 * 这四个纬度的步长是从格式参数推断出来的，并且以这一种方式设置，数据是连续的，
	 * 在内存中没有填充维度
	 * */
	checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc, tensorFormat, dataType, n, c, h, w));

#elif defined(ND_TENSOR_DESCRIPTOR)

	const int nDim = 4;
	int dimA[nDim] = {n,c,h,w};
	int strideA[nDim] = {c*h*w, h*w, w, 1};
	/*初始化一个通用tensor描述符，第四个参数表示每个维度的大小
	 * 第五个参数表示所表示的数组每个维度的步长
	 * */
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, dataType, 4, dimA, strideA));

#else
	/*tensor为4维，和上面第一个类似，但是步长是明确通过参数传递*/
	checkCUDNN(cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, c*h*w, h*w, w, 1));

#endif
}




#define DISABLE_GEMV

void matrixMulti(cublasHandle_t cublasHandle, int m, int n, int batchSize, float alpha,
		  const float*A, const float*x, float beta, float *y)
{
#ifdef DISABLE_GEMV
	checkCublasErrors(cublasSgemm(cublasHandle,
			          CUBLAS_OP_T,
			          CUBLAS_OP_T,
			          n,
			          batchSize,
			          m,
			          &alpha,
			          x,
			          m,
			          A,
			          batchSize,
			          &beta,
			          y,
			          n));

#else

	checkCublasErrors(cublasSgemv(cublasHandle, CUBLAS_OP_T,
			                     m, n,
			                     &alpha,
			                     A, m,
			                     x, 1,
			                     &beta,
			                     y, 1));


#endif

}







