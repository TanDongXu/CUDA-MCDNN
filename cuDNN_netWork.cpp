#include"./cuDNN_netWork.h"
#include<cuda_runtime.h>
#include<cudnn.h>
#include<cublas.h>
#include<cublas_v2.h>
#include"./common/checkError.h"



//#define ND_TENSOR_DESCRIPTOR

/*cudnn set tensor dim*/
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                   cudnnTensorFormat_t& tensorFormat, 
                   cudnnDataType_t& dataType,
		           int n,
		           int c,
		           int h,
		           int w){


#if SIMPLE_TENSOR_DESCRIPTOR
	/*cudnn set 4d tensor*/
	checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc, 
                                          tensorFormat, 
                                          dataType, 
                                                 n,
                                                 c,
                                                 h, 
                                                 w));

#elif defined(ND_TENSOR_DESCRIPTOR)

	const int nDim = 4;
	int dimA[nDim] = {n,c,h,w};
	int strideA[nDim] = {c*h*w, h*w, w, 1};
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, dataType, 4, dimA, strideA));

#else
	checkCUDNN(cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, c*h*w, h*w, w, 1));

#endif
}





/*matrixMulti*/

#define DISABLE_GEMV

void matrixMulti(cublasHandle_t cublasHandle, 
                                       int m, 
                                       int n,
                               int batchSize, 
                                 float alpha,
		                       const float*A, 
                               const float*x, 
                                  float beta, 
                                    float *y)
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







