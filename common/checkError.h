/*
 * checkError.h
 *
 *  Created on: Dec 8, 2015
 *      Author: tdx
 */

#ifndef CHECKERROR_H_
#define CHECKERROR_H_

#include<stdlib.h>
#include<cuda_runtime.h>
#include<iostream>
#include<sstream>


/*__FILE__用以指示本行语句所在源文件的文件名
 * __LINE__用以指示本行语句在源文件中的位置信息
 * */
#define FatalError(s){                                                    \
	std::stringstream _where, _message;                                   \
	_where << __FILE__<<':'<<__LINE__;                                    \
	_message << std::string(s) + "\n" <<__FILE__ <<':'<<__LINE__;         \
	std::cerr << _message.str() <<"\nAboring..\n";                        \
    cudaDeviceReset();                                                   \
    exit(EXIT_FAILURE);                                                   \
}


#define checkCudaErrors(status){                                                  \
	std::stringstream _error;                                                     \
	if(status != 0)                                                               \
	{                                                                             \
		_error<<"Cuda faliure: "<<cudaGetErrorString(status)<<" ERROR"<<std::endl; \
	    FatalError(_error.str());                                                 \
   }                                                                              \
}                                                                                 \



#define checkCUDNN(status){                                                 \
    std:: stringstream _error;                                              \
    if(status != CUDNN_STATUS_SUCCESS)                                      \
    {                                                                       \
    	_error << "CUDNN failure ERROR: "<<cudnnGetErrorString(status)<<endl;\
    	FatalError(_error.str());                                            \
    }                                                                        \
}                                                                            \


#define checkCublasErrors(status){                                            \
   std::stringstream _error;                                                  \
   if(status != 0)                                                            \
   {                                                                          \
	   _error<<"Cublas failure ERROR Code: "<<status;\
	   FatalError(_error.str());                                                \
   }                                                                          \
}



template<typename T>
void CHECK_EQ(T a, T b)
{
	std::string s ="NOT_EQ";
	if(a != b)
	{
		FatalError(s);
	}

}


#endif /* CHECKERROR_H_ */
