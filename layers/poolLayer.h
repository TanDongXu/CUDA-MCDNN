/*
 * poolLayer.h
 *
 *  Created on: Nov 28, 2015
 *      Author: tdx
 */

#ifndef POOLLAYER_H_
#define POOLLAYER_H_

#include<tuple>
#include<string>
#include<cudnn.h>
#include<math.h>
#include"layersBase.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"

using namespace std;


class poolLayer : public convLayerBase
{
public:
	typedef tuple<string, int, int, int, int, int, int, int> param_tuple;
	poolLayer(string name);
	poolLayer(string name, const param_tuple& agrs);
	void forwardPropagation(string train_or_test);
	void backwardPropagation(float Momentum);
	void saveWeight(FILE*file){}
	void readWeight(FILE*file){}
	void Forward_cudaFree();
	void Backward_cudaFree();

	~poolLayer()
	{
		destroyHandles();
	}

	void createHandles();
	void destroyHandles();

	int getOutputSize()
	{
		return outputSize;
	}


private:
	string  poolType;
	int poolDim;
	int pad_h;
	int pad_w;
	int stride_h;
	int stride_w;
	int nonLinearity;
	int outputSize;


private:
	cudnnTensorDescriptor_t srcTensorDesc = NULL;
	cudnnTensorDescriptor_t dstTensorDesc = NULL;
	cudnnPoolingDescriptor_t poolingDesc = NULL;

	cudnnTensorDescriptor_t srcDiffTensorDesc = NULL;
	cudnnTensorDescriptor_t dstDiffTensorDesc = NULL;


};



#endif /* POOLLAYER_H_ */
