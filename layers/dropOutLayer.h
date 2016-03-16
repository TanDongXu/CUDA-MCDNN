/*
 * dropOutLayer.h
 *
 *  Created on: Mar 15, 2016
 *      Author: tdx
 */

#ifndef DROPOUTLAYER_H_
#define DROPOUTLAYER_H_

#include"../config/config.h"
#include"../tests/test_layer.h"
#include"layersBase.h"
#include<curand.h>


class dropOutLayer : public layersBase
{
public:
	dropOutLayer(string name);
	void CreateUniform(int size);
	void Dropout_TrainSet(float* data, int size, float dropout_rate);
	void Dropout_TestSet(float* data, int size, float dropout_rate);
	void forwardPropagation(string train_or_test);
	void backwardPropagation(float Momemtum);
	void saveWeight(FILE* file){}
	void readWeight(FILE* file){}
	void Forward_cudaFree(){}
	void Backward_cudaFree(){}

	void createHandles();
	void destroyHandles();


	~dropOutLayer()
	{
		destroyHandles();
	}
	 int getOutputSize()
	 {
		return outputSize;
	 }




private:
	 int inputSize;
	 int outputSize;
	 float DropOut_rate;
	 float* outputPtr;


private:
	 curandGenerator_t curandGenerator_DropOut;
};



#endif /* DROPOUTLAYER_H_ */
