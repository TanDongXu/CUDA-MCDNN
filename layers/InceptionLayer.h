/*
 * InceptionLayer.h
 *
 *  Created on: Mar 5, 2016
 *      Author: tdx
 */

#ifndef INCEPTIONLAYER_H_
#define INCEPTIONLAYER_H_

#include<tuple>
#include"layersBase.h"
#include"convLayer.h"
#include"poolLayer.h"
#include"../composite/Inception.h"

class InceptionLayer : public convLayerBase
{
public:
	InceptionLayer(string name, int sign);
	void forwardPropagation(string train_or_test);
	void backwardPropagation(float Momentum);
	void saveWeight(FILE* file);
	void readWeight(FILE* file);
	void Forward_cudaFree(){}
	void Backward_cudaFree(){}


	~InceptionLayer(){};

	int getOutputSize()
	{
		return outputSize;
	}


private:
	int one;
	int three;
	int five;
	int three_reduce;
	int five_reduce;
	int pool_proj;
	int outputSize;
	float lambda;
	float epsilon;

	Inception * inception;

};

#endif /* INCEPTIONLAYER_H_ */
