/*
 * Concat.h
 *
 *  Created on: Mar 6, 2016
 *      Author: tdx
 */

#ifndef CONCAT_H_
#define CONCAT_H_

#include"../layers/layersBase.h"
#include"../common/MemoryMonitor.h"
#include"../common/checkError.h"
#include"../cuDNN_netWork.h"
#include<tuple>

class Concat
{
public:
	typedef tuple<int, int, int, int>param_tuple;
	Concat(Layers*& Inner_Layers, const param_tuple& args);
	float*& forwardSetup();
	float*& backwardSetup();



private:
	int number;
	int channels;
    int height;
    int width;
    int prev_number;
    int prev_channels;
    int prev_height;
    int prev_width;
    int one;
    int three;
    int five;
    int pool_proj;
    int oneDim;
    int threeDim;
    int fiveDim;
    int pool_projDim;
	float* one_outResult;
	float* three_outResult;
	float* five_outResult;
	float* pool_proj_outResult;
	float* outputResult;
	float* one_diff;
	float* three_diff;
	float* five_diff;
	float* pool_proj_diff;
	float* diffData;
	Layers *InnerLayers;


};



#endif /* CONCAT_H_ */
