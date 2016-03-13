/*
 * utility.cuh
 *
 *  Created on: Mar 11, 2016
 *      Author: tdx
 */

#ifndef UTILITY_CUH_
#define UTILITY_CUH_

#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>

#include"cuBaseVector.h"
#include"checkError.h"

#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2

/*the ways of initial weight*/
#define RANDOM 1
#define READ_FROM_FILE 2
const double FLAGS_lr_gamma = 0.0001;   //learning rate policy
const double FLAGS_lr_power = 0.75;     //Learing rate policy power

/*展示GPU信息*/
void showDevices();
/*多个数组合并*/
__global__ void MultiChannelsMerge(float** inputs, float* outputs, int* channels, int* indexs, int row, int outChannels);
/*计算三个数组之和*/
__global__ void MultiArrayAdd(float* array1, float* array2, float* array3, float* array4, float* outputs);
#endif /* UTILITY_CUH_ */
