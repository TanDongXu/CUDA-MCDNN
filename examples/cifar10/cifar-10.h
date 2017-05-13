/*
 * cifar-10.h
 *
 *  Created on: Mar 16, 2016
 *      Author: tdx
 */

#ifndef CIFAR_10_H_
#define CIFAR_10_H_

#include"config/config.h"
#include"readData/readCifar10Data.h"
#include"common/cuMatrixVector.h"
#include"common/cuMatrix.h"
#include"columnNet.cuh"
#include"common/utility.cuh"

#include<cudnn.h>


void runCifar10();



#endif /* CIFAR_10_H_ */
