/*
 * mnist.h
 *
 *  Created on: Mar 16, 2016
 *      Author: tdx
 */

#ifndef MNIST_H_
#define MNIST_H_

#include<cudnn.h>
#include"common/cuMatrixVector.h"
#include"common/cuMatrix.h"
#include"readData/readMnistData.h"
#include"config/config.h"
#include"columnNet.cuh"
#include"common/utility.cuh"


void runMnist();



#endif /* MNIST_H_ */
