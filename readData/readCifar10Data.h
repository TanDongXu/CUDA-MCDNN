/*
 * readCifar10Data.h
 *
 *  Created on: Mar 16, 2016
 *      Author: tdx
 */

#ifndef READCIFAR10DATA_H_
#define READCIFAR10DATA_H_


#include"../common/cuMatrixVector.h"
#include"../common/cuMatrix.h"
#include<string>
#include<sstream>
#include<fstream>

void read_Cifar10_Data(cuMatrixVector<float>& trainX,
		               cuMatrixVector<float>& testX,
		               cuMatrix<int>*& trainY,
		               cuMatrix<int>*& testY);

#endif /* READCIFAR10DATA_H_ */
