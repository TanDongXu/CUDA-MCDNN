/*
 * columnNet.cuh
 *
 *  Created on: Nov 26, 2015
 *      Author: tdx
 */

#ifndef COLUMNNET_CUH_
#define COLUMNNET_CUH_

#include"./common/cuMatrix.h"
#include"./common/cuMatrixVector.h"


void creatColumnNet(int sign);
void cuTrainNetWork(cuMatrixVector<float> &trainData, 
                    cuMatrix<int>* &trainLabel, 
                    cuMatrixVector<float> &testData, 
                    cuMatrix<int>*&testLabel, 
                    int batchSize);


#endif /* COLUMNNET_CUH_ */
