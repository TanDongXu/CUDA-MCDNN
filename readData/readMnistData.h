/*
 * readMnistData.h
 *
 *  Created on: Nov 19, 2015
 *      Author: tdx
 */

#ifndef READMNISTDATA_H_
#define READMNISTDATA_H_

#include"../common/cuMatrixVector.h"
#include"../common/cuMatrix.h"
#include<string>

void readMnistData(cuMatrixVector<float>& normalizedData, cuMatrix<int>*& dataY, string Xpath, string Ypath, int normalized_width, int imageSize);


#endif /* READMNISTDATA_H_ */
