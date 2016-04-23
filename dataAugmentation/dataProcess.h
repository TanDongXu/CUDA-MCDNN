/*
 * dataProcess.h
 *
 *  Created on: Apr 23, 2016
 *      Author: tdx
 */

#ifndef DATAPROCESS_H_
#define DATAPROCESS_H_


#include"../common/cuMatrixVector.h"
#include"../common/cuMatrix.h"


void dataProcessing(cuMatrixVector<float>& tranSetX, cuMatrixVector<float>& testSetX);


#endif /* DATAPROCESS_H_ */
