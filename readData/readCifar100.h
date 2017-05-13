/*************************************************************************
	> File Name: readCifar100.h
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2017年05月11日 星期四 15时16分06秒
 ************************************************************************/

#ifndef _READCIFAR100_H
#define _READCIFAR100_H

#include"common/cuMatrixVector.h"
#include"common/cuMatrix.h"

void readCifar100Data(cuMatrixVector<float>&trainX, 
                      cuMatrixVector<float>&testX, 
                      cuMatrix<int>*&trainY, 
                      cuMatrix<int>*&testY);
#endif
