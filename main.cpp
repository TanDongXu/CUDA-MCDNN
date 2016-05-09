/*
* main.cpp
*
*  Created on: Nov 19, 2015
*      Author: tdx
*/
#include<iostream>
#include <cuda.h>
#include"./examples/mnist.h"
#include"./examples/cifar-10.h"

/*use cuDNN Acceleration*/

int main(void)
{
    cout<<"Select the dataSet to run:"<<endl<<"1.MNIST    2.CIFAR-10"<<endl;

    cudaSetDevice(0);
    int cmd;
    cin>> cmd;

    if(1 == cmd)
    runMnist();
    else if(2 == cmd)
    runCifar10();
    else{
        cout<<"DataSet select Error!"<<endl;
        exit(0);
    }
    return 0;
}



