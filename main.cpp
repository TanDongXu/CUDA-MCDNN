/*
* main.cpp
*  Created on: Nov 19, 2015
*      Author: tdx
*/
#include<iostream>
#include <cuda.h>
#include"examples/mnist/mnist.h"
#include"examples/cifar10/cifar-10.h"
#include"examples/cifar100/cifar100.h"
#include"examples/dynamic_g_model/dynamic_g_entry.hpp"
#include<glog/logging.h>

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "Select the DataSet to Run:";
    LOG(INFO) << "1.MNSIT     2.CIFAR-10    3.Dynamic Generation Model     4.CIFAR-100";

    cudaSetDevice(0);
    int cmd;
    cin>> cmd;
    if(1 == cmd)
        runMnist();
    else if(2 == cmd)
         runCifar10();
    else if(3 == cmd)
        dynamic_g_entry();
    else if(4 == cmd)
        runCifar100();
    else
        LOG(FATAL) << "DataSet Select Error.";

    return 0;
}



