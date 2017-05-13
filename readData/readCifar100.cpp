/*************************************************************************
	> File Name: readCifar100.cpp
	> Author: TDX 
	> Mail: sa614149@mail.ustc.edu.cn
	> Created Time: 2017年05月11日 星期四 15时15分54秒
 ************************************************************************/
#include"readCifar100.h"
#include<iostream>
#include<string>
#include<glog/logging.h>

#include"common/cuMatrixVector.h"
#include"common/cuMatrix.h"
#include<fstream>
#include<sstream>

using namespace std;

void read_batch(string filename, cuMatrixVector<float>& imageData, cuMatrix<int>*& imageLabel, int number_of_image)
{
    ifstream file(filename, ios::binary);

    if(file.is_open())
    {
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_image; i++)
        {
            unsigned char type1 = 0;
            unsigned char type2 = 0;
            file.read((char*)&type1, sizeof(type1));
            file.read((char*)&type2, sizeof(type2));

            cuMatrix<float>* channels = new cuMatrix<float>(n_rows, n_cols, 3);
            int index = imageData.size();
            imageLabel->setValue(index, 0, 0, type2);

            for(int ch = 0; ch < 3; ch++)
            {
                for(int r = 0; r < n_rows; r++)
                {
                    for(int c = 0; c < n_cols; c++)
                    {
                        unsigned char temp = 0;
                        file.read((char*)&temp, sizeof(temp));
                        channels->setValue(r, c, ch, (float)temp / 256.0f * 2.0f - 1.0f);
                    }
                }
            }

            imageData.push_back(channels);
        }
    }else
    {
        LOG(FATAL) << "Read CIFAR100 Data: Can Not Find The Data:" << filename;
    }
}

void readCifar100Data(cuMatrixVector<float>&trainX, cuMatrixVector<float>&testX, cuMatrix<int>*&trainY, cuMatrix<int>*&testY)
{
    // Read the train data and label
    string file_dir = "data/cifar100/cifar-100-binary/train.bin";
    trainY = new  cuMatrix<int>(50000, 1, 1);

    read_batch(file_dir, trainX, trainY, 50000);

    // Read the test Data and label
    file_dir = "data/cifar100/cifar-100-binary/test.bin";
    testY = new cuMatrix<int>(10000, 1, 1);
    read_batch(file_dir, testX, testY, 10000);

}
