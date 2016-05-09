#include"readCifar10Data.h"
#include"../common/utility.cuh"

void read_batch(string fileName,
                cuMatrixVector<float>& image_data,
                cuMatrix<int>*& image_label)
{
    ifstream file(fileName, ios::binary);

    if(file.is_open())
    {
        int number_of_images = 10000;
        int n_rows = 32;
        int n_cols = 32;

        for(int i = 0; i < number_of_images; i++)
        {
            unsigned char label;
            file.read((char*)&label, sizeof(label));
            cuMatrix<float>* channels = new cuMatrix<float>(n_rows, n_cols, 3);

            int index = image_data.size();
            image_label->setValue(index, 0, 0, label);

            for(int ch = 0; ch < 3; ch++)
            {
                for(int r = 0; r < n_rows; r++)
                {
                    for(int c = 0; c < n_cols; c++)
                    {
                        unsigned char temp = 0;
                        file.read((char*)&temp, sizeof(temp));
                        channels->setValue(r, c, ch, 2.0f * (float)temp);
                    }
                }
            }

            image_data.push_back(channels);
        }

    }

}

void read_Cifar10_Data(cuMatrixVector<float>& trainX,
                       cuMatrixVector<float>& testX,
                       cuMatrix<int>*& trainY,
                       cuMatrix<int>*& testY)
{
    /*readf the train data and label*/
    string file_dir = "data/cifar-10/data_batch_";
    string suffix = ".bin";

    trainY = new cuMatrix<int>(50000, 1, 1);
    for(int i = 1; i <= 5; i++)
    {
        string fileName = file_dir + int_to_string(i) + suffix;
        read_batch(fileName, trainX, trainY);
    }

    /*read the test data and label*/
    file_dir = "data/cifar-10/test_batch.bin";
    testY = new cuMatrix<int>(10000, 1, 1);
    read_batch(file_dir, testX, testY);
}
