#include"dataProcess.h"

void dataProcessing(cuMatrixVector<float>& trainSetX, cuMatrixVector<float>& testSetX)
{
    int n_rows = trainSetX[0]->rows;
    int n_cols = trainSetX[0]->cols;
    int n_channels = trainSetX[0]->channels;
    int trainSize = (int)trainSetX.size();

    cuMatrix<float>* avg = new cuMatrix<float>(n_rows, n_cols, n_channels);

    for(int id = 0; id < trainSize; id++)
    {
        int len = trainSetX[0]->getLength();
        for(int j = 0; j < len; j++)
        {
            avg->getHost()[j] += trainSetX[id]->getHostData()[j];
        }
    }


    for(int i = 0; i < avg->getLength(); i++)
    {
        avg->getHostData()[i] /= trainSize;
    }

    for(int id  = 0; id < trainSize; id++)
    {
        int len = trainSetX[0]->getLength();
        for(int j = 0; j < len; j++)
        {
            trainSetX[id]->getHostData()[j] -= avg->getHostData()[j];
        }
    }


    MemoryMonitor::instanceObject()->cpuMemoryMemset(avg->getHostData(), avg->getLength() * sizeof(float));

    int testSize = (int)testSetX.size();

    for (int id = 0; id < testSize; id++)
    {
        int len = testSetX[0]->getLength();
        for (int j = 0; j < len; j++) {
            avg->getHostData()[j] += testSetX[id]->getHostData()[j];
        }
    }

    for (int i = 0; i < avg->getLength(); i++) {
        avg->getHostData()[i] /= testSize;
    }

    for (int id = 0; id < testSize; id++)
    {
        int len = testSetX[0]->getLength();
        for (int j = 0; j < len; j++) {
            testSetX[id]->getHostData()[j] -= avg->getHostData()[j];
        }
    }

    delete avg;
    }
