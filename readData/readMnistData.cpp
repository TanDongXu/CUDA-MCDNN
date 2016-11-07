#include"../common/cuMatrix.h"
#include"../common/cuMatrixVector.h"
#include"opencv2/highgui/highgui_c.h"
#include"opencv2/highgui.hpp"
#include"opencv2/imgproc.hpp"
#include"readMnistData.h"
#include<string>
#include<fstream>
#include<iostream>

using namespace std;
using namespace cv;

/*Reverse the int*/
int ReverseInt(int digit)
{
    unsigned char ch1,ch2,ch3,ch4;
    ch1 = digit & 255;
    ch2 = (digit >> 8) & 255;
    ch3 = (digit >> 16) & 255;
    ch4 = (digit >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ((int)ch4);
}

/*read the data from mnist*/
void read_Mnist(string xpath, vector<Mat> &get_image_data)
{
    ifstream file(xpath, ios::binary);
    if(file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);

        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        for(int i = 0; i < number_of_images; i++)
        {
            Mat tpmat=Mat::zeros(n_rows, n_cols,CV_8UC1);
            for(int row = 0; row < n_rows; row ++)
            {
                for(int col = 0; col < n_cols ; col ++)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tpmat.at<uchar>(row,col) = temp;
                }
            }
            get_image_data.push_back(tpmat);
        }

    }else{

        printf("read_Mnist:DataSet open error\n");
        exit(1);
    }
}

/*read the label from mnist*/
void read_Mnist_label(string ypath, cuMatrix<int>* &image_label)
{
    ifstream file(ypath, ios::binary);

    if(file.is_open())
    {
        int magic_number = 0;
        int number_of_label = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        file.read((char*) &number_of_label, sizeof(number_of_label));
        number_of_label = ReverseInt(number_of_label);

        /*alloc label memory*/
        image_label = new cuMatrix<int>(number_of_label, 1, 1);

        for(int i=0; i<number_of_label; i++)
        {
            unsigned char temp = 0;
            file.read((char*) &temp,sizeof(temp));
            image_label->setValue(i, 0, 0, temp);
        }
    }else{

        printf("read_Mnist_label:labelSet open error\n");
        exit(1);
    }
}

/*pading digit*/
Mat pad_image(Mat& input_mat, int end_size)
{
    Mat out_Mat;
    int pading = end_size - input_mat.cols;
    int left_top_pad = round(pading / 2);
    int right_down_pad = pading - left_top_pad;
    if(left_top_pad + right_down_pad > 0)
    {
        copyMakeBorder(input_mat, 
                       out_Mat, 
                       left_top_pad, 
                       right_down_pad, 
                       left_top_pad, 
                       right_down_pad, 
                       BORDER_REPLICATE);
    }else{

        int start_index = - left_top_pad;
        int end_index = input_mat.cols + right_down_pad;
        out_Mat = input_mat(Range(start_index,end_index), Range(start_index,end_index));
    }
    return out_Mat;

}

/*normalized digit*/
Mat normalized_digit(Mat &inputMat, int normalized_width, int end_size)
{
    /*Non zero col numbers*/
    int nZeroCols = 0;

    for(unsigned int i = 0;i < inputMat.cols; i++)
    {
        unsigned int tempSum = 0;
        for(unsigned j = 0; j< inputMat.rows; j++)
        {
            tempSum += inputMat.at<uchar>(j,i);
        }
        /*if the col is nonzero ,then +1*/
        if(tempSum != 0) nZeroCols++;
    }

    Mat temp_outMat = inputMat;

    /*The difference between the current pixel of the normalized digital pixel*/
    int width_diff = normalized_width - nZeroCols;

    if(width_diff)
    {
        int re_size = inputMat.cols + width_diff;
        resize(inputMat, temp_outMat, Size(re_size,re_size));
    }
    return pad_image(temp_outMat,end_size);
}

/*normalized dataset*/
void get_normalizedData(vector<Mat> &train_data, 
                        cuMatrix<int>* &data_label,
                        cuMatrixVector<float> &normalizedData, 
                        int normalized_width, 
                        int end_size)
{
    vector<Mat> tempTrainData;
    if(normalized_width || end_size != 28)
    {
        if(normalized_width)
        cout<<"... normalizing digits to width "<<normalized_width<<" with extra padding "<<end_size-28<<endl;
        else
        cout<<"... (un)padding digits from "<<28<<" ——> "<<end_size<<endl;

        /*use normalized_width to normalize*/

        for(int i = 0; i < train_data.size(); i++)
        {
            /*don't normalized images of digit 1*/
            if(normalized_width && (data_label->getValue(i,0,0) != 1))
            {
                tempTrainData.push_back(normalized_digit(train_data[i],normalized_width,end_size));

            }else
            {
                tempTrainData.push_back(pad_image(train_data[i],end_size));
            }
        }
    }else
    {
        cout<<"... skipping digit normalization and image padding"<<endl;
        tempTrainData = train_data;
    }

    /*read the normalized image to cuMatrixVector and trainsfrom real pixle*/
    for(int i = 0; i < train_data.size(); i++)
    {
        cuMatrix<float>* tmpmat = new cuMatrix<float>(tempTrainData[0].rows, tempTrainData[0].cols, 1);
        for(int r = 0; r < tempTrainData[0].rows; r++)
        {
            for(int c = 0; c < tempTrainData[0].cols; c++)
            {
                float temp = (float)(tempTrainData[i].at< unsigned char>(r,c) * 2.0f / 255.0f - 1.0f);
                tmpmat->setValue(r, c, 0, temp);
            }
        }
        normalizedData.push_back(tmpmat);
    }
}

/*read the data and label*/
void readMnistData(cuMatrixVector<float>& normalizedData, 
                   cuMatrix<int>*& dataY, 
                   string Xpath, 
                   string Ypath, 
                   int normalized_width, 
                   int out_imageSize)
{
    /*read mnist images into vector<Mat>*/
    vector<Mat> trainData;
    read_Mnist(Xpath,trainData);

    /*read mnist label into cuMatrix<int>*/
    read_Mnist_label(Ypath,dataY);

    /*normalized data set*/
    get_normalizedData(trainData, 
                       dataY,
                       normalizedData,
                       normalized_width,
                       out_imageSize);
}
