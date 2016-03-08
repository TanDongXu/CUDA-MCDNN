/*
 * test_layer.h
 *
 *  Created on: Dec 26, 2015
 *      Author: tdx
 */

#ifndef TEST_LAYER_H_
#define TEST_LAYER_H_

#include<iostream>


void CHECK_HOST_MATRIX_EQ(float* A, int sizeA, float* B, int sizeB);
void CHECK_DEV_HOST_MATRIX_EQ(float* A, int sizeA, float* B, int sizeB);
void printf_HostParameter(int number, int channels, int height, int width, float* A);
void printf_DevParameter(int number, int channels, int height,int width, float*A);

void copy_DeviceToHost(float*devData, float*&hostData, int number, int channels, int height, int width);
void copy_HostToDevice(float*hostData, float*&devData, int number, int channels, int height, int width);

#endif /* TEST_LAYER_H_ */
