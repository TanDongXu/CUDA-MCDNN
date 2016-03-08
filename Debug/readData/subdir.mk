################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../readData/readMnistData.cpp \
../readData/readNetWork.cpp 

OBJS += \
./readData/readMnistData.o \
./readData/readNetWork.o 

CPP_DEPS += \
./readData/readMnistData.d \
./readData/readNetWork.d 


# Each subdirectory must supply rules for building sources it contributes
readData/%.o: ../readData/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "readData" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


