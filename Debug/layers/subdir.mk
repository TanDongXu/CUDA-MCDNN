################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../layers/layersBase.cpp 

CU_SRCS += \
../layers/InceptionLayer.cu \
../layers/LRNLayer.cu \
../layers/ShareLayer.cu \
../layers/activationLayer.cu \
../layers/convLayer.cu \
../layers/dataLayer.cu \
../layers/hiddenLayer.cu \
../layers/poolLayer.cu \
../layers/softMaxLayer.cu 

CU_DEPS += \
./layers/InceptionLayer.d \
./layers/LRNLayer.d \
./layers/ShareLayer.d \
./layers/activationLayer.d \
./layers/convLayer.d \
./layers/dataLayer.d \
./layers/hiddenLayer.d \
./layers/poolLayer.d \
./layers/softMaxLayer.d 

OBJS += \
./layers/InceptionLayer.o \
./layers/LRNLayer.o \
./layers/ShareLayer.o \
./layers/activationLayer.o \
./layers/convLayer.o \
./layers/dataLayer.o \
./layers/hiddenLayer.o \
./layers/layersBase.o \
./layers/poolLayer.o \
./layers/softMaxLayer.o 

CPP_DEPS += \
./layers/layersBase.d 


# Each subdirectory must supply rules for building sources it contributes
layers/%.o: ../layers/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "layers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

layers/%.o: ../layers/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "layers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


