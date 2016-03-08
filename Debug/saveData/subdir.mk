################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../saveData/saveNetWork.cpp 

OBJS += \
./saveData/saveNetWork.o 

CPP_DEPS += \
./saveData/saveNetWork.d 


# Each subdirectory must supply rules for building sources it contributes
saveData/%.o: ../saveData/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "saveData" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


