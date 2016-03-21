>CUDA-MCDNN
>==========
>Document   
>1. The first version author is <a href="https://github.com/zhxfl/CUDA-CNN"> zhxfl </a>   
>
>Results
>--------
>CNN accelerated by cuda and lib <a href ="https://developer.nvidia.com/cudnn">CUDNN</a>,The training time is greatly reduced compared to the first version 
>1. Test on <a href="http://yann.lecun.com/exdb/mnist/"> mnist</a>    
>2. Test on cifar-10   
***

>Feature
>--------
>1. Use cudnn lib to develop CNN
>2. Use<a href="http://cs.nyu.edu/~wanli/dropc/"> Dropout</a> and<a href="http://arxiv.org/abs/1312.4400"> NetWork In NetWork(NIN)</a> to train the NetWork
>3. Use <a href="http://arxiv.org/abs/1409.4842">GoogLeNet Inception</a> structure to train NetWork

>Compile
>-------
>Depend on opencv and cuda    
>You can compile the code on windows or linux.   
###SDK include path(-I)   
>* linux: /usr/local/cuda/samples/common/inc/ (For include file "helper_cuda"); /usr/local/include/opencv/ (Depend on situation)        
>* windows: X:/Program Files (x86) /NVIDIA Corporation/CUDA Samples/v6.5/common/inc (For include file "helper_cuda"); X:/Program Files/opencv/vs2010/install/include (Depend on situation)
>
###Library search path(-L)   
>* linux: /usr/local/lib/   
>* windows: X:/Program Files/opencv/vs2010/install/x86/cv10/lib (Depend on situation)    
>
###libraries(-l)      
>* ***cublas***   
>* ***curand***   
>* ***cudadevrt***   
>

###GPU compute 
>* capability 2.0   

###CMake for Linux
>1. mkdir build  
>2. cd build  
>3. cmake ..  
>4. make -j16  
>5. cd ../mnist/  
>6. sh get_mnist.sh  
>7. cd ../cifar-10  
>8. sh get_cifar10.sh  
>9. cd ../  
>10. ./build/CUDA-MCDNN  

###Windows
>1. Install vs2010.
>3. Download and install <a href="https://developer.nvidia.com/cuda-downloads"> cuda-5.0</a> or other higher versions
>4. When you create a new project using VS2010, You can find NVIDIA-CUDA project template, create a cuda-project.
>5. View-> Property Pages-> Configuration Properties-> CUDA C/C++ -> Device-> Code Generation-> compute_20,sm_20   
>6. View-> Property Pages-> Configuration Properties-> CUDA C/C++ -> Common-> Generate Relocatable Device Code-> Yes(-rdc=true) 
>7. View-> Property Pages-> Configuration Properties-> Linker-> Input-> Additional Dependencies-> libraries(-l)   
>8. View-> Property Pages-> Configuration Properties-> VC++ Directories-> General-> Library search path(-L)  
>9. View-> Property Pages-> Configuration Properties-> VC++ Directories-> General-> Include Directories(-I)  

###Linux
>1. Install opencv and cuda
>2. Start the ***nsight*** from cuda
>3. Create an 'empty cuda' project and import the clone code  
>4. Project->Proerties for add-> Build-> Settings->CUDA->Device linker mode: separate compilation   
>5. Project->Proerties for add-> Build-> Settings->CUDA->Generate PTX code 2.0
>6. Project->Proerties for add-> Build-> Settings->CUDA->Generate GPU code 2.0
>7. Project->Proerties for add-> Build-> Settings->Tool Settings->NVCC Compiler->includes: +/usr/local/cuda/samples/common/inc/;  
>8. Project->Proerties for add-> Build-> Settings->Tool Settings->NVCC Linkers->Libraries: libraries(-l)   
>9. Project->Proerties for add-> Build-> Settings->Tool Settings->NVCC Linkers->Libraries search path(-L): /usr/local/lib/    

***
>Config   
>1. <a href="https://github.com/TanDongXu/CUDA-MCDNN/blob/master/profile/MnistConfig.txt">MNIST</a>   
>2. <a href="https://github.com/TanDongXu/CUDA-MCDNN/blob/master/profile/Cifar10Config.txt">CIFAR-10</a>   
***

>Informations
>------------
>* Author :tdx  
>* Mail   :sa614149@mail.ustc.edu.cn  
>* Welcome for any suggest!!   
>* 

