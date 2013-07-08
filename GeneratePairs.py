###############################################################################
#
#   William Sell
#
#   This class will generate the Pairs for the Random Walk. 
#
#
###############################################################################
#------------------------------------------------------------------------------
#   IMPORTS
#------------------------------------------------------------------------------
import numpy as np
import time
import math

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
from pycuda.curandom import rand as curand
# Initialize the CUDA device
import pycuda.autoinit
import EPD_Kernel_Agg as EK


# This CUDA kernel will take our log random vector 1 and create vector 2 with given inputs 
kernel_source = \
"""

__global__ void kernel(float* rand1, float* rand2, float mean1, float stddev1, float mean2, float stddev2, int size, float m1, float m2)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  
  float stddevRatio = stddev2/stddev1;
   
  if (tid  < size){   
   
  rand2[tid] = mean2 + ( ( rand1[tid] - mean1 ) * stddevRatio );

  rand1[tid] = ( rand1[tid]*m1 );  
  rand2[tid] = ( rand2[tid]*m2 );
  

  }
  
}


"""

# This CUDA kernel will take our log random vector 1 and create vector 2 with given inputs 
single_kernel_source = \
"""

__global__ void single_kernel(float* rand1, float* rand2, float mean1, float stddev1, float mean2, float stddev2, int size, float m1, float m2)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  
  float stddevRatio = stddev2/stddev1;
   
  if (tid  < size){   
   
  rand2[tid] = mean2 + ( ( ( rand1[tid] / m1) - mean1 ) * stddevRatio );

 
  rand2[tid] = ( rand2[tid]*m2 );
  

  }
  
}


"""


# Kernel will correct correlation to desired level
corr_kernel_source = \
"""

__global__ void corr_kernel(float* rand1, float* rand2, int switch1, int switch2, int size, float mean1, float mean2)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  

  if (tid  == 0){   

  float firstValue = rand2[switch1];
  
  rand2[switch1] = rand2[switch2];
  rand2[switch2] = firstValue;

  }
  
}


"""


def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)
  
  
  
def generateCorrSingle(rand1_d, mean2, stddev2, corr):

    
  # Compile the CUDA kernel
  single_kernel = cuda_compile(single_kernel_source,"single_kernel") 
  corr_kernel = cuda_compile(corr_kernel_source,"corr_kernel") 

  # On the host, define the kernel parameters
  blocksize = (512,1,1)     # The number of threads per block (x,y,z)
  gridsize  = (512,1)   # The number of thread blocks     (x,y)

  randSize = rand1_d.size
  
  randShape = (randSize,1)
  
  #send device pointer & array size for stats
  mean1, stddev1, minVal, maxVal = EK.getStats(rand1_d)
  
  # means
  m1 = 1
  m2 = 1
  # stddev's
  s1 = stddev1/mean1
  s2 = stddev2/mean2

  rand2_d = gpu.zeros(randShape, dtype=np.float32)

  single_kernel(rand1_d, rand2_d, np.float32(m1), np.float32(s1), np.float32(m2), np.float32(s2), np.int32(randSize), np.float32(mean1), np.float32(mean2), block=blocksize, grid=gridsize)
  
  
  correlation = 1.0
 
  while correlation > corr:
    
    #generate our random switch positions
    switch1 = np.random.randint(0,randSize)
    switch2 = np.random.randint(0,randSize)
    
    #run the correlation kernel
    corr_kernel(rand1_d, rand2_d, np.int32(switch1), np.int32(switch2), np.int32(randSize), block=(1024,1,1), grid=(1,1))
    
    d_arr1 = np.float32(rand1_d.get())
    d_arr2 = np.float32(rand2_d.get())
    d_arr1 = np.reshape(d_arr1, (-1))
    d_arr2 = np.reshape(d_arr2, (-1)) 
    correlation = np.corrcoef(d_arr1, d_arr2)[1][0]


  
  return rand2_d  



def generateCorrPairs(mean1, stddev1, mean2, stddev2, corr, size=1000):
    
  # Compile the CUDA kernel
  kernel = cuda_compile(kernel_source,"kernel") 
  corr_kernel = cuda_compile(corr_kernel_source,"corr_kernel") 

  # On the host, define the kernel parameters
  blocksize = (512,1,1)     # The number of threads per block (x,y,z)
  gridsize  = (512,1)   # The number of thread blocks     (x,y)

  randSize = size
  randShape = (randSize,1)

  start = time.time()
  
  #desired values for data
  # means
  m1 = 1
  m2 = 1
  # stddev's
  s1 = stddev1/mean1
  s2 = stddev2/mean2
  
  # variances
  v1 = s1**2
  v2 = s2**2

  #lognormal mean & stddev calculation
  logMean1 = math.log(  (m1**2) / (math.sqrt( v1 + (m1**2) )) )
  logStddev1 = math.sqrt ( math.log( (v1/(m1**2))+1 )   )
  
  generator = pycuda.curandom.XORWOWRandomNumberGenerator()
    
  #generate the lognormal array on the gpu  
  rand1_d = generator.gen_log_normal(randShape, dtype= np.float32, mean=logMean1, stddev=logStddev1    )
  #and the variable 2 random walk  
  rand2_d = gpu.zeros(randShape, dtype=np.float32)
  

  kernel(rand1_d, rand2_d, np.float32(m1), np.float32(s1), np.float32(m2), np.float32(s2), np.int32(randSize), np.float32(mean1), np.float32(mean2), block=blocksize, grid=gridsize)
  
  i = 0 
  correlation = 1.0
 
  while correlation > corr:
    
    #generate our random switch positions
    switch1 = np.random.randint(0,randSize)
    switch2 = np.random.randint(0,randSize)
    
    #run the correlation kernel
    corr_kernel(rand1_d, rand2_d, np.int32(switch1), np.int32(switch2), np.int32(randSize), block=(1024,1,1), grid=(1,1))
    
    d_arr1 = np.float32(rand1_d.get())
    d_arr2 = np.float32(rand2_d.get())
    d_arr1 = np.reshape(d_arr1, (-1))
    d_arr2 = np.reshape(d_arr2, (-1)) 
    correlation = np.corrcoef(d_arr1, d_arr2)[1][0]
    
    i = i + 1

  end = time.time()
  #print "time: ", end-start  
  

  return rand1_d, rand2_d
