# ref: https://stackoverflow.com/questions/59280745/inference-with-tensorrt-engine-file-on-python
# official ref: https://github.com/NVIDIA/object-detection-tensorrt-example/blob/master/SSD_Model/utils/inference.py
import tensorrt as trt
import numpy as np
import os

import pycuda.driver as cuda
import pycuda.autoinit
import time
import numpy as np



class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TRTEngine:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        # to log/print the warning message into stderr
        # ref: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Logger.html
        self.logger = trt.Logger(trt.Logger.WARNING)
        # ref: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Plugin/IPluginRegistry.html#tensorrt.init_libnvinfer_plugins
# it is basically register pre-build and custom plugin with the following engine automatically after instanstiated. 
        trt.init_libnvinfer_plugins(self.logger, '')
        # ref: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Runtime.html
        self.runtime = trt.Runtime(self.logger)
        
        #initialize the model engine
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        # intermediary variable that will pass/send back our I/O, ref: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#perform_inference_python
        self.context = self.engine.create_execution_context()

                
    # ref: https://github.com/NVIDIA/object-detection-tensorrt-example/blob/master/SSD_Model/utils/engine.py#L95-L99
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        # trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        # ref: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Runtime.html?highlight=deserialize_cuda_engine#tensorrt.Runtime.deserialize_cuda_engine
        # it is literally just read the binary from your .trt model, then load it to the cuda engine/IcudaEngine to be used
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    # ref: https://github.com/NVIDIA/object-detection-tensorrt-example/blob/master/SSD_Model/utils/engine.py#L25-L67
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        # Create a stream in which to copy inputs/outputs and run inference on its content sequentially.
        # we could use more than one stream
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            # https://stackoverflow.com/questions/56472282/an-illegal-memory-access-was-encountered-using-pycuda-and-tensorrt
            # self.dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            self.dtype = trt.float16
            print('binding:', binding, self.dtype, self.engine.get_binding_dtype(binding), self.engine.get_binding_shape(binding), size)
            
            # ref: https://forums.developer.nvidia.com/t/question-about-page-locked-memory/9032
            # this basically register(pinned) memory for GPU/cuda access, 
            # so that whenever a data is stored to the host_memory,
            # then GPU/cuda could copy the stored data quicker
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            # allocate exactly the same number of bytes in GPU for faster CPU-GPU data storage
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # record the location of the allocated memory in int format
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                # record the host data to here for input
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                # record the host data to here for output
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,
                 x:np.ndarray,
                 batch_size:int):
        
        x = x.astype(self.dtype)
        
        # copy the flattened and contiguous array to the pinner host memory
        # ref: https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            # copy host's pinned memory data to the device/GPU
            # ref: https://documen.tician.de/pycuda/driver.html#pycuda.driver.memcpy_htod_async
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]


if __name__ == "__main__":
 
    batch_size = 1
    dtype=np.float16
    trt_engine_path = os.path.join("","yolox_m16.trt")
    model = TRTEngine(trt_engine_path, dtype=dtype)
    shape = model.engine.get_binding_shape(0)

    
    data = (np.random.randint(0,255,(batch_size,*shape[1:]))/255).astype(dtype)
    for i in range(10):
        start = time.time()
        result = model(data,batch_size)#[0]
        # result = result.reshape(1, -1, 85)
        print(i,'speed: {:.3f}s'.format(time.time()-start))