import sys
import torch
import onnxruntime as ort
import numpy as np
import time
from glob import glob
from memory_profiler import memory_usage
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def main(model_path):
    
    # Initialize NVML
    nvmlInit()

    # Get the first GPU handle
    handle = nvmlDeviceGetHandleByIndex(0)
    
    # Load the ONNX model
    providers = ['CUDAExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    # Prepare input data for the model
    inputs = session.get_inputs()
    input_data = {inputs[i].name : np.random.random(inputs[i].shape).astype(np.float32) for i in range(len(inputs))}

    # Define a function to run inference
    def run_inference():
        start_time = time.time()
        outputs = session.run(None, input_data)
        end_time = time.time()
        return outputs, end_time - start_time

    # Measure CPU memory usage and runtime
    mems, gpu_mems, runtimes = [],[],[]
    for i in range(20):
    
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = info.used / (1024 ** 2)
    
        mem_usage, (outputs, runtime) = memory_usage(run_inference, retval=True)
        
        mems.append(max(mem_usage))
        runtimes.append(runtime)

        # Measure GPU memory usage
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_used = info.used / (1024 ** 2) - gpu_mem # Convert bytes to MiB
            
        gpu_mems.append(gpu_mem_used)
        
    runtimes = list(sorted(runtimes))[5:15]
    mems = list(sorted(mems))[5:15]
    gpu_mems = list(sorted(gpu_mems))[5:15]

    print(f'Inference Runtime: {np.mean(runtimes):.4f} seconds')
    print(f'Peak CPU Memory Usage: {np.mean(mems):.2f} MiB')
    print(f'GPU Memory Usage: {np.mean(gpu_mems):.2f} MiB')

if __name__ == '__main__':
    path = sys.argv[-1]
    for size in [512, 768, 1024, 1280, 1536, 1792, 2048, 4096]:
        print(size)
        main(path + f'onnx_model_{size}.onnx')
        print('-'*10 + '\n')
