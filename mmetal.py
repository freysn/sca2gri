"""
requires PyObjC and pyobjc-framework-Metal packages.

https://developer.apple.com/documentation/metal/
https://pyobjc.readthedocs.io/en/latest/api/module-objc.html

"""


import Metal
import ctypes

def load(kernel_source, func_name):
    # create a Metal device, library, and kernel function
    device = Metal.MTLCreateSystemDefaultDevice()
    library = device.newLibraryWithSource_options_error_(kernel_source, None, None)[0]
    kernel_function = library.newFunctionWithName_(func_name)
    
    return {'device' : device, 'kernel_function' : kernel_function}

def create_buffer(instance, array_length, ctype):

    device=instance['device']
    
    # create input and output buffers

    sizeof_elem=ctypes.sizeof(ctype)

    buf_len = array_length * sizeof_elem  # 4 bytes per float
    buf=device.newBufferWithLength_options_(buf_len, Metal.MTLResourceStorageModeShared)
    return {'buf': buf, 'n' : array_length, 'size':buf_len, 'ctype':ctype}

def upload(buf,input_list):
    # map the Metal buffer to a Python array
    input_array = (buf['ctype'] * buf['n']).from_buffer(buf['buf'].contents().as_buffer(buf['size']))  

    input_array[:len(input_list)] = input_list

def download(buf):
    # map the Metal buffer to a Python array
    output_data = (buf['ctype'] *buf['n']).from_buffer(buf['buf'].contents().as_buffer(buf['size']))

    return output_data

def run(instance, n, buf_list):

    device=instance['device']
    kernel_function=instance['kernel_function']

    # create a command queue and command buffer
    commandQueue = device.newCommandQueue()
    commandBuffer = commandQueue.commandBuffer()

    # set the kernel function and buffers
    pso = device.newComputePipelineStateWithFunction_error_(kernel_function, None)[0]
    computeEncoder = commandBuffer.computeCommandEncoder()
    computeEncoder.setComputePipelineState_(pso)
    
    for i, buf in enumerate(buf_list):
        computeEncoder.setBuffer_offset_atIndex_(buf['buf'], 0, i)


    # define threadgroup size
    threadsPerThreadgroup = Metal.MTLSizeMake(n, 1, 1)
    threadgroupSize = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)

    # dispatch the kernel
    computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerThreadgroup, threadgroupSize)
    computeEncoder.endEncoding()

    # commit the command buffer
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()


if __name__ == '__main__':

    import random
    from math import log
    import time

    # define a Metal kernel function
    kernel_source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void log_kernel(device float *in  [[ buffer(0) ]],
                           device float *out [[ buffer(1) ]],
                           uint id [[ thread_position_in_grid ]]) {
        out[id] = log(in[id]);
    }
    """

    instance=load(kernel_source, 'log_kernel')

    max_array_length=1024*1024*16

    input_buffer=create_buffer(instance, max_array_length, ctypes.c_float)
    output_buffer=create_buffer(instance, max_array_length, ctypes.c_float)

    

    for array_length in [1024, 1024*8, 1024*64, 1024*256, 23453, 1024*1024, 1024*1024*16]:
        # populate input buffer with random values
        input_list = [random.uniform(0.0, 1.0) for _ in range(array_length)]


        upload(input_buffer, input_list)
                
        gpu_start = time.time()
        run(instance, array_length, [input_buffer, output_buffer])
        gpu_end = time.time()

        print(f'runtime {gpu_end-gpu_start} for {array_length} elems')

        output_list=download(output_buffer)

        # check that the outputs are correct
        cpu_start=time.time()
        output_python = [log(x) for x in input_list]
        cpu_end = time.time()

        print(f'cpu runtime {cpu_end-cpu_start} for {array_length} elems')
        
        assert all([abs(a - b) < 1e-5 for a, b in zip(output_list, output_python)]), "output does not match reference!"
        print("reference matches output!")
