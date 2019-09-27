import chainer
import cupy
import chainer.functions as F
import chainer.links as L
import torch
import torch.nn.functional as TF

from pynvrtc.compiler import Program
from collections import namedtuple

Stream = namedtuple('Stream', ['ptr'])
s = Stream(ptr=torch.cuda.current_stream().cuda_stream)

from cupy.cuda import function

CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

kernel = '''
extern "C"
__global__ void copy(float *dst, const float *src, int total)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i >= total)
      return;
   dst[i] = src[i];
}
'''

program = Program(kernel, 'copy.cu')
ptx = program.compile()

m = function.Module()
m.load(bytes(ptx.encode()))

copy_func = m.get_function('copy')


def datacopy(src_ptr, dst_ptr, size):
    copy_func(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(size), 1, 1),
              args=[src_ptr, dst_ptr, size],
              stream=cupy.cuda.get_current_stream())


class PyTorchFunc(chainer.function_node.FunctionNode):
    def __init__(self, torch_func):
        self.torch_func = torch_func
    
    def forward_gpu(self, x):
        for i in range(len(x)):
            # Assert all inputs are float32
            assert x[i].dtype == cupy.float32
            
        self.gpu = chainer.cuda.get_device_from_array(x[0]).id
        
        torch_x = [torch.zeros(*x[i].shape, device=torch.device("cuda:"+str(self.gpu)), requires_grad=True, dtype=torch.float32).contiguous() for i in range(len(x))]
        
        for i in range(len(x)):
            datacopy(torch_x[i].data.data_ptr(), x[i], x[i].size)
        
        self.torch_x = torch_x
        torch_y = self.torch_func(*torch_x)
        self.torch_y = torch_y
        
        output = [cupy.ascontiguousarray(cupy.zeros(tuple(torch_y[i].shape), dtype=cupy.float32)) for i in range(len(torch_y))]
        for i in range(len(torch_y)):
            datacopy(output[i], torch_y[i].data.data_ptr(), torch_y[i].numel())
            
        return tuple(output)
    
    def backward(self, indexes, gx):
        torch_y_grad = [torch.zeros(*gx[i].shape, device=torch.device("cuda:"+str(self.gpu)), requires_grad=False, dtype=torch.float32).contiguous() for i in range(len(gx))]
        for i in range(len(gx)):
            if gx[i] is not None:
                datacopy(torch_y_grad[i].data.data_ptr(), cupy.ascontiguousarray(gx[i].data), torch_y_grad[i].numel())
                
        for i in range(len(gx)):
            if gx[i] is not None:
                self.torch_y[i].backward(torch_y_grad[i])
        
        x_grad = [cupy.ascontiguousarray(cupy.zeros(tuple(self.torch_x[indexes[i]].grad.shape), dtype=cupy.float32)) for i in range(len(indexes))]
        for i in range(len(indexes)):
            datacopy(x_grad[i], self.torch_x[indexes[i]].grad.contiguous().data_ptr(), x_grad[i].size)
            
        # Not support Double Backprop for now
        x_grad = [chainer.Variable(x, requires_grad=False) for x in x_grad]
        return tuple(x_grad)
    

def pytorch_func(func, inputs):
    return PyTorchFunc(func).apply(tuple(inputs))