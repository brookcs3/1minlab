export function Lab4Content() {
  return (
    <>
      <h3>PyTorch Setup Check</h3>
      <p>
        Before building models, it's crucial to verify the PyTorch installation and check if a GPU is available for acceleration. This ensures the environment is ready for deep learning tasks.
      </p>
      <p>
        <em>It's important my exact text is mentioned. These are tutorials and we need to stress 1 min. Please include my entire text in the view write up.</em>
      </p>
      <pre>
        <code>
{`import torch

print(f"PyTorch version: {torch.__version__}")
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

if gpu_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
`}
        </code>
      </pre>
      <h3>NumPy-to-Tensor Conversion</h3>
      <p>
        Deep learning pipelines often start with data in NumPy arrays. PyTorch provides seamless, zero-copy conversion between NumPy arrays and its own tensor format, which is highly efficient.
      </p>
      <pre>
        <code>
{`import numpy as np

# Create a NumPy array
numpy_array = np.random.rand(2, 3)
print("NumPy array:\\n", numpy_array)

# Convert to a PyTorch tensor
torch_tensor = torch.from_numpy(numpy_array)
print("\\nPyTorch tensor:\\n", torch_tensor)
`}
        </code>
      </pre>
      <h3>Pipeline Integration with GPU</h3>
      <p>
        Tensors can be moved to the GPU for massively parallel computation. This is a fundamental step for accelerating training and inference in any deep learning pipeline.
      </p>
      <pre>
        <code>
{`if gpu_available:
    device = torch.device("cuda")
    tensor_on_gpu = torch_tensor.to(device)
    print(f"\\nTensor moved to: {tensor_on_gpu.device}")
else:
    print("\\nGPU not available, tensor remains on CPU.")
`}
        </code>
      </pre>
    </>
  );
}
