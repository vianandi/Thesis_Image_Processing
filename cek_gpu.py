import torch

print("PyTorch version:", torch.__version__)
print("CUDA available? ", torch.cuda.is_available())
print("CUDA version (compiled):", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))