import torch
import sys

print("Testing CUDA availability:", flush=True)
print(f"Is CUDA available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("CUDA is NOT available. Check your environment/drivers.", flush=True)

sys.stdout.flush()