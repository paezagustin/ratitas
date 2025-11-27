
import torch
import segmentation_models_pytorch as smp
from monai.inferers import SlidingWindowInferer

# Mock the model
model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=4)
model.eval()

# Create a dummy large image (1, 3, 1024, 1024)
input_tensor = torch.randn(1, 3, 1024, 1024)

# Define MONAI inferer
# roi_size is the patch size (e.g., 512x512)
inferer = SlidingWindowInferer(roi_size=(512, 512), sw_batch_size=4, overlap=0.25)

# Perform inference
with torch.no_grad():
    output = inferer(input_tensor, model)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
