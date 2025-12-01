import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models import BaseSegmentor

@MODELS.register_module()
class MyUNet(BaseSegmentor):
    def __init__(self, num_classes, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Encoder (Downsampling)
        self.encoder1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features[1], features[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features[2], features[3])
        
        # Decoder (Upsampling)
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = self._block(features[2] * 2, features[2])
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = self._block(features[1] * 2, features[1])
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = self._block(features[0] * 2, features[0])
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)
    
    def loss(self, seg_logit, seg_label):
        """Compute loss"""
        loss = dict()
        seg_logit = F.interpolate(seg_logit, size=seg_label.shape[1:], 
                                 mode='bilinear', align_corners=False)
        loss['loss_seg'] = F.cross_entropy(seg_logit, seg_label, ignore_index=255)
        loss['acc_seg'] = self.accuracy(seg_logit, seg_label)
        return loss
    
    def predict(self, inputs, data_samples):
        """Prediction forward"""
        seg_logits = self.forward(inputs)
        return seg_logits