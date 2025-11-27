from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction, PredictionResult
import numpy as np
import torch
import cv2

class CustomUnetModel(DetectionModel):
    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    def load_model(self):
        if self.device:
            self.model.to(self.device)

    def perform_inference(self, image: np.ndarray):
        # Preprocess
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        
        if self.device:
            image_tensor = image_tensor.to(self.device)
            
        # Inference
        with torch.no_grad():
            prediction = self.model(image_tensor)
            # prediction: (1, C, H, W)
            mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
            
        # Convert to ObjectPrediction
        object_prediction_list = []
        
        # Assuming classes 1, 2, 3 (0 is background)
        # We can get num_classes from model or prediction shape
        num_classes = prediction.shape[1]
        
        for class_id in range(1, num_classes):
            class_mask = (mask == class_id).astype(np.uint8)
            if class_mask.sum() == 0:
                continue
                
            # Find connected components to treat as instances
            num_labels, labels = cv2.connectedComponents(class_mask)
            for label in range(1, num_labels):
                component_mask = (labels == label).astype(bool)
                
                prediction = ObjectPrediction(
                    bool_mask=component_mask,
                    category_id=class_id,
                    category_name=str(class_id),
                    score=1.0,
                    shift_amount=[0, 0], # Sahi handles shift
                    full_shape=None # Sahi handles this
                )
                object_prediction_list.append(prediction)
                
        self._prediction_result = PredictionResult(
            object_prediction_list=object_prediction_list,
            image=image,
        )
