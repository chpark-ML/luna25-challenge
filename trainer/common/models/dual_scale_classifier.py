import torch
import torch.nn as nn

class DualScaleClassifier(nn.Module):
    def __init__(self, patch_model, image_model, feature_dim=128, hidden_dim=256):
        super().__init__()
        self.patch_model = patch_model
        self.image_model = image_model
        
        # Freeze patch model parameters
        for param in self.patch_model.parameters():
            param.requires_grad = False
            
        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, patch_input, image_input):
        # Get features from frozen patch model
        with torch.no_grad():
            patch_features = self.patch_model(patch_input)
            
        # Get features from trainable image model
        image_features = self.image_model(image_input)
        
        # Concatenate features
        combined_features = torch.cat([patch_features, image_features], dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        
        return output 