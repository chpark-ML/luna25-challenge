import torch
import torch.nn as nn

class DualScaleClassifier(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=256):
        super().__init__()
        
        # Layer normalization for each feature stream (using LayerNorm instead of BatchNorm)
        self.patch_norm = nn.BatchNorm1d(feature_dim)
        self.image_norm = nn.BatchNorm1d(feature_dim)
        
        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # 64*2 = 128 input features
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, patch_features, image_features):
        # Normalize each feature stream
        patch_features = self.patch_norm(patch_features)  # (B, C)
        image_features = self.image_norm(image_features)  # (B, C)
        
        # Concatenate features along feature dimension
        combined_features = torch.cat([patch_features, image_features], dim=1)  # (B, 128)
        
        # Final classification
        output = self.fusion(combined_features)
        
        return output 
    