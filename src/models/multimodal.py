import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .model import Viscarus

class ViscarusMultiModal(Viscarus):
    """
    Multi-modal version of Viscarus model that can process both images and other modalities.
    
    Features:
    - Supports image, text, and other modality inputs
    - Cross-modal attention mechanism
    - Modality fusion strategies
    - Task-specific optimizations
    """
    def __init__(
        self,
        base_model: str = 'efficientnet_b0',
        num_classes: int = 1000,
        modalities: List[str] = ['image'],
        fusion_type: str = 'concat',
        task_domain: Optional[str] = None,
        pretrained: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(
            base_model=base_model,
            num_classes=num_classes,
            task_domain=task_domain,
            pretrained=pretrained,
            device=device
        )
        
        self.modalities = modalities
        self.fusion_type = fusion_type
        
        # Initialize modality-specific encoders
        self.modality_encoders = nn.ModuleDict()
        for modality in modalities:
            if modality != 'image':  # Image encoder is handled by parent class
                self.modality_encoders[modality] = self._create_modality_encoder(modality)
        
        # Cross-modal attention
        if len(modalities) > 1:
            self.cross_modal_attention = self._create_cross_modal_attention()
            
        # Update classifier for multi-modal input
        if fusion_type == 'concat':
            total_features = self.num_features * len(modalities)
            self.classifier = self._build_classifier(num_classes, total_features)
    
    def _create_modality_encoder(self, modality: str) -> nn.Module:
        """Create encoder for specific modality."""
        if modality == 'text':
            return nn.Sequential(
                nn.Linear(768, self.num_features),  # Assuming BERT-like embeddings
                nn.ReLU(),
                nn.Linear(self.num_features, self.num_features)
            )
        else:
            return nn.Sequential(
                nn.Linear(512, self.num_features),  # Default feature size
                nn.ReLU(),
                nn.Linear(self.num_features, self.num_features)
            )
    
    def _create_cross_modal_attention(self) -> nn.Module:
        """Create cross-modal attention mechanism."""
        return nn.MultiheadAttention(
            embed_dim=self.num_features,
            num_heads=8,
            batch_first=True
        )
    
    def _build_classifier(self, num_classes: int, total_features: int) -> nn.Module:
        """Build classifier for multi-modal input."""
        return nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(total_features // 2, num_classes)
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass processing multiple modalities.
        Args:
            inputs: Dictionary of modality inputs
        Returns:
            Dictionary containing predictions and attention information
        """
        features = {}
        
        # Process each modality
        for modality in self.modalities:
            if modality == 'image':
                features[modality] = super().forward(inputs[modality])['predictions']
            else:
                features[modality] = self.modality_encoders[modality](inputs[modality])
        
        # Apply cross-modal attention if multiple modalities
        if len(self.modalities) > 1:
            feature_list = list(features.values())
            attn_output, _ = self.cross_modal_attention(
                feature_list[0],
                torch.stack(feature_list[1:]),
                torch.stack(feature_list[1:])
            )
            features['cross_modal'] = attn_output
        
        # Fuse features
        if self.fusion_type == 'concat':
            fused_features = torch.cat(list(features.values()), dim=1)
        else:  # Add more fusion strategies as needed
            fused_features = sum(features.values()) / len(features)
        
        # Generate predictions
        predictions = self.classifier(fused_features)
        
        return {
            'predictions': predictions,
            'features': features
        }

class ViscarusMultiModalB2(ViscarusMultiModal):
    """
    Enhanced multi-modal version with B2 features.
    Features:
    - EfficientNet-B2 backbone
    - Advanced feature refinement
    - Cross-layer feature interaction
    - Improved multi-scale fusion
    - Multi-modal support
    - Enhanced cross-modal attention
    Total parameters: ~25M
    """
    def __init__(
        self,
        num_classes: int = 1000,
        modalities: List[str] = ['image'],
        fusion_type: str = 'concat',
        task_domain: Optional[str] = None,
        pretrained: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(
            base_model='efficientnet_b2',
            num_classes=num_classes,
            modalities=modalities,
            fusion_type=fusion_type,
            task_domain=task_domain,
            pretrained=pretrained,
            device=device
        )
        
        # Adaptive feature refinement (5M params)
        self.adaptive_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 2, 1),
                nn.BatchNorm2d(self.num_features * 2),
                nn.ReLU(),
                nn.Conv2d(self.num_features * 2, self.num_features * 2, 1),
                nn.BatchNorm2d(self.num_features * 2),
                nn.ReLU(),
                nn.Conv2d(self.num_features * 2, self.num_features, 1),
                nn.BatchNorm2d(self.num_features),
                nn.ReLU()
            ) for _ in range(3)
        ])
        
        # Cross-layer interaction (3M params)
        self.cross_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.num_features * 2, self.num_features * 2, 1),
                nn.BatchNorm2d(self.num_features * 2),
                nn.ReLU(),
                nn.Conv2d(self.num_features * 2, self.num_features, 1),
                nn.BatchNorm2d(self.num_features),
                nn.ReLU()
            ) for _ in range(2)
        ])
        
        # Improved multi-scale fusion (4M params)
        self.multi_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 2, 1),
                nn.BatchNorm2d(self.num_features * 2),
                nn.ReLU(),
                nn.Conv2d(self.num_features * 2, self.num_features, 1),
                nn.BatchNorm2d(self.num_features),
                nn.ReLU()
            ) for _ in range(3)
        ])
        
        # Enhanced cross-modal attention (2M params)
        self.enhanced_cross_attention = nn.MultiheadAttention(
            embed_dim=self.num_features * 2,
            num_heads=8,
            batch_first=True
        )
        
        # Move new layers to device
        self.adaptive_refinement.to(device)
        self.cross_interaction.to(device)
        self.multi_scale_fusion.to(device)
        self.enhanced_cross_attention.to(device)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with enhanced multi-modal processing.
        Args:
            inputs: Dictionary of modality inputs
        Returns:
            Dictionary containing predictions and attention information
        """
        features = {}
        
        # Process each modality
        for modality in self.modalities:
            if modality == 'image':
                # Get base features
                base_features = super().forward(inputs[modality])['predictions']
                
                # Apply adaptive refinement
                refined_features = []
                for refinement_layer in self.adaptive_refinement:
                    refined_features.append(refinement_layer(base_features))
                refined_features = torch.stack(refined_features).mean(dim=0)
                
                # Apply multi-scale fusion
                fused_features = []
                for fusion_layer in self.multi_scale_fusion:
                    # Create different scale features
                    target_size = max(2, int(refined_features.size(2) * 0.5))
                    scaled_features = F.adaptive_avg_pool2d(refined_features, (target_size, target_size))
                    scaled_features = F.interpolate(scaled_features, 
                        size=refined_features.size()[2:], 
                        mode='bilinear', 
                        align_corners=False)
                    fused_features.append(fusion_layer(scaled_features))
                fused_features = torch.stack(fused_features).mean(dim=0)
                
                features[modality] = fused_features
            else:
                features[modality] = self.modality_encoders[modality](inputs[modality])
        
        # Apply cross-layer interaction
        if len(self.modalities) > 1:
            feature_list = list(features.values())
            for interaction_layer in self.cross_interaction:
                feature_list[0] = interaction_layer(
                    torch.cat([feature_list[0], feature_list[1]], dim=1)
                )
        
        # Apply enhanced cross-modal attention
        if len(self.modalities) > 1:
            feature_list = list(features.values())
            attn_output, _ = self.enhanced_cross_attention(
                feature_list[0],
                torch.stack(feature_list[1:]),
                torch.stack(feature_list[1:])
            )
            features['cross_modal'] = attn_output
        
        # Fuse features
        if self.fusion_type == 'concat':
            fused_features = torch.cat(list(features.values()), dim=1)
        else:
            fused_features = sum(features.values()) / len(features)
        
        # Generate predictions
        predictions = self.classifier(fused_features)
        
        return {
            'predictions': predictions,
            'features': features,
            'attention_maps': self.attention_maps,
            'feature_maps': self.feature_maps
        }