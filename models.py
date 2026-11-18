import torch
from torch import nn

#CNN Model for Handwritten Digit Recognition
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



#Vision Transformer for Handwritten Digit Recognition
class ViT(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_embedding = nn.Linear(4, 32)

        #Positional embeddings for 16 patches
        self.position_embedding = nn.Parameter(
            torch.randn(1, 16, 32) * 0.02
        )

        #Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )
        #Transformer Encoder with 2 layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        #Ten output classes (digits 0-9)
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):

        #Input shape: [N, 1, 8, 8]

        #Dividing images into 2x2 patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)

        #Rearranging into [N, 16, 4]
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(x.size(0), 16, 4)

        #Projecting patches into embeddings
        x = self.patch_embedding(patches)
        x = x + self.position_embedding

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x

