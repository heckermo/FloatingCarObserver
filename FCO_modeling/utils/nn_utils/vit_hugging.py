import torch
from torch import nn
#from transformers import ViTModel, ViTConfig
from torchvision import transforms

class ViTHugging(nn.Module):
    def __init__(self, vector_dim=2, num_classes=1, sigmoid_activation=True):
        super(ViTHugging, self).__init__()

        # Load pre-trained ViT model without the classification head
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        # Freeze the ViT model if you don't want to fine-tune it
        # for param in self.vit.parameters():
        #     param.requires_grad = False

        vit_output_dim = self.vit.config.hidden_size  # Typically 768 for ViT-base

        # Define your custom MLP layers
        self.mlp = nn.Sequential(
            nn.LayerNorm(vit_output_dim),
            nn.Linear(vit_output_dim, 64),
            nn.ReLU()
        )

        # Define the classification head
        combined_dim = 64 + vector_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        if sigmoid_activation:
            self.classifier.add_module('sigmoid', nn.Sigmoid())

        
        # self.preprocess = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     #transforms.Normalize(mean=[0.5], std=[0.5]),
        #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the channel to create a 3-channel image
        #     # If repeating channels, adjust normalization accordingly
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])
    
    def _preprocess_tensor(self, img_tensor):
        # Step 1: Resize to (224, 224) - Note: torchvision.transforms works with PIL images, so we use torch for resizing
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        # Step 2: Normalize
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        img_tensor = normalize(img_tensor)

        # Step 3: Repeat to make 3 channels, if necessary
        img_tensor = img_tensor.repeat(1, 3, 1, 1)  # Now has shape (batch_size, 3, 224, 224)

        return img_tensor

    def forward(self, img, vector):
        # Preprocess the input image
        img = self._preprocess_tensor(img)

        # Pass the image through the ViT model
        vit_outputs = self.vit(pixel_values=img)
        x = vit_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Pool the outputs (use CLS token or mean pooling)
        x = x[:, 0]  # Assuming CLS token pooling

        # Pass through the MLP layers
        x = self.mlp(x)

        # Concatenate with the additional vector
        x = torch.cat((x, vector), dim=1)

        # Pass through the classification head
        logits = self.classifier(x)

        return logits

if __name__ == "__main__":
    # Create an instance of the model
    model = ViTHugging()

    # Load an example image and vector
    img = torch.randn(3, 1, 224, 224)  # Batch size 1, 3-channel image
    vector = torch.randn(3, 2)  # Batch size 1, 2-dimensional vector

    # Forward pass
    output = model(img, vector)
    print(output)