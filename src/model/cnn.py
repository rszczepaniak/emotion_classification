import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        conv_output_size = self._get_conv_output_size((3, 48, 48))  # Adjust based on actual input size  (3, 24, 48)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            return x.numel()

    def forward(self, x):
        # Apply first convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Apply second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Apply third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # Dynamically calculate batch size

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
