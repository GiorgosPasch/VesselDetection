from torch import nn


# class CNN(nn.Module):
#     def __init__(self, classes):
#         super().__init__()
#         # Convolutional layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#
#         # Fully connected layers for regression
#         self.classification_layers = nn.Sequential(
#             nn.Linear(128 * 8 * 8, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         return self.classification_layers(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.activation_fn = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512*4*4, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.activation_fn(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.activation_fn(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.activation_fn(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.activation_fn(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
