import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv_input = nn.Conv2d(35, 128, 5, padding="same")
        self.bn_input = nn.BatchNorm2d(128)
        self.relu_input = nn.ReLU()

        self.res_layers = nn.ModuleList([self.residual_layer() for _ in range(7)])

        self.conv_policy = nn.Conv2d(128, 2, 1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.relu_policy = nn.ReLU()
        self.flatten_policy = nn.Flatten()
        self.fc_policy = nn.Linear(8 * 8 * 2, 1968)

        self.conv_value = nn.Conv2d(128, 4, 1)
        self.bn_value = nn.BatchNorm2d(4)
        self.relu_value = nn.ReLU()
        self.flatten_value = nn.Flatten()
        self.fc_value1 = nn.Linear(8 * 8 * 4, 32)
        self.fc_value2 = nn.Linear(32, 1)

    def residual_layer(self):
        return nn.Sequential(
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = self.relu_input(x)

        for layer in self.res_layers:
            # x = self.relu_input(layer(x) + x)
            x = checkpoint.checkpoint(layer, x) + x

        policy_out = self.conv_policy(x)
        policy_out = self.bn_policy(policy_out)
        policy_out = self.relu_policy(policy_out)
        policy_out = self.flatten_policy(policy_out)
        policy_out = self.fc_policy(policy_out)
        policy_out = F.softmax(policy_out, dim=1)

        value_out = self.conv_value(x)
        value_out = self.bn_value(value_out)
        value_out = self.relu_value(value_out)
        value_out = self.flatten_value(value_out)
        value_out = self.fc_value1(value_out)
        value_out = self.relu_value(value_out)
        value_out = self.fc_value2(value_out)
        value_out = torch.tanh(value_out)

        return policy_out, value_out
