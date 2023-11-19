import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv_input1 = nn.Conv2d(19, 64, 3, padding="same")
        self.bn_input1 = nn.BatchNorm2d(64)
        self.relu_input1 = nn.ReLU()
        self.conv_input2 = nn.Conv2d(64, 128, 3, padding="same")
        self.bn_input2 = nn.BatchNorm2d(128)
        self.relu_input2 = nn.ReLU()

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

        self._init_weights()

    def residual_layer(self):
        return nn.Sequential(
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nan_found = False
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"NaN or Inf found in parameter: {name}")
                nan_found = True

        if nan_found:
            raise ValueError("NaN or Inf found in parameters")

    def forward(self, x):
        x = self.conv_input1(x)
        x = self.bn_input1(x)
        x = self.relu_input1(x)
        x = self.conv_input2(x)
        x = self.bn_input2(x)
        x = self.relu_input2(x)

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
