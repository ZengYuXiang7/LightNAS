import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, config):
        super(LSTM, self).__init__()
        self.config = config
        self.d_model = config.d_model

        self.num_classes = 29 if config.dataset == 'nnlqp' else 5
        self.transfer = nn.Linear(self.num_classes, self.d_model)
        self.lstm = nn.LSTM(self.d_model, self.d_model, num_layers=1, batch_first=True)

        if config.dataset == 'nnlqp':
            self.attn = nn.Linear(self.d_model, 1)
            self.fc = nn.Linear(self.d_model, 1)  # attention pooled: [B, d_model]
        else:
            # 假设输入序列长度固定为 T，e.g., 6
            self.seq_len = 6
            self.fc = nn.Sequential(
                nn.Linear(self.d_model * self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, 1)
            )

    def forward(self, features):
        # features: [B, T, input_dim]
        if self.config.dataset != 'nnlqp':
            features = F.one_hot(features.long(), num_classes=self.num_classes).float()
        # -> [B, T, input_dim, num_classes]
        x = self.transfer(features)        # 自动 flatten: [B, T, d_model]
        out, _ = self.lstm(x)              # [B, T, d_model]

        if self.config.dataset == 'nnlqp':
            attn_score = self.attn(out)                 # [B, T, 1]
            attn_weights = F.softmax(attn_score, dim=1) # [B, T, 1]
            pooled = torch.sum(attn_weights * out, dim=1)  # [B, d_model]
        else:
            # MLP拼接融合：out: [B, T, d_model] -> [B, T*d_model]
            pooled = out.reshape(out.size(0), -1)       # [B, T*d_model]

        y = self.fc(pooled)                             # [B, 1]
        return y