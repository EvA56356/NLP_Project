import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy


class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes,
                 dropout_rate=0.5, lstm_dropout_rate=0.2, pretrained_embedding=None,
                 attention=None, loss_type='ce', weight=None):
        super(TextBiLSTM, self).__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.bilstm_layer = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                    batch_first=True, bidirectional=True)
        self.layer_norm = LayerNorm(hidden_size * 2)
        self.lstm_dropout = nn.Dropout(lstm_dropout_rate)
        self.classifier = nn.Linear(hidden_size*2, num_classes)
        if loss_type == 'ce':
            if weight is not None:
                self.criterion = CrossEntropyLoss(weight=weight)
            else:
                self.criterion = CrossEntropyLoss()
        elif loss_type == "focal":
            self.criterion = FocalLoss()

    def forward(self, input_ids, input_mask, target=None):
        embs = self.embedding(input_ids)
        embs = self.emb_dropout(embs)
        batch_max_sentence_len = embs.shape[1]
        length = torch.sum(input_mask, dim=-1)
        emb_packed = pack_padded_sequence(embs, length.cpu(),
                                          batch_first=True, enforce_sorted=False)
        lstm_out_packed, _ = self.bilstm_layer(emb_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True,
                                          total_length=batch_max_sentence_len)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = lstm_out * input_mask.float().unsqueeze(2)
        if self.attention:
            M = self.tanh1(lstm_out)  # [128, 32, 256]
            alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
            out = lstm_out * alpha  # [128, 32, 256]
            out = torch.sum(out, 1)  # [128, 256]
            out = F.relu(out)
            lstm_out_res = self.fc1(out)
        else:
            lstm_out_res = torch.mean(lstm_out, dim=1).squeeze()
            lstm_out_res = self.lstm_dropout(lstm_out_res)
        logits = self.classifier(lstm_out_res)
        loss = self.criterion(logits, target)
        return loss, logits
