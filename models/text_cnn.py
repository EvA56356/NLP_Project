import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes,
                 num_filters=64, emb_dropout=0.5, 
                 pretrained_embedding=None, weight=None):
        super(TextCNN, self).__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(emb_dropout)
        self.conv1 = nn.Conv2d(1, num_filters, (3, embedding_size))  
        self.conv2 = nn.Conv2d(1, num_filters, (4, embedding_size)) 
        self.dropout = nn.Dropout(emb_dropout)
        self.classifier = nn.Linear(num_filters*2, num_classes)
        self.criterion = FocalLoss()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        return x

    def forward(self, input_ids, input_mask, target=None):
        x = self.embedding(input_ids)
        x = self.embedding_dropout(x)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)
        x2 = self.conv_and_pool(x, self.conv2)
        x = torch.cat([x1, x2], 1)
        logits = self.classifier(x)
        loss = self.criterion(logits, target) if target is not None else 0
        return loss, logits
