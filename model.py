import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import nltk


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.hidden2word = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        
        embed = self.embed(captions[:,:-1])

        embed = torch.cat((features.unsqueeze(1), embed), 1)
        
        # hidden state is not used
        lstm_out, hidden_state = self.lstm(embed)

        outputs = self.hidden2word(lstm_out)
 
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []

        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            output = self.hidden2word(lstm_out)

            # needless to use softmax because we need an index as return value
            predicted = torch.argmax(output, dim=-1)

            result.append(predicted[0,0].item())
            inputs = self.embed(predicted)

        return result