import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_dimension = hidden_size
        #initialize hidden state.
        self.hidden = (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size))
        #Define Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #Define LSTM Cell 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        #Define Linear layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        
    
    def forward(self, features, captions):
        caption_embedding = self.embedding(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), caption_embedding),1)
        #pass to LSTM cell
        lstm_out, self.hidden = self.lstm(embeddings)
        #pass LSTM out to linear for Softmax
        outputs = self.linear(lstm_out)
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        captions = []
        hidden = (torch.randn(1, 1, self.hidden_dimension).to(inputs.device),
                  torch.randn(1, 1, self.hidden_dimension).to(inputs.device))
        for i in range(max_len):
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(outputs)
            outputs = outputs.squeeze(1)
            target_index = outputs.argmax(dim=1)
            captions.append(target_index.item())
            inputs = self.embedding(target_index.unsqueeze(0))
        return captions
            
            
            
        