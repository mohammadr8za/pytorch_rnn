# Recurrent Neural Network (RNN)

In this tutorial, we aim to provide a detailed description of RNNs and provide sample codes for implementation. RNNs are basically introduced to capture dependencies in the sequences. This sequence can be a time serie, tokens (words) in a text, patches of image and so on. Recurrent Neural Networks were firstly introduced in 1986 by Rumelhart et al. and are currently the basis of many applications (we refer to this as "primary RNN" in the text).
intro needed

## Primary RNN in Pytorch

RNNs are simply defined in pytorch using the following command:

*torch.nn.RNN(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)*

let's consider our input data has the length of L. (e.g a sentence that comprises L tokens (words) or any other sequence including L samples) and each sample in the sequence in presented with F features (e.g. each token is embedded to an array of length F). So, our input has the shape of (L, F). Considering batch processing (batch_size=B), we feed inputs of shape (B, L, F) to the network (batch_first=True). Therefore:

**INPUTS**

*input_length:*

length of each input sample (F)

*hidden_size:*

length of hidden state which is the output in each timestamp and fed to the next RNN cell in the same layer (it shapes the size of weight matrices). Any hidden_size may be considered (usually 64, 128, or 256)

*num_layers:*

it defines number of RNN cells that are placed on the top of each other (any number of layers may be considered accroding to the complexity of the input data)

*nonlinearity:*

it defines the activation function that applies on the weighted combination of input and previous hidden state (Defalut=tanh)

*batch_first:*

it depends on the dimension of input sequence. if batch_first=Flase, input data must be of shape (L, B, F), however, we usually prefer to put batch size dimension in the dim=0 (B, L, F).

*bias:*

let us decide to have or remove bias from the equations (usually bias=True)

*bidirectional:*

it lets us decide to have one-directional or bi-directional RNN. (We will go through bidirectional RNN in further sections)
**OUTPUTS**

output, hidden = rnn_layer(input)

output is the stack of all hidden states in the sequence (of size (B, L, F)) hidden is the last hidden state in the sequence of RNN cells

Therefore, last_output == hidden 
