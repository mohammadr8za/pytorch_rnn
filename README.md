# Recurrent Neural Network (RNN) 

In this tutorial, we aim to provide a detailed description of RNNs and provide sample codes for implementation. RNNs are basically introduced to capture dependencies in the sequences. This sequence can be a time serie, tokens (words) in a text, patches of image and so on. Recurrent Neural Networks were firstly introduced in 1986 by Rumelhart et al. and are currently the basis of many applications (we refer to this as "primary RNN" in the text).
intro needed

![1_B0q2ZLsUUw31eEImeVf3PQ](https://github.com/mohammadr8za/pytorch_rnn/assets/72736177/bdeb10b6-7ef2-4d10-94c7-fb2f83041c4d)

## Primary RNN in Pytorch (Left Architecture in the Picture Above)

RNNs are simply defined in pytorch using the following command:

*torch.nn.RNN(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)*

let's consider our input data has the length of L. (e.g a sentence that comprises L tokens (words) or any other sequence including L samples) and each sample in the sequence in presented with F features (e.g. each token is embedded to an array of length F). So, our input has the shape of (L, F). Considering batch processing (batch_size=B), we feed inputs of shape (B, L, F) to the network (batch_first=True). Therefore:

**INPUTS**

* *input_length:*

  length of each input sample (F)



* *hidden_size:*

  length of hidden state which is the output in each timestamp and fed to the next RNN cell in the same layer (it shapes the size of weight matrices). Any hidden_size may be considered (usually 64, 128, or 256)


* *num_layers:*

  it defines number of RNN cells that are placed on the top of each other (any number of layers may be considered accroding to the complexity of the input data)



* *nonlinearity:*

  it defines the activation function that applies on the weighted combination of input and previous hidden state (Defalut=tanh)



* *batch_first:*

  it depends on the dimension of input sequence. if batch_first=Flase, input data must be of shape (L, B, F), however, we usually prefer to put batch size dimension in the dim=0 (B, L, F).



* *bias:*

  let us decide to have or remove bias from the equations (usually bias=True)



* *bidirectional:*

  it lets us decide to have one-directional or bi-directional RNN. (We will go through bidirectional RNN in further sections)



**OUTPUTS**

* *output, hidden = rnn_layer(input)*

  output is the stack of all hidden states in the sequence (of size (B, L, F)) hidden is the last hidden state in the sequence of RNN cells



Therefore, last_output == hidden 

for more info: [Pytorch RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
```
import torch
from torch import nn 

# first create a sample input 
batch_size = 8
seq_len = 10
feature_size = 16

tensor = torch.randn(size=(batch_size, seq_len, feature_size))

# Define RNN 
rnn_layer = nn.RNN(input_size=16, hidden_size=64, num_layers=1, batch_first=True)

# feed input to the RNN and initialize the first hidden state
output, hidden = rnn_layer(tensor, torch.randn(size=(1, 8, 64)))

print(fr"output shape: {output.shape} \nhidden shape: {hidden.shape}")

# last output and hidden
output[0, 9, :], hidden[0, 0, :]
# Equal!
```

## Long-Short Term Memory (LSTM) (Middle Architecture in the Picture Above)

![featured](https://github.com/mohammadr8za/pytorch_rnn/assets/72736177/a6bcbd0d-2262-4d1b-baec-3af932af97f4)


LSTM is an effective variant of RNNs that is brilliantly desinged with more parameters to solve the problem of vanishing gradient in the traditional RNNs. LSTM was introduced by Hochreiter and Schmidhuber in 1997. Let's dig into LSTM atrchitecture. 
LSTM comprises three main gates: forget gate, input gate, and output gate. 

* Forget Gate: in this gate the architecture choose which past information to keep and which ones to remove. In the forget gate, f_t is defined as a function of hidden state of previous timestamp and the input. Then by applying a sigmoid function the values in f_t are mapped between 0 and 1. 0 means removing information and 1 mean keeping information from previous timestamps (needless to say any value between in the f_t shows how much info should be passes). By multiplying f_t in the values of previous cell state (which brings informtion from previous timestamps) important imformation comes to the current timestamp (or LSTM cell). 
Formula: 

  f_t = sigmoid(W_f1 * x_t + W_f2 * h_t-1 + b_f)

* Input Gate: this gate with a function of previous hidden state and the current input, defines the necessary new information that should be added to the cell state. it defines a function called i_t and maps it values between 0 and 1. New information is also defined as a function of prervious hidden state and the current input and its output is called N_t. i_t is then multiplied by N_t to decide which new information should be passed to the next cells. Finally the outcome of Forget gate and the Input Gate, update the current cell state using past and new information. 

  i_t = sigmoid(W_i1 * x_t + W_i2 * h_t-1 + b_i)
  
  N_t = tanh(W_c1 * x_t + W_c2 * h_t-1 + b_c)
  
  c_t = f_t * c_t-1 + i_t * N_t
  
* Output Gate: in the output gate, another function is defined to choose which information should be presented in the output of the current cell (hidden state of the current cell). this function is o_t with values between 0 and 1 (by applying a sigmoid function). then this o_t effects on the current cell state (which was updated by Forget and Input gate). (Note: cell state c_t is firstly passes through a tanh activation function)  

  o_t = sigmoid(W_o1 * x_t + W_o2 * h_t-1 + b_o)

  h_t = o_t * tanh(c_t)

### LSTM in Pytorch

Here we present a short and simple code to show how LSTM is implemented in the Pytorch framework. Firstly, let's look at the specific command: 

*torch.nn.LSTM(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)*

**input_size, hidden_size, num_layers, bias=True, batch_first, dropout, bidirectional** are exactly the same as mentioned in the primary RNN. 

**proj_size** is the size we may define to change the dimension of hidden state. LSTM performs this projection using a learnable matrix W. It projects the hidden state size before passing to the output:

h_t_projected = W_projection * h_t_primary 

W_projection.shape: (proj_size, hidden_size)

**INPUTS**

inputs of LSTM layer are: 

* input_sequence of size (B, L, F)

* initialization of hidden state and cell state in the first LSTM cell of the first layer. (default are initialized to random numbers with normal distribution (0, 1))

**OUTPUTS** 

it return three sets of outputs: (outputs, (h_o, c_o))

* *outputs* is the concatenaion of all hidden state of shape (B, L, hidden_size)

* *h_o* is the last hidden state (equals the last *output* in the stack) of shape: (1, B, hidden_size)

* *c_o* is the last cell state (or cell state of the last LSTM unit in the layer) of shape: (1, B, hidden_size)


```
import torch
from torch import nn

# Define dimensions 
batch_size = 8
seq_len = 10
feature_size = 16
hidden_size = 64
num_layers = 1

# Input sample
tensor = torch.randn(size=(batch_size, seq_len, feature_size))

lstm_layer = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

h0, c0 = torch.randn(size=(1, batch_size, hidden_size)), torch.randn(size=(1, batch_size, hidden_size))

output, (h_o, c_o) = lstm_layer(tensor, (h0, c0))
```
```
output.shape, h_o.shape, c_o.shape
```
```
# Last output equals last hidden state
output[0, 9, :], h_o[0, 0, :]
```
