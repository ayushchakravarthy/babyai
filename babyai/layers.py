import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from torch.autograd import Variable
from typing import List, Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sequence_mask(sequence_lengths: torch.LongTensor, max_len=None) -> torch.tensor:
    """
    Create a sequence mask that masks out all indices larger than some sequence length as defined by sequence_length entries.

    :param sequence_lengths: [batch_size] sequence lengths per example in batch
    :param max_len: int defining the maximum sequence length in the batch
    :return: [batch_size, mask_len] boolean mask
    """
    if max_len is None:
        max_len = sequence_lengths.data.max()
    batch_size = sequence_lengths.size(0)
    sequence_range = torch.arange(0, max_len).long().to(device=device)

    # [batch_size, max_len]
    sequence_range_expand = sequence_range.unsqueeze(0).expand(batch_size, max_len)

    # [batch_size, max_len]
    seq_length_expand = (sequence_lengths.unsqueeze(1).expand_as(sequence_range_expand))

    return sequence_range_expand < seq_length_expand
    

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=800):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, x.size(1)], requires_grad=False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
        dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, d_model, h, seq_len, in_channels, imm_channels, out_channels):
        super().__init__()
        self.position_encoding = PositionalEncoding(d_model, dropout=0.0)
        self.multi_head = MultiHeadedAttention(h, d_model, dropout=0.0)
        self.transition = nn.Linear(in_channels, imm_channels)
        self.layer_norm1 = LayerNorm(torch.Size([d_model]))
        self.reshape = nn.Linear(imm_channels, out_channels)
        self.layer_norm2 = LayerNorm(torch.Size([out_channels]))
        self.apply(initialize_parameters)

    def forward(self,x,t):
        x = self.position_encoding(x)
        for _ in range(t):
            mh = self.multi_head(x, x, x)
            x = x + mh
            norm = self.layer_norm1(x)

            tran = self.transition(norm.squeeze(0))
            tran = tran + norm
            x = self.layer_norm1(tran)
            
            x = self.reshape(x)
            x = self.layer_norm2(x)

        return x

class Gate(nn.Module):
    def __init__(self, h, in_channels, out_channels):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.multihead = MultiHeadedAttention(h, out_channels)
        # self.layer_norm = [
        #     nn.LayerNorm(torch.Size([8,8,out_channels])),
        #     nn.LayerNorm(torch.Size([4,4,out_channels])),
        #     nn.LayerNorm(torch.Size([2,2,out_channels])),
        # ]
        self.layer_norm = nn.LayerNorm(torch.Size([7, 7, out_channels]))


    def forward(self, obs, instr, t=None):
        x = self.multihead(obs, instr, instr)
        x = x.view(x.size(0), obs.size(1), obs.size(2), x.size(-1))
        x = self.layer_norm(x)
        x = x + obs
        x = x.permute(0, 3, 1, 2)
        x = self.transition(x)

        return x


class ReAttention(nn.Module):
    def __init__(self, h, in_channels, out_channels):
        super().__init__()
        self.attention = MultiHeadedAttention(h, out_channels)
        self.layer_norm = nn.LayerNorm(torch.Size([7, 7, out_channels]))
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.linears = nn.ModuleList(
            [nn.Linear(2 * out_channels, out_channels) for _ in range(3)])


    def forward(self, outputs, enc_out):
        dec1 = outputs[0]
        dec2 = outputs[1]
        dec3 = outputs[2]

        dec1 = dec1.permute(0, 2, 3, 1)
        dec2 = dec2.permute(0, 2, 3, 1)
        dec3 = dec3.permute(0, 2, 3, 1)

        out = enc_out

        att1 = self.att(dec1, out, out)
        att2 = self.att(dec2, out, out)
        att3 = self.att(dec3, out, out)

        alpha1 = torch.sigmoid(self.linears[0](torch.cat([dec1, att1], -1)))
        alpha2 = torch.sigmoid(self.linears[1](torch.cat([dec2, att2], -1)))
        alpha3 = torch.sigmoid(self.linears[2](torch.cat([dec3, att3], -1)))

        outs = (alpha1 * att1 + alpha2 * att2 + alpha3 * att3) / math.sqrt(3)
        outs = self.transition(outs.permute(0, 3, 1, 2))

        return outs

    def att(self, query, key, value):
        att = self.attention(query, key, value)
        att = att.view(query.size(0), query.size(1), query.size(2), -1)
        out = self.layer_norm(att)

        out = out + query

        return out

class ConvolutionalNet(nn.Module):
    """Simple conv. nt. convolves the input channel but retains input image width."""
    def __init__(
        self,
        num_channels: int,
        cnn_kernel_size: int, 
        num_conv_channels: int,
        dropout_probability: float,
        stride=1):
        super(ConvolutionalNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=(3, 3),
                                padding=1, stride=stride)
        self.conv_2 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=5,
                                stride=stride, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=cnn_kernel_size,
                                stride=stride, padding=cnn_kernel_size // 2)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()
        layers = [self.relu, self.dropout]
        self.layers = nn.Sequential(*layers)
        self.output_dimension = num_conv_channels * 3

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        :param input_images: [batch_size, image_channels, image_width, image_width]
        :return: [batch_size, image_width * image_width, 3 * num_conv_channels]
        """
        batch_size = input_images.size(0)
        conved_1 = self.conv_1(input_images)
        conved_2 = self.conv_2(input_images)
        conved_3 = self.conv_3(input_images)
        images_features = torch.cat([conved_1, conved_2, conved_3], dim=1)
        _, num_channels, _, image_dimension = images_features.size()
        images_features = images_features.transpose(1, 3)
        images_features = self.layers(images_features)
        return images_features.reshape(batch_size, image_dimension * image_dimension, num_channels)

class DownSamplingConvolutionalNet(nn.Module):
    """TODO: make more general and describe"""
    def __init__(self, num_channels: int, num_conv_channels: int, dropout_probability: float):
        super(DownSamplingConvolutionalNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=5,
                                stride=5)
        self.conv_2 = nn.Conv2d(in_channels=num_conv_channels, out_channels=num_conv_channels, kernel_size=3,
                                stride=3, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=num_conv_channels, out_channels=num_conv_channels, kernel_size=3,
                                stride=3, padding=1)
        self.dropout = nn.Dropout2d(dropout_probability)
        self.relu = nn.ReLU()
        layers = [self.conv_1, self.relu, self.dropout, self.conv_2, self.relu, self.dropout, self.conv_3,
                  self.relu, self.dropout]
        self.layers = nn.Sequential(*layers)
        self.output_dimension = num_conv_channels * 3

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        :param input_images: [batch_size, image_width, image_width, image_channels]
        :return: [batch_size, 6 * 6, output_dim]
        """
        batch_size = input_images.size(0)
        input_images = input_images.transpose(1, 3)
        images_features = self.layers(input_images)
        _, num_channels, _, image_dimension = images_features.size()
        images_features = images_features.transpose(1, 3)
        return images_features.reshape(batch_size, image_dimension, image_dimension, num_channels)


class EncoderRNN(nn.Module):
    """
    Embed a sequence of symbols using an LSTM

    RNN hidden vector is captured for attention
    """
    def __init__(self, input_size: int, embedding_dim: int, rnn_input_size: int, hidden_size: int, num_layers: int,
                 dropout_probability: float, bidirectional: bool):
        """
        :param input_size: number of input symbols
        :param embedding_dim: number of hidden layers in the RNN encoder, and size of all embeddings
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol encodings and RNN
        :param bidirectional: using bidirectional LSTM instead and sum of the resulting embeddings
        """

        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.dropout_probability = dropout_probability
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout_probability, bidirectional=bidirectional)

    def forward(self, input_batch: torch.LongTensor, input_lengths: List[int]) -> Tuple[torch.Tensor, dict]:
        """
        :param input_batch: [batch_size, max_length]; batched padded input sequences
        :param input_lengths: length of each padded input sequence
        :return hidden states for last layer of last time step, the output of the last layer per time step and
        the sequence lengths per example in the batch.
        NB: The hidden states in the bidirectional case represent the final hidden state of each directional encoder,
        meaning the whole sequence in both directions, whereas the output per time step represents different parts of
        the sequences (0:t for the forward LSTM, t:T for the backward LSTM).
        """
        assert input_batch.size(0) == len(input_lengths)
        input_embeddings = self.embedding(input_batch) # [batch_size, max_length, embedding_dims]
        input_embeddings = self.dropout(input_embeddings) # [batch_size, max_length, embedding_dims]

        # Sort sequences in descending order
        batch_size = len(input_lengths)
        max_length = max(input_lengths)
        input_lengths = torch.tensor(input_lengths, device=device, dtype=torch.long)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_embeddings = input_embeddings.index_select(dim=0, index=perm_idx)

        # RNN Embeddings
        packed_input = pack_padded_sequence(input_embeddings, input_lengths.cpu(), batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # hidden, cell [num_layers * num_directions, batch_size, embedding_dim]
        # hidden and cell are unpacked, such that they store the last hidden state for each sequence in the batch
        output_per_timestep, _ = pad_packed_sequence(packed_output) # [max_length, batch_size, hidden_size * num_directions]

        # If biLSTM, sum the outputs for each direction
        if self.bidirectional:
            output_per_timestep = output_per_timestep.view(int(max_length), batch_size, 2, self.hidden_size)
            output_per_timestep = torch.sum(output_per_timestep, 2) # [max_length, batch_size, hidden_size]
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            hidden = torch.sum(hidden, 1) # [num_layers, batch_size, hidden_size]
        hidden = hidden[-1, :, :] # [batch_size, hidden_size] (get the last layer)

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden.index_select(dim=0, index=unperm_idx)
        output_per_timestep = output_per_timestep.index_select(dim=1, index=unperm_idx)
        input_lengths = input_lengths[unperm_idx].tolist()
        return hidden, {"encoder_outputs": output_per_timestep, "sequence_lengths": input_lengths}

    def extra_repr(self) -> str:
        return f"EncoderRNN\n bidirectional={self.bidirectional} \n num_layers={self.num_layers} \n hidden_size={self.hidden_size} \n dropout={self.dropout_probability} \n n_input_symbols={self.input_size} \n"


class Attention(nn.Module):
    def __init__(self, key_size: int, query_size: int, hidden_size: int):
        super(Attention, self).__init__()
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, queries: torch.Tensor, projected_keys: torch.Tensor, values: torch.Tensor,
                memory_lengths: List[int]):
        """
        Key-value memory which takes queries and retrieves weighted combinations of values
          This version masks out certain memories, so that you can differing numbers of memories per batch.
        :param queries: [batch_size, 1, query_dim]
        :param projected_keys: [batch_size, num_memory, query_dim]
        :param values: [batch_size, num_memory, value_dim]
        :param memory_lengths: [batch_size] actual number of keys in each batch
        :return:
            soft_values_retrieval : soft-retrieval of values; [batch_size, 1, value_dim]
            attention_weights : soft-retrieval of values; [batch_size, 1, n_memory]
        """
        batch_size = projected_keys.size(0)
        assert len(memory_lengths) == batch_size
        memory_lengths = torch.tensor(memory_lengths, dtype=torch.long, device=device)

        # Project queries down to the correct dimension.
        # [bsz, 1, query_dimension] x [bsz, query_dimension, hidden_dim] = [bsz, 1, hidden_dim]
        queries = self.query_layer(queries)

        # [bsz, 1, query_dimension] x [bsz, query_dimension, num_memory] = [bsz, num_memory, 1]
        scores = self.energy_layer(torch.tanh(queries + projected_keys))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out the keys that are on a padding location
        mask = sequence_mask(memory_lengths) # [batch_size, num_memory]  
        mask = mask.unsqueeze(1) # [batch_size, 1, num_memory]
        scores = scores.masked_fill(mask == 0, float('-inf')) # fill with large negative numbers based on sequence mask
        attention_weights = F.softmax(scores, dim=2) # [batch_size, 1, num_memory]

        # [bsz, 1, num_memory] x [bsz, num_memory, value_dim] = [bsz, 1, value_dim]
        soft_value_retrieval = torch.bmm(attention_weights, values)
        return soft_value_retrieval, attention_weights


class BahdanauAttentionDecoderRNN(nn.Module):
    """One-step batch decoder with Bahdanau et al. attention"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, textual_attention: Attention,
                 visual_attention: Attention, dropout_probability=0.1, padding_idx=0):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(BahdanauAttentionDecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(hidden_size * 3, hidden_size, num_layers=num_layers, dropout=dropout_probability)
        self.textual_attention = textual_attention
        self.visual_attention = visual_attention
        self.output_to_hidden = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=False)

    def forward_step(self, input_tokens: torch.LongTensor, last_hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoded_commands: torch.Tensor, commands_lengths: torch.Tensor,
                     encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run batch decoder forward for a single time step.
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)
        :param input_tokens: one time step inputs tokens of length batch_size
        :param last_hidden: previous decoder state, which is pair of tensors [num_layers, batch_size, hidden_size]
        (pair for hidden and cell)
        :param encoded_commands: all encoder outputs, [max_input_length, batch_size, hidden_size]
        :param commands_lengths: length of each padded input sequence that were passed to the encoder.
        :param encoded_situations: the situation encoder outputs, [image_dimension * image_dimension, batch_size,
         hidden_size]
        :return: output : un-normalized output probabilities, [batch_size, output_size]
          hidden : current decoder state, which is a pair of tensors [num_layers, batch_size, hidden_size]
           (pair for hidden and cell)
          attention_weights : attention weights, [batch_size, 1, max_input_length]
        """
        last_hidden, last_cell = last_hidden

        # Embed each input symbol
        embedded_input = self.embedding(input_tokens)  # [batch_size, hidden_size]
        embedded_input = self.dropout(embedded_input)
        embedded_input = embedded_input.unsqueeze(0)  # [1, batch_size, hidden_size]

        # Bahdanau attention
        context_command, attention_weights_commands = self.textual_attention(
            queries=last_hidden.transpose(0, 1), projected_keys=encoded_commands.transpose(0, 1),
            values=encoded_commands.transpose(0, 1), memory_lengths=commands_lengths)
        batch_size, image_num_memory, _ = encoded_situations.size()
        situation_lengths = [image_num_memory for _ in range(batch_size)]

        queries = last_hidden.transpose(0, 1)

        context_situation, attention_weights_situations = self.visual_attention(
            queries=queries, projected_keys=encoded_situations,
            values=encoded_situations, memory_lengths=situation_lengths)
        # context : [batch_size, 1, hidden_size]
        # attention_weights : [batch_size, 1, max_input_length]

        # Concatenate the context vector and RNN hidden state, and map to an output
        attention_weights_commands = attention_weights_commands.squeeze(1)  # [batch_size, max_input_length]
        attention_weights_situations = attention_weights_situations.squeeze(1)  # [batch_size, im_dim * im_dim]
        concat_input = torch.cat([embedded_input,
                                  context_command.transpose(0, 1),
                                  context_situation.transpose(0, 1)], dim=2)  # [1, batch_size hidden_size*3]

        last_hidden = (last_hidden, last_cell)
        lstm_output, hidden = self.lstm(concat_input, last_hidden)
        # lstm_output: [1, batch_size, hidden_size]
        # hidden: tuple of each [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        # output = self.hidden_to_output(lstm_output)  # [batch_size, output_size]
        # output = output.squeeze(dim=0)

        # Concatenate all outputs and project to output size.
        pre_output = torch.cat([embedded_input, lstm_output,
                                context_command.transpose(0, 1), context_situation.transpose(0, 1)], dim=2)
        pre_output = self.output_to_hidden(pre_output)  # [1, batch_size, hidden_size]
        output = self.hidden_to_output(pre_output)  # [batch_size, output_size]
        output = output.squeeze(dim=0)   # [batch_size, output_size]

        return (output, hidden, attention_weights_situations.squeeze(dim=1), attention_weights_commands,
                attention_weights_situations)
        # output : [un-normalized probabilities] [batch_size, output_size]
        # hidden: tuple of size [num_layers, batch_size, hidden_size] (for hidden and cell)
        # attention_weights: [batch_size, max_input_length]

    def forward(self, input_tokens: torch.LongTensor, input_lengths: List[int],
                init_hidden: Tuple[torch.Tensor, torch.Tensor], encoded_commands: torch.Tensor,
                commands_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, List[int],
                                                                                        torch.Tensor]:
        """
        Run batch attention decoder forward for a series of steps
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)
        :param input_tokens: [batch_size, max_length];  padded target sequences
        :param input_lengths: [batch_size] for sequence length of each padded target sequence
        :param init_hidden: tuple of tensors [num_layers, batch_size, hidden_size] (for hidden and cell)
        :param encoded_commands: [max_input_length, batch_size, embedding_dim]
        :param commands_lengths: [batch_size] sequence length of each encoder sequence (without padding)
        :param encoded_situations: [batch_size, image_width * image_width, image_features]; encoded image situations.
        :return: output : unnormalized log-score, [max_length, batch_size, output_size]
          hidden : current decoder state, tuple with each [num_layers, batch_size, hidden_size] (for hidden and cell)
        """
        batch_size, max_time = input_tokens.size()

        # Sort the sequences by length in descending order
        input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_tokens_sorted = input_tokens.index_select(dim=0, index=perm_idx)
        initial_h, initial_c = init_hidden
        hidden = (initial_h.index_select(dim=1, index=perm_idx),
                  initial_c.index_select(dim=1, index=perm_idx))
        encoded_commands = encoded_commands.index_select(dim=1, index=perm_idx)
        commands_lengths = torch.tensor(commands_lengths, device=device)
        commands_lengths = commands_lengths.index_select(dim=0, index=perm_idx)
        encoded_situations = encoded_situations.index_select(dim=0, index=perm_idx)

        # For efficiency
        projected_keys_visual = self.visual_attention.key_layer(
            encoded_situations)  # [batch_size, situation_length, dec_hidden_dim]
        projected_keys_textual = self.textual_attention.key_layer(
            encoded_commands)  # [max_input_length, batch_size, dec_hidden_dim]

        all_attention_weights = []
        lstm_output = []
        for time in range(max_time):
            input_token = input_tokens_sorted[:, time]
            (output, hidden, context_situation, attention_weights_commands,
             attention_weights_situations) = self.forward_step(input_token, hidden, projected_keys_textual,
                                                               commands_lengths,
                                                               projected_keys_visual)
            all_attention_weights.append(attention_weights_situations.unsqueeze(0))
            lstm_output.append(output.unsqueeze(0))
        lstm_output = torch.cat(lstm_output, dim=0)  # [max_time, batch_size, output_size]
        attention_weights = torch.cat(all_attention_weights, dim=0)  # [max_time, batch_size, situation_dim**2]

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = lstm_output.index_select(dim=1, index=unperm_idx)  # [max_time, batch_size, output_size]
        seq_len = input_lengths[unperm_idx].tolist()
        attention_weights = attention_weights.index_select(dim=1, index=unperm_idx)

        return lstm_output, seq_len, attention_weights.sum(dim=0)
        # output : [unnormalized log-score] [max_length, batch_size, output_size]
        # seq_len : length of each output sequence

    def initialize_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the encoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message:  [batch_size, hidden_size] tensor
        :return: tuple of Tensors representing the hidden and cell state of shape: [num_layers, batch_size, hidden_dim]
        """
        encoder_message = encoder_message.unsqueeze(0)  # [1, batch_size, hidden_size]
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # [num_layers, batch_size, hidden_size]
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "AttentionDecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )