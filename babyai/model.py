import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import babyai.rl
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from babyai.rl.utils.supervised_losses import required_heads
from babyai.layers import Encoder, Gate, clones, ReAttention
from babyai.layers import ConvolutionalNet
from babyai.layers import Attention, BahdanauAttentionDecoderRNN


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(3 * max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class ImageBOWEmbeddingPretrained(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.embedding = pretrained_model.get_input_embeddings()
        # self.apply(initialize_parameters)

    def forward(self, inputs):
        return self.embedding(inputs.long()).sum(1).permute(0, 3, 1, 2)


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False,
                 arch="bow_endpool_res", aux_info=None, finetune_transformer = False):
        super().__init__()

        endpool = 'endpool' in arch
        self.use_bow = 'bow' in arch
        self.pixel = 'pixel' in arch
        self.res = 'res' in arch
        self.use_attention = 'attention' in arch
        use_film = not self.use_attention

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        if self.res and image_dim != 128:
            raise ValueError(f"image_dim is {image_dim}, expected 128")
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space

        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res', 'attention', 'film']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        if self.lang_model == 'transformer':
            if not self.use_instr:
                raise ValueError("Transformers cannot be used when instructions are disabled")
            self.use_transformer = True
            self.instr_dim = 768
            if finetune_transformer:
                self.instr_rnn = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
            else:
                self.instr_rnn = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').requires_grad_(False)
            self.final_instr_dim = self.instr_dim
        else:
            self.use_transformer = False

        if use_film:
            self.image_conv = nn.Sequential(*[
                *([ImageBOWEmbedding(obs_space['image'], 128)] if self.use_bow else []),
                *([ImageBOWEmbeddingPretrained(self.instr_rnn)] if self.use_transformer and not self.pixel else []),
                *([nn.Conv2d(
                    in_channels=3, out_channels=128, kernel_size=(8, 8),
                    stride=8, padding=0)] if self.pixel else []),
                nn.Conv2d(
                    in_channels=128 if (self.use_bow and not self.use_transformer) or self.pixel else 768 if self.use_transformer else 3, out_channels=128,
                    kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
            ])
            self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if endpool else (2, 2), stride=2)
        elif self.use_attention:
            self.image_conv = nn.Sequential(*[
                *([ImageBOWEmbedding(obs_space['image'], 128)] if self.use_bow else []),
                *([nn.Conv2d(
                    in_channels=3, out_channels=128, kernel_size=(8, 8),
                    stride=8, padding=0)] if self.pixel else []),
                nn.Conv2d(
                    in_channels=128 if self.use_bow or self.pixel else 3, out_channels=128,
                    kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ])
        else:
            raise ValueError("Incorrect Architecture name: {}".format(arch))

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                elif self.lang_model == 'transformer':
                    pass
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model in ['attgru', 'transformer']:
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)
              
            if use_film:
                num_module = 2
                self.controllers = []
                for ni in range(num_module):
                    mod = FiLM(
                        in_features=self.final_instr_dim,
                        out_features=128 if ni < num_module-1 else self.image_dim,
                        in_channels=128, imm_channels=128)
                    self.controllers.append(mod)
                    self.add_module('FiLM_' + str(ni), mod)
        if self.use_attention:
            self.encoder = Encoder(self.instr_dim, 4, 9, self.instr_dim, self.instr_dim, self.image_dim)
            self.gates = clones(Gate(4, self.image_dim, self.image_dim), 3)
            self.post_cnn = nn.Sequential(*[
                nn.Conv2d(128, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=7, stride=7)
            ])
            self.reattention = ReAttention(4, 128, 128)

        # Define memory and resize image embedding
        self.embedding_size = self.image_dim
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
            self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None, probe_attention = False):
        if self.use_instr and instr_embedding is None:
            instr_embedding= self._get_instr_embedding(obs.instr)

        if self.use_instr and self.lang_model in {"attgru", "transformer"}:
            # outputs: B x L x D
            # memory: B x M
            if self.lang_model == 'transformer':
                mask = obs.instr.attention_mask.float()
            else:
                mask = (obs.instr != 0).float()
            # The mask tensor has the same length as obs.instr, and
            # thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, :instr_embedding.shape[1]]
            instr_embedding = instr_embedding[:, :mask.shape[1]]
            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if 'pixel' in self.arch:
            x /= 256.0

        x = self.image_conv(x)

        if self.use_instr and not self.use_attention:
            for controller in self.controllers:
                out = controller(x, instr_embedding)
                if self.res:
                    out += x
                x = out
            x = F.relu(self.film_pool(x))
        
        if self.use_attention:
            instr_encoded = self.encoder(instr_embedding, t=1)

            module_outputs = []
            for t in range(3):
                x = self.gates[t](x.permute(0, 2, 3, 1), instr_encoded, t)
                module_outputs.append(x)
            x = self.reattention(module_outputs, instr_encoded)
            x = self.post_cnn(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            print(hidden[0].shape, hidden[1].shape)
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        if not probe_attention:
            return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}
        else:
            if self.lang_model == 'transformer':
                return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions, 'attention': attention, 'encoded_inputs': dict(obs.instr)}
            elif self.lang_model == 'attgru':
                return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions, 'attention': attention, 'encoded_inputs': obs.instr}
            else:
                print('No probes available for instruction architecture {}'.format(self.lang_model))
                return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

    def _get_instr_embedding(self, instr):
        if self.lang_model == 'gru':
            lengths = (instr != 0).sum(1).long()
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden

        elif self.lang_model in ['bigru', 'attgru']:
            lengths = (instr != 0).sum(1).long()
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous() # [batch_size, num_layers * num_directions, hidden_size]
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == 'attgru' else final_states

        elif self.lang_model == 'transformer':
            outputs = self.instr_rnn(**instr).last_hidden_state
            return outputs

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))


class gSCAN(nn.Module):
    def __init__(self, obs_space, action_space,
                 num_encoder_layers: int, num_decoder_layers: int, image_dim: int, aux_info=None):
        super(gSCAN, self).__init__()
    
        self.obs_space = obs_space
        self.aux_info = aux_info
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.memory_dim = 128
        self.image_dim = image_dim * 2

        # Input: [batch_size, num_channels, image_height, image_width]
        # Output: [batch_size, image_height * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=3,
                                                 cnn_kernel_size=3,
                                                 num_conv_channels=128,
                                                 dropout_probability=0.5)

        # Input: [bsz, 1, decoder_hidden_size], [bsz, image_height * image_width, cnn_hidden_num_channels * 3]
        # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_height * image_width]
        self.visual_attention = Attention(key_size=128 * 3, query_size=128,
                                          hidden_size=128)


        # Instruction Encoder LSTM
        self.word_embedding = nn.Embedding(obs_space["instr"], 128)
        self.instr_rnn = nn.LSTM(
            input_size=128, hidden_size=128, batch_first=True,
            bidirectional=True, num_layers=self.num_encoder_layers)
        self.final_instr_dim = 128
        # Used to project the final encoder state to the decoder hidden state such that it can be initialized with it
        self.enc_hidden_to_dec_hidden = nn.Linear(128, 128)
        self.textual_attention = Attention(key_size=128, query_size=128, hidden_size=128)

        self.memory_rnn = nn.LSTM(input_size=self.image_dim, hidden_size=self.memory_dim,
                                  num_layers=self.num_decoder_layers)
        self.embedding_size = self.memory_dim

        # Input: [batch_size, max_target_length], initial hidden: ([batch_size, hidden_size], [batch_size, hidden_size])
        # Input for attention: [batch_size, max_input_length, hidden_size],
        #                      [batch_size, image_width * image_width, hidden_size]
        # Output: [max_target_length, batch_size, target_vocabulary_size]
                                                             
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    def _get_instr_embedding(self, instr):
        # get lenghts and masks
        lengths = (instr != 0).sum(1).long()
        masks = (instr != 0).float()

        # if reverse sorting is needed, reverse sort
        if lengths.shape[0] > 1:
            seq_lengths, perm_idx = lengths.sort(0, descending=True)
            iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
            if instr.is_cuda: iperm_idx = iperm_idx.cuda()

            for i, v in enumerate(perm_idx):
                iperm_idx[v.data] = i

            inputs = self.word_embedding(instr)
            inputs = inputs[perm_idx]

            inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

            outputs, (final_states, cell) = self.instr_rnn(inputs)
            # final_states [num_layers * num_directions, batch_size, hidden_size]
        else:
            instr = instr[:, 0:lengths[0]]
            outputs, final_states = self.instr_rnn(self.word_embedding(instr))
            iperm_idx = None

        final_states = final_states.transpose(0, 1).contiguous() # [batch_size, num_layers * num_directions, hidden_size]
        final_states = final_states.view(final_states.shape[0], self.num_encoder_layers, 2, -1) # [batch_size, num_layers, num_directions, hidden_size]

        # sum backward and forward directions of LSTM
        final_states = torch.sum(final_states, 2) # [batch_size, num_layers, hidden_size]

        # get the last layer
        final_states = final_states[:, -1, :] # [batch_size, hidden_size] (get last layer)

        if iperm_idx is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True) # [batch_size, seq_len, 2 * hidden_size]
            outputs = outputs.view(outputs.size(0), outputs.size(1), 2, -1)
            # same for outputs
            outputs = torch.sum(outputs, 2) # [batch_size, seq_len, hidden_size]
            outputs = outputs.index_select(dim=0, index=iperm_idx)
            final_states = final_states.index_select(dim=0, index=iperm_idx)

        return outputs, final_states, lengths

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim


    def encode_inputs(self, instr, visual_input):
        encoded_image = self.situation_encoder(visual_input)
        outputs, final_state, command_lengths = self._get_instr_embedding(instr)
        return {"encoded_image": encoded_image, "encoded_commands": outputs, "final_state": final_state, "command_lengths": command_lengths}
    
    
    def initialize_hidden(self, encoder_message):
        encoder_message = encoder_message.unsqueeze(0) # [1, batch_size, hidden_size]
        encoder_message = encoder_message.expand(
            self.num_decoder_layers,
            -1, 
            -1
        ).contiguous()
        return encoder_message.clone(), encoder_message.clone()


    def forward(self, obs, memory, probe_attention=False):
        # compute encoder output
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        encoder_output = self.encode_inputs(obs.instr, x)

        # get encoder outputs
        initial_hidden = encoder_output["final_state"]
        encoded_commands = encoder_output["encoded_commands"]
        command_lengths = encoder_output["command_lengths"]
        encoded_situations = encoder_output["encoded_image"]

        # for efficiency
        projected_keys_visual = self.visual_attention.key_layer(encoded_situations)
        projected_keys_textual = self.textual_attention.key_layer(encoded_commands)

        hidden = self.initialize_hidden(
            torch.tanh(self.enc_hidden_to_dec_hidden(initial_hidden)))
        last_hidden, last_cell = hidden
        
        # context_command [bsz, 1, value_dim]
        # attention_weights_command [bsz, 1, num_memory]
        context_command, attention_weights_commands = self.textual_attention(
            queries=last_hidden.transpose(0, 1), projected_keys=projected_keys_textual,
            values=projected_keys_textual, memory_lengths=command_lengths)
        batch_size, image_num_memory, _ = projected_keys_visual.size()
        situation_lengths = [image_num_memory for _ in range(batch_size)]

        queries = last_hidden.transpose(0, 1)

        # context : [batch_size, 1, hidden_size]
        # attention_weights : [batch_size, 1, max_input_length]
        context_situation, attention_weights_situations = self.visual_attention(
            queries=queries, projected_keys=projected_keys_visual,
            values=projected_keys_visual, memory_lengths=situation_lengths)

        # Concatenate the context vector and RNN hidden state, and map to an output
        attention_weights_commands = attention_weights_commands.squeeze(1)  # [batch_size, max_input_length]
        attention_weights_situations = attention_weights_situations.squeeze(1)  # [batch_size, im_dim * im_dim]

        concat_input = torch.cat([context_command.transpose(0, 1),
                                  context_situation.transpose(0, 1)], dim=2)  # [batch_size, hidden_size*3]

        hidden = (last_hidden, last_cell)

        _, hidden = self.memory_rnn(concat_input, hidden)

        # remove num_layers dimension from hidden output
        (h_n, c_n) = hidden
        h_n = h_n.squeeze(0)
        c_n = c_n.squeeze(0)

        # set embedding
        embedding = h_n

        # reconstitute back into tuple
        hidden = (h_n, c_n)

        memory = torch.cat(hidden, dim=1)
        
        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}