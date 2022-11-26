https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 11/14/20 8:47 PM
"""Seq2Seq and its componenets"""
import random
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import consts as C
import utils


def init_model(args: Namespace, input_dim: int, output_dim: int,
               tree_stoi: C.STOI_T, device: torch.device):
  is_attentional = False
  if args.model == 'seq2seq':
    print("\nVanilla Seq2Seq init")
    dec = Decoder(output_dim,
                  embed_dim=args.embed_dim,
                  hidden_dim=args.hidden_dim,
                  num_layers=args.num_layers,
                  dropout=args.dropout,
                  rnn_type=args.rnn)
  else:
    is_attentional = True
    if args.model == 'bahdanau':
      print("\nBahdanau Attention Seq2Seq init")
      dec = BahdanauAttentionDecoder(vocab_size=output_dim,
                                     embed_dim=args.embed_dim,
                                     hidden_dim=args.hidden_dim,
                                     num_layers=args.num_layers,
                                     dropout=args.dropout,
                                     rnn_type=args.rnn)
    else:
      mode = args.model.split('_')[-1] # `dot` vs `general`
      print(f"\nLuong {mode.capitalize()} Attention Seq2Seq init")
      dec = LuongAttentionDecoder(vocab_size=output_dim,
                                  embed_dim=args.embed_dim,
                                  hidden_dim=args.hidden_dim,
                                  num_layers=args.num_layers,
                                  dropout=args.dropout,
                                  rnn_type=args.rnn,
                                  attn_mode=mode)
  enc = Encoder(input_dim,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                rnn_type=args.rnn)
  if args.glove_dir:
    glove = utils.load_glove(args.glove_dir, args.embed_dim)
    enc.init_pretrained_embedding(glove.vectors, args.finetune_pretrained)

  net = Seq2Seq(enc, dec, tree_stoi, is_attentional=is_attentional, device=device)

  return net

class Encoder(nn.Module):
  """Recurrent Encoder"""
  def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
               rnn_type='lstm'):
    super().__init__()
    ### configs
    self.vocab_size = vocab_size
    self.rnn_type = rnn_type

    ### layers
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    if rnn_type == 'lstm':
      self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
    else:
      self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
    self.dropout = nn.Dropout(dropout)

  def init_pretrained_embedding(self, weights, finetune_pretrained=False):
    """initializes nn.Embedding layer with pre-trained embedding weights

    Args:
      weights: pre-trained embedding tensor, (`src_vocab_size`, `embed_dim`)
      finetune_pretrained: whether to finetune the embedding matrix during training, boolean
    """
    self.embedding = nn.Embedding.from_pretrained(weights, freeze=not finetune_pretrained)

  def forward(self, x, lengths):
    """encodes source sentences

    Args:
      x: source tensor, (`batch_size`, `src_seq_len`)
      lengths: valid source length tensor, (`batch_size`)

    Returns:
      outputs: RNN hidden states for all time-steps, (`batch_size`, `src_seq_len`, `hidden_dim`)
      state: Last RNN hidden state. Shape determined by whether RNN is LSTM or GRU:
        If LSTM: Tuple(
          (`num_layers`, `batch_size`, `hidden_dim`),
          (`num_layers`, `batch_size`, `hidden_dim`)
        ) where the first element is hidden state, the second is cell state
        If GRU:  (`num_layers`, `batch_size`, `hidden_dim`)
    """
    x = self.dropout(self.embedding(x))  # (`batch_size`, `seq_len`, `embed_size`)
    x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    outputs, state = self.rnn(x)
    outputs, _ = pad_packed_sequence(x, batch_first=True)
    return outputs, state

class Decoder(nn.Module):
  """Reducrrent Decoder without Attention"""
  def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
               rnn_type='lstm'):
    super().__init__()
    ### configs
    self.vocab_size = vocab_size
    self.rnn_type = rnn_type

    ### layers
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    if rnn_type == 'lstm':
      self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
    else:
      self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
    self.dense = nn.Linear(hidden_dim, vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, state):
    """decodes for `x.size(1)` (target sequence length) step(s).

    Target sequence length of x may be 1, in which case we take a single decoding
    step. Or, it may be bigger than 1, in which case x is likely a padded tensor
    and we always feed in gold target input to the decoder.

    When decoder takes a first step at t=0, `state` and `encoder_state` will be
    the same. From t=1 and afterwards, `state` will have been updated while
    `encoder_state` remains the same throughout.

    Args:
      x: target tensor, (`batch_size`, `tgt_seq_len`)
      state: decoder's previous RNN hidden state. Shape determined by whether RNN is LSTM or GRU:
        If LSTM: Tuple(
          (`num_layers` * `num_directions`, `batch_size`, `hidden_dim`),
          (`num_layers` * `num_directions`, `batch_size`, `hidden_dim`)
        ) where the first element is hidden state, the second is cell state
        If GRU: (`num_layers` * `num_directions`, `batch_size`, `hidden_dim`)

    Returns:
      output: token-level logits, (`batch_size`, `tgt_seq_len`, `vocab_size`)
      state: decoder's last RNN hidden state. Similar shape as `state`
    """
    x = self.dropout(self.embedding(x))  # (`batch_size`, `tgt_seq_len`, `embed_size`)

    # output: (`batch_size`, `tgt_seq_len`, `hidden_dim`)
    output, state = self.rnn(x, state)

    # output: (`batch_size`, `tgt_seq_len`, `vocab_size`)
    output = self.dense(output)
    return output, state

class BahdanauAttentionDecoder(nn.Module):
  """Bahdanau (Additive) Attentional Decoder

  score = v^T \cdot \tanh(W_h \cdot H_h + W_e \cdot H_e)
  """
  def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout,
               rnn_type='lstm'):
    """Initialize configs and define layers for Decoder with Bahdanau Attention

    Args:
      vocab_size: size of target vocab
      embed_dim: embedding feature dimension
      hidden_dim: RNN hidden dimension
      num_layers: RNN number of layers
      dropout: dropout probability
      rnn_type: GRU or LSTM
    """
    super().__init__()
    ### configs
    self.vocab_size = vocab_size
    self.rnn_type = rnn_type

    ### layers
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.dropout = nn.Dropout(dropout)
    adjusted_input_dim = embed_dim + hidden_dim
    if self.rnn_type == 'lstm':
      self.rnn = nn.LSTM(adjusted_input_dim, hidden_dim, num_layers=num_layers,
                         batch_first=True)
    else:
      self.rnn = nn.GRU(adjusted_input_dim, hidden_dim, num_layers=num_layers,
                        batch_first=True)
    self.fc_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.fc_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.fc_v = nn.Linear(hidden_dim, 1, bias=False)
    self.fc_classifier = nn.Linear(hidden_dim, vocab_size)

  def forward(self, x, decoder_state, encoder_outputs, mask=None):
    """Single decoder step with Bahdanau attention. Additive attention takes place
    when the outptus of `fc_hidden` and `fc_encoder` are summed. There are other
    interpretations that prefer concat over addition.

    `x` may be a gold decoder input (with teacher forcing) or a decoder's previous
    prediction (without teacher forcing).

    We do not use input-feeding approach, which conatenates the attentional context
    vector with the outputs from `fc_classifier`

    Args:
      x: decoder input, (`batch_size`, 1)
      decoder_state: decoder's previous RNN hidden state. Shape determined by whether RNN is LSTM or GRU:
        If LSTM: Tuple(
          (`num_layers` * `num_directions`, `batch_size`, `hidden_dim`),
          (`num_layers` * `num_directions`, `batch_size`, `hidden_dim`)
        ) where the first element is hidden state, the second is cell state
        If GRU: (`num_layers` * `num_directions`, `batch_size`, `hidden_dim`)
      encoder_outputs: RNN hidden states for all time-steps, (`batch_size`, `src_seq_len`, `hidden_dim`)
      mask: boolean matrix , (`batch_size`, 1)

    Returns:
      output: logits, (`batch_size`, 1, `vocab_size`)
      state: decoder's last RNN hidden state. Similar shape as `decoder_state`
    """
    # (`batch_size`, 1, `hidden_dim`)
    embedded = self.dropout(self.embedding(x))

    state = decoder_state
    if self.rnn_type == 'lstm':
      # (`num_layers`, `batch_size`, `hidden_dim`)
      decoder_state = decoder_state[0] # ignore cell state

    # (`batch_size`, 1, `hidden_dim`)
    top_decoder_state = decoder_state[-1].unsqueeze(1)

    # (`batch_size`, `src_seq_len`, `hidden_dim`)
    top_decoder_state = top_decoder_state.repeat(1, encoder_outputs.size(1), 1)

    # (`batch_size`, `src_seq_len`, `hidden_dim`)
    x = torch.tanh(self.fc_hidden(top_decoder_state) + self.fc_encoder(encoder_outputs))

    # (`batch_size`, `src_seq_len`)
    alignment_scores = self.fc_v(x).squeeze(-1)

    if mask is not None:
      # masked in-place so padding indices get negligible values after softmax
      alignment_scores.masked_fill_(mask, -1e10)

    # (`batch_size`, 1, `src_seq_len`)
    attn_weights = F.softmax(alignment_scores, dim=1).unsqueeze(1)

    # (`batch_size`, 1, `hidden_dim`)
    context_vector = torch.bmm(attn_weights, encoder_outputs)

    # (`batch_size`, 1, `embed_dim` + `hidden_dim`)
    output = torch.cat([embedded, context_vector], dim=2)

    # (`batch_size`, 1, `hidden_dim`)
    output, hidden = self.rnn(output, state)

    # (`batch_size`, 1, `vocab_size`)
    output = self.fc_classifier(output)

    return output, hidden

class LuongAttentionDecoder(nn.Module):
  """Luong (Multiplicative) Attention

  Dot:
    score = H_e \cdot H_h
  General:
    score = H_e \cdot W \cdot H_h

  Concat mode is similar to Bahdanau above, and you are not asked to implement it.
  """
  def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float,
               rnn_type='lstm', attn_mode='dot'):
    """Initialize configs and define layers for Decoder with Luong Attention

    Args:
      vocab_size: size of target vocab
      embed_dim: embedding feature dimension
      hidden_dim: RNN hidden dimension
      num_layers: RNN number of layers
      dropout: dropout probability
      rnn_type: GRU or LSTM
      attn_mode: 'dot' vs 'general'
    """
    super().__init__()
    raise NotImplementedError()

  def forward(self, x, decoder_state, encoder_outputs, mask=None):
    """Single decoder step with Luong attention.

    `x` may be a gold decoder input (with teacher forcing) or a decoder's previous
    prediction (without teacher forcing).

    We do not use input-feeding approach, which conatenates the attentional context
    vector with the outputs from `fc_classifier`

    Args:
      x: decoder input, (`batch_size`, 1)
      decoder_state: decoder's previous RNN hidden state. Shape determined by whether RNN is LSTM or GRU:
        If LSTM: Tuple(
          (`num_layers` * `num_directions`, `batch_size`, `hidden_dim`),
          (`num_layers` * `num_directions`, `batch_size`, `hidden_dim`)
        ) where the first element is hidden state, the second is cell state
        If GRU: (`num_layers` * `num_directions`, `batch_size`, `hidden_dim`)
      encoder_outputs: RNN hidden states for all time-steps, (`batch_size`, `src_seq_len`, `hidden_dim`)
      mask: boolean matrix , (`batch_size`, 1)

    Returns:
      output: logits, (`batch_size`, 1, `vocab_size`)
      state: decoder's last RNN hidden state. Similar shape as `decoder_state`
    """
    # ASSUME: `tgt_seq_len` == 1
    raise NotImplementedError()

class Seq2Seq(nn.Module):
  """Seq2seq"""
  def __init__(self, encoder, decoder, tree_stoi, is_attentional, device):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

    self.tree_stoi = tree_stoi
    self.bos_idx = tree_stoi[C.BOS]
    self.eos_idx = tree_stoi[C.EOS]

    self.is_attentional = is_attentional

    self.device = device

  def xavier_init_weights(self):
    """Xavier initialization"""
    # `torch.no_grad` occurs inside `xavier_uniform_`
    def xavier_init_fn(m):
      if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
      if type(m) == nn.GRU:
        for param in m._flat_weights_names:
          if "weight" in param:
            torch.nn.init.xavier_uniform_(m._parameters[param])
    self.apply(xavier_init_fn)

  def forward_tf(self, x, y, x_lengths):
    """Assuming full teacher forcing, allows RNNs to be unrolled automatically
    in a graph

    Args:
      x: see `forward`
      y: see `forward`
      x_lengths: see `forward`

    Returns:
      see `forward`
    """
    encoder_outputs, encoder_state = self.encoder(x, x_lengths)
    dec_output, dec_state = self.decoder(y, encoder_state)
    return dec_output, dec_state

  def forward(self, x, y, x_lengths, mask=None, teacher_forcing_ratio=0.0):
    """Process a single batch of data

    If `teacher_forcing_ratio` is set to 1, we allow RNNs to be unrolled automatically
    in a graph. Otherwise, we iterate through each time-step of a target sequence `y`

    Args:
      x: batch of sentences, (`batch_size`, `src_seq_len`)
      y: batch of trees, (`batch_size`, `tgt_seq_len`)
      x_lengths: valid sentence lengths, (`batch_size`)
      mask: mask over x padding indices, (`batch_size`, `src_seq_len`)
      teacher_forcing_ratio: ratio as a float to determine how to use teacher forcing in decoder

    Returns:
      outputs: Token-level logits, (`batch_size`, `tgt_seq_len`, `vocab_size`)
      state: Last RNN hidden state, (`num_layers`, `batch_size`, `hidden_dim`)
    """
    if teacher_forcing_ratio == 1.0 and not self.is_attentional:
      # hand over control to let graph take care of unrolling RNNs much more efficiently
      return self.forward_tf(x, y, x_lengths)

    batch_size, y_len = y.shape
    vocab_size = self.decoder.vocab_size

    ### encoder step
    encoder_outputs, state = self.encoder(x, x_lengths)

    ### deocder setup
    input_ = torch.tensor([[self.bos_idx]], dtype=torch.long, device=self.device)
    input_ = input_.repeat(batch_size, 1)

    ### decoder iteration
    outputs = torch.zeros(y_len, batch_size, vocab_size, device=self.device)

    # loop below iterates `y_len` - 1 times -> outputs[-1] will remain zero vectors
    # but regardless, outputs[-1] should be ignored since it's decoder's output
    # from padding or from EOS, which are both meaningless
    for i in range(y_len-1):
      # output: (`batch_size`, 1, `vocab_size`)
      if self.is_attentional:
        output, state = self.decoder(input_, state, encoder_outputs, mask)
      else:
        output, state = self.decoder(input_, state)
      outputs[i] = output.transpose(0, 1) # (1, `batch_size`, `vocab_size`)

      # input_: (`batch_size`, 1)
      if random.random() < teacher_forcing_ratio:
        input_ = y[:,i+1].contiguous().view(-1, 1)
      else:
        input_ = output.argmax(-1)

    # (`batch_size`, `seq_len`, `hidden_dim`)
    outputs = outputs.permute(1, 0, 2)

    return outputs, state
