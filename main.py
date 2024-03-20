# main.py
# Implementation of Neural Machine Translation in pytorch.
# Dataset: https://huggingface.co/datasets/ted_hrlr
# (https://huggingface.co/datasets/ted_hrlr/viewer/pt_to_en)


from __future__ import unicode_literals, print_function, division
from io import open
import json
import logging
import random
import re
import time
import unicodedata

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data import Dataset, dataloader


# Set device.
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

# Load the data.
data = load_dataset("ted_hrlr", "pt_to_en")
train_examples, val_examples = data["train"], data["validation"]
# print(train_examples)
# print(train_examples["translation"])
print("Translations:")
print(json.dumps(train_examples["translation"][:3], indent=4))

# Original example used a pretrained tensorflow subword tokenizer. It 
# is not possible to convert this tokenizer to pytorch quickly. I was
# going to use the tokenizer devised in this pytorch example/tutorial:
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# however, this was primarily for the english to french translation
# dataset (provided in that pytorch example) and didnt have the
# necessary data for english to portuguese translation done in the
# tensorflow example. I decided to use this multilingual tokenizer for
# BERT multilingual base cased (thank you Google):
# https://huggingface.co/google-bert/bert-base-multilingual-cased
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Test tokenizer.
samples = train_examples["translation"][:3]
en_pt = [(sample["en"], sample["pt"]) for sample in samples]
en = [sample[0] for sample in en_pt]
pt = [sample[1] for sample in en_pt]
print("> English:")
print(f"{en[0]}")
print("> English tokenized:")
print(tokenizer.encode(en[0]))
print("> Portuguese:")
print(f"{pt[0]}")
print("> Portuguese tokenized:")
print(tokenizer.encode(pt[0]))


# Use pytorch dataset to do the translation.
class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_tokens):
        super().__init__()
        self.dataset = hf_dataset["translation"]
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Tokenize the texts. Dont forget to pad and truncate to
        # MAX_TOKENS. Reference guide here:
        # https://huggingface.co/docs/transformers/en/pad_truncation
        en_tokens = tokenizer.encode(
            sample["en"], 
            truncation=True,
            max_length=self.max_tokens,
            padding="max_length",
            return_tensors="pt"
        )
        pt_tokens = tokenizer.encode(
            sample["pt"], 
            truncation=True,
            max_length=self.max_tokens,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "en_inputs": en_tokens[:, :-1],
            "en_labels": en_tokens[:, 1:],
            "pt": pt_tokens
        }
    

# Pass the huggingface dataset through the custom dataset subclass for
# each split.
MAX_TOKENS = 128
train_set = TranslationDataset(train_examples, tokenizer, MAX_TOKENS)
valid_set = TranslationDataset(val_examples, tokenizer, MAX_TOKENS)

# Batch the dataset with a dataloader.
BATCH_SIZE = 64
train_loader = DataLoader(
    train_set, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
valid_loader = DataLoader(
    valid_set, 
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Test dataset.
train_features = next(iter(train_loader))
print(f"> inputs")
print(train_features["en_inputs"])
print(train_features["pt"])
print(f"> outputs")
print(train_features["en_labels"])


# Positional Encoding.
def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    ) 
    
    return torch.from_numpy(pos_encoding).type(torch.float32)


# Positional Embedding.
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            length=2048, depth=d_model
        )


    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    

    def forward(self, x):
        length = x.shape[1]
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, torch.float32))
        x = x + self.pos_encoding[:length, :].unsqueeze(0)
        return x
    

# Base Attention.
class BaseAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(**kwargs)
        self.layernorm = nn.LayerNorm()
        self.add = torch.add()


# Cross Attention.
class CrossAttention(BaseAttention):
    def forward(self, x, content):
        attn_output = self.mha(
            x, content, content
        )
        
        x = self.add(x, attn_output)
        x = self.layernorm(x)
        
        return x
    

# Global Self Attention.
class GlobalSelfAttention(BaseAttention):
    def forward(self, x):
        attn_output = self.mha(x, x, x)
        x = self.add(x, attn_output)
        x = self.layernorm(x)
        return x
    

# Causal Self Attention.
class CausalSelfAttention(BaseAttention):
    def forward(self, x):
        attn_output = self.mha(x, x, x, is_causal=True)
        x = self.add(x, attn_output)
        x = self.layernorm(x)
        return x
    

# Feed Forward.
class FeedForward(nn.Module):
    def __init__(self, input_dim, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = nn.Sequential([
            nn.Linear(input_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout_rate),
        ])
        self.add = torch.add()
        self.layernorm = nn.LayerNorm()


    def forward(self, x):
        x = self.add(x, self.seq(x))
        x = self.layernorm(x)
        return x



# Encoder Layer.
class EncoderLayer(nn.Module):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        embed_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)


    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    

# Encoder.
class Encoder(nn.Module):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.
    

# Decoder Layer.
class DecoderLayer(nn.Module):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)


    def forward(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x
        

# Decoder.
class Decoder(nn.Module):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = nn.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

    def forward(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


# Transformer.
class Transformer(nn.Module):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = nn.Linear(d_model, target_vocab_size)


    def forward(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x  = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        # try:
        #     # Drop the keras mask, so it doesn't scale the losses/metrics.
        #     # b/250038731
        #     del logits._keras_mask
        # except AttributeError:
        #     pass

        # Return the final output and the attention weights.
        return logits


# Hyperparameters.
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizer.vocab_size,
    target_vocab_size=tokenizer.vocab_size,
    dropout_rate=dropout_rate
)

optimizer = optim.Adam(
   transformer.parameters, lr=1e-5, betas=(0.9, 0.98), eps=1e-9
)

exit()

