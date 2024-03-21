# main.py
# Implementation of Neural Machine Translation in pytorch.
# Dataset: https://huggingface.co/datasets/ted_hrlr
# (https://huggingface.co/datasets/ted_hrlr/viewer/pt_to_en)
# Transformer model source: 
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch


from __future__ import unicode_literals, print_function, division
from io import open
import json
import logging
import math
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
print(f"Compute Device chosen: {device}")

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


# MultiHeadAttention.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

# Point-wise Feed Forward Network.
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()


    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

# Positional Encoding.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

# Encoder Layer.
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# Decoder Layer.
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# Transformer.
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(tgt_mask.device) # Code added by me. Had to add device logic so all tensors went to same device.
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask


    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


# Hyperparameters.
src_vocab_size = tokenizer.vocab_size
tgt_vocab_size = tokenizer.vocab_size
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
# max_seq_length = 100
max_seq_length = MAX_TOKENS
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
# src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

# Initialize Transformer model.
transformer = Transformer(
    src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout
)
transformer.to(device)

# Train the model.
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    loss_total = 0
    counter = 0
    for i, data in enumerate(train_loader):
        src_data = data["pt"]
        tgt_data = data["en_labels"]
        tgt_input = data["en_inputs"]

        src_data = src_data.to(device)
        tgt_data = tgt_data.to(device)
        tgt_input = tgt_input.to(device)

        # Check shapes. Expected to be just (batch_size, MAX_TOKENS).
        # print(f"src_data: {src_data.shape}")    # (batch_size, 1, MAX_TOKENS)
        # print(f"tgt_data: {tgt_data.shape}")    # (batch_size, 1, MAX_TOKENS - 1)
        # print(f"tgt_input: {tgt_input.shape}")  # (batch_size, 1, MAX_TOKENS - 1)
        src_data = src_data.squeeze(1)
        tgt_data = tgt_data.squeeze(1)
        tgt_input = tgt_input.squeeze(1)

        optimizer.zero_grad()
        # output = transformer(src_data, tgt_data[:, :-1])
        # loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        output = transformer(src_data, tgt_input)
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data.contiguous().view(-1))
        loss.backward()
        loss_total += loss.item()
        counter += 1
        optimizer.step()
    loss = loss_total / counter
    # print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    print(f"Epoch: {epoch+1}, Loss: {loss}")

# Model evaluation.
transformer.eval()

# Generate random sample validation data
# val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
# val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

with torch.no_grad():

    val_loss_total = 0
    counter = 0
    for i, data in enumerate(valid_loader):
        val_src_data = data["pt"]
        val_tgt_data = data["en_labels"]
        val_tgt_input = data["en_inputs"]

        val_src_data = val_src_data.to(device)
        val_tgt_data = val_tgt_data.to(device)
        val_tgt_input = val_tgt_input.to(device)

        val_src_data = val_src_data.squeeze(1)
        val_tgt_data = val_tgt_data.squeeze(1)
        val_tgt_input = val_tgt_input.squeeze(1)

        # val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        # val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
        val_output = transformer(val_src_data, val_tgt_input)
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data.contiguous().view(-1))
        val_loss_total += val_loss.item()
        counter += 1
    val_loss = val_loss_total / counter
    # print(f"Validation Loss: {val_loss.item()}")
    print(f"Validation Loss: {val_loss}")

exit()