# Simple Transformers

This is a very simple repository with to goal to understand how Transformers work from scratch entirely based on the YouTube tutorial: [Transformers from Scratch - by Ari Seff](https://www.youtube.com/watch?v=ISNdQcPhsts). 

## What Are Transformers?

Transformers are a type of neural network architecture that rely heavily on a mechanism called **self-attention**. Self-attention allows the model to consider relationships between each token in a sequence, including the token's relationship with itself. This makes Transformers be able to handle sequences like language, where context matters.

The Transformer architecture is composed of repeated blocks. The representations I like the most is this diagram from Wikipedia:

![Transformer Architecture](https://upload.wikimedia.org/wikipedia/commons/1/10/Transformer%2C_full_architecture.png)

### Positional Encodings

One of the first key elements in the architecture is **positional encoding**. Unlike RNNs, which are autoregressive (the model ries to predict token i, based on what has already seen, meaning all the tokens before it  i-1), Transformers process all tokens **in parallel**. This makes them **permutation invariant**, meaning without additional info, the model wouldn't know the correct order of the words.



To solve this, Transformers add **positional encodings** which are usually based on sine and cosine functions to the input embeddings. This tells the model the position of each token in the sequence.

### Self-Attention and Multi-Head Attention

A core component of the Transformer is the **multi-head self-attention mechanism**, which operates using:

- **Q (Query)**
- **K (Key)**
- **V (Value)**

A helpful analogy I take is searching for a video.

- The  **query** might be `"movie reviews"`.
- The **keys** are all possible video titles (e.g., `"Blade Runner 2049 review"`, `"Inception analysis"`, etc.).
- The **values** are the actual videos returned.

The self-attention operation is defined as:

Attention(Q, K, V) = softmax((Q × Kᵀ) / √dₖ) × V

Where:
- `Q` is the query matrix
- `K` is the key matrix
- `V` is the value matrix
- `dₖ` is the dimension of the key vectors

This operation lets the model calculate a weighted representation of the input, where the weights reflect how much focus each word should place on the others.


### Multi-Head?

Instead of computing one single attention representation, **multi-head attention** allows the model to look at different parts of the sequence from different representation subspaces. Each "head" learns a different aspect of the attention, and the results are concatenated and linearly transformed.

This allows the model to capture more complex relationships within the input sequence.

An amazing eplanation on this if found on this video: ![Visual Guide to Transformer Neural Networks](https://www.youtube.com/watch?v=mMa2PmYJlCo)

### Masking and Language Modeling

In decoder-based Transformers (like GPT), a **mask** is applied during training to prevent the model from looking ahead. This mimics real language use: when generating a sentence, we only know what came before, not after.

This is called **causal masking**, and it's essential for language generation tasks.

In contrast, **BERT** uses a different strategy called **Masked Language Modeling (MLM)**. It masks random tokens in the input and trains the model to predict them using **both** left and right context (i.e., bidirectional attention).
