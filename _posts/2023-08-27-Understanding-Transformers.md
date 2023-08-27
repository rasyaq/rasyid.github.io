# Transformers: The Model Architecture Revolutionizing AI
Transformers have emerged as one of the most influential innovations in deep learning. Originally introduced for natural language processing (NLP) tasks like translation and text generation, transformers are now driving advances across computer vision, speech recognition, and other AI domains. Transformers have catalyzed major advances in artificial intelligence, particularly through powering large language models (LLMs) and foundation models. This enabled a scale-up in model size unprecedented in deep learning. Transformers gave rise to large language models like GPT (Generative Pre-trained Transformer) and one chatbot application that took the world by storm: OpenAI's ChatGPT.
But what exactly are transformers and how do they work? In this post, I’ll unpack the transformer architecture at an intuitive level.

## A Picture is Worth a Thousand Words
Transformers introduced the novel concept of attention mechanisms for modeling sequences. While recurrent neural networks (RNNs) process sequences sequentially, transformers process the entire sequence at once in parallel.
The attention mechanism creates direct connections between all inputs in a sequence, allowing the model to focus on the most relevant parts as needed. This is akin to how we pay visual attention to certain regions in an image.

### Attention: Finding What Matters
At its core, attention is about focusing on what's relevant. Just like we pay visual attention to certain objects in a scene, attention in deep learning models lets them focus on pertinent parts of their input.

But how does attention actually work? The first step is to calculate similarity scores between a query and a set of key-value pairs. 

$$x_i = \alpha (q, k_i)$$

The query $q$ encapsulates what the model wants to focus on. The keys $k$ represent candidate relevant inputs. 
These similarity scores are calculated using functions like the scaled dot product. This determines how closely the query aligns with each key based on their vector representations. Keys $k_i$ with higher dot products have greater relevance to the query $q$.

## The Intuition Behind Self-Attention
Self-attention, the core innovation behind transformers, is conceptually simple. For each input, it calculates attention scores against all other inputs based on their relatedness. Highly related inputs get higher scores.
These scores determine a weighted sum representing the relevant context for each input. By focusing on the most important parts of the sequence, self-attention provides the right contextual representation to encode the inputs effectively.
In self-attention, the queries, keys and values all come from the same place - different positions of the input sequence. This allows relating different parts of the sequence to each other. For example, a token could attend to previous tokens that provide context for understanding it.
For some curious rechnical readers, let me put the code snippets for the attention function as follows. 

    # Attention implementation (Scaled Dot Product)
    def attention(query_vec, key_mat, value_mat, mask=None, dropout=None):
        # Calculate dot product between query and keys
        key_dim = query_vec.size(-1)
        relevance_scores = torch.matmul(query_vec, key_mat.transpose(-2, -1)) / math.sqrt(key_dim)
  
        # Apply optional mask 
        if mask is not None:
            relevance_scores = relevance_scores.masked_fill(mask == 0, -1e9)
  
        # Convert scores to probabilites  
        attention_weights = F.softmax(relevance_scores, dim=-1)

        # Apply optional dropout 
        if dropout is not None:
        attention_weights = dropout(attention_weights)

        # Calculate weighted average of values
        attention_output = torch.matmul(attention_weights, value_mat)

        return attention_output, attention_weights


### Output Scoring
Next, a softmax layer turns the scores into normalized probabilities. This assigns likelihood values to each input indicating how pertinent it is given the query. The probabilities are used to calculate a weighted sum of the values - placing more emphasis on relevant inputs.

$$y_1, \cdots, y_n = softmax (x_i, \cdots, x_n)$$


## Behind the Magic - A High-Level View
Fundamentally, the transformer consists of an encoder and a decoder. The encoder maps the input to a rich, contextual representation. The decoder then uses this representation to generate the output.
In NLP, the encoder operates on the input text while the decoder generates the translated or summarized text. For image processing, the encoder processes the input image and the decoder reconstructs or generates it.
The encoder itself has two components - a self-attention layer and a feedforward network. Self-attention identifies important context across the entire input. The feedforward network further refines this representation.
The decoder also has a cross-attention layer that focuses on relevant parts of the encoder output. This allows the decoder to produce the output while looking at the appropriate input context.

## Simpler Than It Looks!
While transformers enable complex modeling, the implementation involves simple, repeated blocks of self-attention, cross-attention and feedforward layers. This repetitive structure makes transformers easy to optimize and scale to huge datasets across diverse domains.

