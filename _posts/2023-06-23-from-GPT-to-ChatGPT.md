# The Path to More Human-Like AI: Charting the Progression from Transformers to ChatGPT and the Cutting Edge of Large Language Modeling 

## When It All Began
First introduced in 2017 by Google Researchers, the transformer architecture that utilized the transformer block consisting solely of attention mechanisms represented a departure from the convolutional and recurrent layers used in prior state-of-the-art models, 
enabling modeling of longer-range dependencies in sequence data and greater parallelizability, achieving superior performance on translation tasks over models reliant on convolution or recurrence. 
This demonstrated the promise of attention-based networks for advancing language modeling and understanding tasks.

## The Birth of GPT
Building upon the capabilities of the transformer architecture, OpenAI introduced the Generative Pre-trained Transformer (GPT) in 2018 which pioneered the pre-trained language model approach of first pre-training a model on massive unlabeled corpora in an unsupervised manner to create a general language representation before fine-tuning on downstream tasks. 
This transfer learning methodology outperformed training on the end tasks from scratch, underscoring the value of pre-training on large unlabeled data. 

The term "large language model" (LLM) became popular in recent years as new models with billions of parameters were developed that achieved strong performance on language tasks. The trend towards larger models started in 2018 with the release of BERT by Google AI. BERT had 110 million parameters, orders of magnitude more than previous popular models like word2vec and LSTM networks. BERT showed that pretraining language models on large unlabeled corpora led to big gains in many NLP tasks. This catalyzed further research into scaling up model size. In 2019, OpenAI released GPT-2 with 1.5 billion parameters, showing further gains from increased scale. GPT-2 impressed many with its ability to generate coherent long-form text. In 2020, GPT-3 was released by OpenAI with 175 billion parameters! This firmly established the "large language model" as a key approach, with GPT-3 showing new capabilities like few-shot learning.

Successive iteratively larger versions of GPT like GPT-2, GPT-3, and the recently announced GPT-4 have progressively increased the number of parameters and scale of data used for pre-training, with GPT-4 expected to utilize over than 1 trillion parameters, far exceeding the already enormous 175 billion parameters in GPT-3. The massive scale of data and model size has enabled remarkable generative capabilities even in a zero-shot setting without any task-specific fine-tuning.

## The Math of GPT
In the GPT architecture, each token $u_i$ in the sequence $u = (u_1, ..., u_T)$ is represented by a context vector $h_i$.  Mathematically, GPT relies on the transformer decoder architecture and is trained to model the probability $p(u)$ of a sequence of tokens $u$ using the chain rule to decompose the joint probability into conditional probabilities:

$$p(u) = Π_p^{T=1} p(u_t | u_1, ..., u_{t-1})$$ 

where the context vector $h_i$ for predicting next token $u_i$ is derived by applying self-attention on the prior subsequence $u_1$ to $u_{t-1}$.

This context vector encodes the relevant contextual information from the previous tokens $(u_1 to u_{i-1})$ that can help predict the next token ui.
The context vector $h_i$ is obtained by applying multi-headed self-attention on the previous token embeddings. Specifically, the self-attention layer takes as input the sequence of token embeddings $(e_1, ..., e_{i-1})$ up to position $i-1$. It draws global dependencies between these token embeddings to create a new contextualized representation for each token.
Mathematically, a self-attention head computes attention scores between each pair of tokens $(e_j, e_k)$ using:

$$\text{Attention}(e_j, e_k) = (W_Qe_j)^T(W_Ke_k)$$
where $W_Q$ and $W_K$ are projection matrices that transform the embeddings into "query" and "key" vectors respectively.
These attention scores are normalized into a probability distribution using softmax. The attention distribution is used to compute a weighted average of the "value" vectors $Ve_k$, giving us the final context-aware token representation $h_j$ for token $j$.
Multiple such self-attention heads are used, whose outputs are concatenated to obtain the final context vector $h_i$ that incorporates contextual information from previous tokens.
The $W$ matrices refer to the trainable weight matrices used to transform the token embeddings in the self-attention calculations. For example, $W_Q$ projects embeddings into "query" vectors, $W_K$ into "key" vectors, and $W_V$ into "value" vectors. 

## The Birth of ChatGPT
While earlier GPT versions were optimized for text generation capabilities, ChatGPT specialized in more natural conversational abilities. OpenAI trained ChatGPT on a large dataset of dialog conversations generated through human demonstrators interacting in conversation. A key innovation was the use of reinforcement learning from human feedback (RLHF) to train the model to converse responsively. 
In RLHF, the model is rewarded for responding appropriately to conversation context, admitting ignorance rather than guessing, and refusing inappropriate requests. This reinforcement signal from human evaluators provides feedback to enhance the model's conversational abilities. One of key components of RLHF is Proximal Policy Optimization (PPO).

The goal of PPO is to maximize the expected return $J(\theta)$ of a stochastic policy $\pi_\theta(a_t|s_t)$ over all timesteps $t$, where $\theta$ are the policy parameters, $s_t$ is the state and $a_t$ is the action.
The PPO objective function contains three main terms:
1. Clipped Surrogate Loss
2. Value Function Loss
3. Entropy Bonus

Adding entropy bonus encourages exploration and prevents premature convergence.
The overall PPO loss function is:

$$L_{PPO}(\theta) = L_{CLIP}(\theta) - c_1 \cdot L_{VF}(\theta) + c_2 \cdot L_H(\theta)$$

Where $c_1, c_2$ are coefficients to balance the terms. $\theta$ is updated via stochastic gradient ascent on $L_{PPO}$.
So in summary, PPO uses clipped surrogate objective, value function prediction, and entropy regularization to achieve stable and sample efficient policy optimization for large policies like ChatGPT. 

![](/images/chatgpt_process.png)
Image source: https://www.linkedin.com/pulse/unleashing-power-chat-gpt-beginners-guide-manoz-acharya/


Architecturally, ChatGPT leverages a decoder-only transformer similar to GPT-3 to model the conditional probability over token sequences. The distinguishing innovations are in the training objective and training data. Optimization for dialog conversation instead of monologue text generation powers ChatGPT's nuanced conversational skills.
The training dataset of genuine dialog exchanges creates data better suited for chatbot-style interaction, rather than the monologue-formatted text GPT was trained on. Modeling genuine dialog context enables ChatGPT's capabilities in follow-up coherence, consistency, and overall conversational flow. 

![](/images/2022-Alan-D-Thompson-ChatGPT-Sparrow-Rev-0d.png)
Image source: [https://www.linkedin.com/pulse/unleashing-power-chat-gpt-beginners-guide-manoz-acharya/](https://lifearchitect.ai/chatgpt/)

ChatGPT's launch in November 2022 sparked excitement over its human-like conversational capabilities and also raised many ethical concerns around potential for harm, factual accuracy, and transparency about its true capabilities necessitating thorough discussion.
The rapid progress from transformers to ChatGPT over 5 years illustrates the potential of combining architectures like transformers with techniques like pre-training, reinforcement learning, and conversational training data. While modeling wide knowledge and reasoning in conversation remains challenging, the pace of progress promises more impressive capabilities emerging from continued research and compute scale gains as evidenced by the launch of GPT-4 and other LLMs.
