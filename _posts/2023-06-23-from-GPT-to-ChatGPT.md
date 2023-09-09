# The Path to More Human-Like AI: Charting the Progression from Transformers to ChatGPT and the Cutting Edge of Large Language Modeling 

## When It All Began
First introduced in 2017 by Google Researchers, the transformer architecture that utilized the transformer block consisting solely of attention mechanisms represented a departure from the convolutional and recurrent layers used in prior state-of-the-art models, 
enabling modeling of longer-range dependencies in sequence data and greater parallelizability, achieving superior performance on translation tasks over models reliant on convolution or recurrence. 
This demonstrated the promise of attention-based networks for advancing language modeling and understanding tasks.

## The Birth of GPT
Building upon the capabilities of the transformer architecture, OpenAI introduced the Generative Pre-trained Transformer (GPT) in 2018 which pioneered the pre-trained language model approach of first pre-training a model on massive unlabeled corpora in an unsupervised manner to create a general language representation before fine-tuning on downstream tasks. 
This transfer learning methodology outperformed training on the end tasks from scratch, underscoring the value of pre-training on large unlabeled data. 
Successive iteratively larger versions of GPT like GPT-2, GPT-3, and the recently announced GPT-4 have progressively increased the number of parameters and scale of data used for pre-training, with GPT-4 expected to utilize over 100 trillion parameters, far exceeding the already enormous 175 billion parameters in GPT-3. The massive scale of data and model size has enabled remarkable generative capabilities even in a zero-shot setting without any task-specific fine-tuning.

## The Math of GPT
Mathematically, GPT relies on the transformer decoder architecture and is trained to model the probability $p(u)$ of a sequence of tokens $u = (u_1, ..., u_T)$ using the chain rule to decompose the joint probability into conditional probabilities:

$p(u) = Π_p^{T=1} p(u_t | u_1, ..., u_{t-1})$ 

where the context vector $h_i$ for predicting next token ui is derived by applying self-attention on the prior subsequence $u_1$ to $u_{t-1}$.

## The Birth of ChatGPT
While earlier GPT versions were optimized for text generation capabilities, ChatGPT specialized in more natural conversational abilities. OpenAI trained ChatGPT on a large dataset of dialog conversations generated through human demonstrators interacting in conversation. A key innovation was the use of reinforcement learning from human feedback (RLHF) to train the model to converse responsively. 
In RLHF, the model is rewarded for responding appropriately to conversation context, admitting ignorance rather than guessing, and refusing inappropriate requests. This reinforcement signal from human evaluators provides feedback to enhance the model's conversational abilities.

![](/images/chatgpt_process.png)
Image source: https://www.linkedin.com/pulse/unleashing-power-chat-gpt-beginners-guide-manoz-acharya/

Architecturally, ChatGPT leverages a decoder-only transformer akin to GPT-3 to model the conditional probability over token sequences. The distinguishing innovations are in the training objective and training data. Optimization for dialog conversation instead of monologue text generation powers ChatGPT's nuanced conversational skills.
The training dataset of genuine dialog exchanges creates data better suited for chatbot-style interaction, rather than the monologue-formatted text GPT was trained on. Modeling genuine dialog context enables ChatGPT's capabilities in follow-up coherence, consistency, and overall conversational flow. 

![](/images/2022-Alan-D-Thompson-ChatGPT-Sparrow-Rev-0d.png)
Image source: [https://www.linkedin.com/pulse/unleashing-power-chat-gpt-beginners-guide-manoz-acharya/](https://lifearchitect.ai/chatgpt/)

ChatGPT's launch in November 2022 sparked excitement over its human-like conversational capabilities and also raised many ethical concerns around potential for harm, factual accuracy, and transparency about its true capabilities necessitating thorough discussion.
The rapid progress from transformers to ChatGPT over 5 years illustrates the potential of combining architectures like transformers with techniques like pre-training, reinforcement learning, and conversational training data. While modeling wide knowledge and reasoning in conversation remains challenging, the pace of progress promises more impressive capabilities emerging from continued research and compute scale gains as evidenced by the pending launch of GPT-4.
