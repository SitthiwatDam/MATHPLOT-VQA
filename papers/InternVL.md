### InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks

#### Overview:

- The paper introduced InternVL, a large-scale vision-language foundation model. 
- The model architecture is different from prior VLLMs allowing the vision encoder and middleware to have flexible combinations for both contrastive and generative tasks.

#### Results:

- InternViT-6B, a large-scale vision foundation model, can align with LLM-initialized language middleware such as QLLaMA and help leverage image-text data from various sources for efficient training. 
- According to evaluation from this paper, the performance of InternVL-Chat surpassed LLaVA-1.5.

#### Key Takeaway:

- Similar to LLaVA and many VQA systems, InternVL uses a large multimodal model (LMM) method. However, instead of using a vision encoder to feed text into the language model, InternVL proposed using contrastive learning between the vision encoder and language model generative learning. This improves the model capabilities on some visual-language tasks.



