### MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts

#### Overview:

- MATHVISTA is introduced as a benchmark to evaluate mathematical reasoning abilities of large language models (LLMs) and large multimodal models (LMMs) in visually complex scenarios.
- It combines challenges from diverse mathematical and visual tasks and comprises 6,141 examples from existing datasets and three newly created ones.
- The study evaluates 12 prominent foundation models, with GPT-4V emerging as the best performer.

#### Data Creation:

- MATHVISTA dataset creation involved collecting existing MathQA datasets and reviewing VQA datasets for math-related instances.
- Three new datasets (IQTest, FunctionQA, PaperQA) were created to address gaps in logical reasoning, statistical reasoning, and scientific reasoning.
- An annotation process ensured data quality, with metadata annotation facilitating comprehensive analysis.

#### Experimental Setup:

- Models were evaluated under three setups: Text-Only LLMs, Augmented LLMs, and LMMs.
- valuation protocols included response generation, answer extraction, and score calculation.

#### Experimental Results:

- GPT-4V achieved the highest accuracy of 49.9%, outperforming other models.
- However, there's still a 10.4% gap between GPT-4V and human performance.
- Augmented LLMs showed superior performance compared to text-only LLMs, with GPT-4 employing program-of-thought prompting achieving 33.9% accuracy.
- Multimodal Bard achieved 34.8% accuracy, but fell short of human performance by 58%.

#### Fine-Grained Results

- GPT-4V outperformed other models across different tasks, mathematical reasoning abilities, visual context types, and grade levels.
- However, it showed shortcomings in logical reasoning and numeric common sense tasks.

#### Qualitative Analysis:

- Analysis of Multimodal Bard's predictions highlighted modes of success and failure, with hallucination identified as a major error source.
- Augmented GPT-4 examples demonstrated successful reasoning with accurate OCR text, but also showed failure cases due to hallucination caused by external visual models.




