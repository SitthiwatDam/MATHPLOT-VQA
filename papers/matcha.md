### MATCHA: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering
![Matcha](./figures/Matcha.png)
Figure1: MATCHA defines two types of pretraining tasks: (1) chart derendering (light blue boxes) and (2) mathematical reasoning (light red boxes). In chart derendering, given a chart, the model needs to decode its underlying
rendering code or data table. In math reasoning, given a math question rendered as an image, the model needs to
decode its answer. Chart derendering teaches the models layout understanding (including number extraction and
their organizations) and math reasoning teaches the models numerical reasoning capabilities.

#### Definition:
MATCHA (Math Reasoning and Chart Derendering Pretraining) enhances visual language models' capabilities by integrating math reasoning and chart derendering tasks. Beginning with Pix2Struct as the base model, MATCHA further pretrains it with chart derendering, math reasoning, and screenshot parsing techniques.

- **Chart Derendering:** Identifies visual patterns in images, effectively parsing and grouping them to extract key information.
- **Math Reasoning:** Explicitly injects numerical reasoning knowledge into the image-to-text model by learning from textual math data.
- **Screenshot Parsing:** Continuously applies screenshot parsing pretraining from Pix2Struct to prevent catastrophic forgetting.

#### Datasets:
The pretraining mixture comprises 40% math reasoning, 40% chart derendering, and 20% screenshot parsing tasks, each with different rates. Chart derendering sources include internally synthesized chart-table pairs, ChartQA, PlotQA synthesis, and chart-to-code datasets. Math reasoning tasks are sourced from MATH and DROP datasets, offering categorized questions and standard QA format, respectively.

The mixture is evaluated on datasets covering ChartQA, PlotQA, and Chart-to-Text summarization, along with Pix2Struct tasks involving documents, user interfaces, and natural images, ensuring comprehensive evaluation across diverse multimodal English QA and generation tasks.

#### Results:
**Main Results (ChartQA, PlotQA, Chart-to-Text Summarization):** MATCHA demonstrates superior performance across all setups and tasks, outperforming Pix2Struct by approximately 10% on average, even without access to gold tables.

**Results on Pix2Struct Tasks:** Rerunning Pix2Struct experiments with MATCHA yields an average 2.3% improvement across tasks, showcasing its transferability to diverse visual language domains.

#### Analysis:
Fine-grained analysis of ChartQA reveals challenges in data extraction, math reasoning, and plot attributes. MATCHA excels in data extraction and math reasoning but struggles with complex math problems. Error analysis highlights issues in math reasoning, data extraction, and plot attributes.

#### Limitations:
- MATCHA faces challenges in complex math problems.
- The method of math calculations needs further study.
- Plot attribute performance lags behind PaLI, possibly due to lacking specific types of training.
- Expensive experimental setup with limited runs.
- Increased test runs could enhance paper quality.
- The study focuses on a small segment of the vast visual language domain.

### Tables:
| Component         | Task/Dataset               | Rate | Size  |
|-------------------|----------------------------|------|-------|
| Math reasoning    | MATH dataset               | 20%  | 2M    |
|                   | DROP                       | 20%  | 96K   |
| Chart derendering | Chart-to-code (GitHub; ours) | 4% | 23M   |
|                   | Chart-to-table (synthetic; ours) | 12% | 270K |
|                   | Chart-to-table (ChartQA)  | 12%  | 22K   |
|                   | Chart-to-table (PlotQA)   | 12%  | 224K  |
| Pix2Struct        | Screenshot parsing         | 20%  | 80M   |

Table 1: Mixture rates for all tasks in pretraining and the absolute size of each dataset. The mixture rate is used to sample each example within the batch.

| Task              | Dataset             | # Tables | # Pairs |
|-------------------|---------------------|----------|---------|
| Chart Question    | Answering           |          |         |
|                   | ChartQA (Human)     | 4.8K     | 9.6K    |
|                   | ChartQA (Machine)   | 17.1K    | 23.1K   |
|                   | PlotQA (v1)         | 224K     | 8M      |
|                   | PlotQA (v2)         | 224K     | 29M     |
| Chart Summarization| Chart-to-Text (Pew)| 9K       | 9K      |
|                   | Chart-to-Text (Statista)| 35K   | 35K     |

Table 2: Statistics of the finetuning datasets.
| Model              | Gold Table? | ChartQA     |           |             | PlotQA    |               |           |Chart-to-Text |    |         | Average (all) |
|--------------------|-------------|-------------|-----------|-------------|-----------|---------------|-----------|---------|---------|---------|----------------|   
|                    |             | aug.        | human     | avg.        | v1        | v2            | avg.      | Pew     |Statista | avg.    |                |
| T5                 |yes          | -           | -         | 59.8        | 93.2      | 85.6          | 89.4      | -       |37.0     |     -   |  -             |
| VL-T5              |yes          | -           | -         | 59.1        | **96.4**  | 84.7          | 90.6      | -       |-        |     -   |  -             |
| VisionTaPas        |yes          | -           | -         | 61.8        | 80.2      | 58.3          | 69.3      | -       |-        |     -   |  -             |
| CRCT               |no           | -           | -         | -           | 76.9         |  34.4            |  55.7         |  -      |  -      |     -   |    -           |
| VL-T5-OCR          |no           | -           | -         | 41.6           |  75.9         |   56.0            | 66.0         |  -      |  -      |     -   |    -           |
| T5-OCR             |no           | -           | -         | 41.0          |  72.6         |  56.2            | 64.4         |   10.5      |  35.3      |      22.9   | 42.8  |
| VisionTaPas-OCR    |no           | -           | -         |  45.5           | 65.3         |   42.5            |53.9         |  -      |  -      |     -   |    -           |
| PaLI-17B (res. 224)|no           | 11.2           |  15.2         | 13.2           | 56.9         |   13.1| 35.0 |10.0| 40.2| 25.1 |24.4            |
| PaLI-17B (res. 588)|no           | 64.9 |30.4 |47.6 |64.5 |15.2| 39.8 |11.2 |<font color="Red">**41.4**</font>|<font color="BLUE">**26.3**</font>| 37.9 |
| Pix2Struct         |no           |  81.6 |30.5| 56.0 |73.2 |71.9 |72.5| 10.3| 38.0| 24.2 |50.9|
| MATCHA             |no           |<font color="BLUE">**90.2**</font> |<font color="BLUE">**38.2**</font>| <font color="BLUE">**64.2**</font> |92.3 |<font color="BLUE">**90.7**</font>| <font color="BLUE">**91.5**</font>| <font color="BLUE">**12.2**</font>| 39.4 |25.8 |<font color="BLUE">**60.5**</font>|


Table 3: Main experimental results on ChartQA, PlotQA, and Chart-to-Text benchmarks. 

| Task         | ChartQA | AI2D | OCR-VQA | RefExp | WidgetCap | Screen2Words | TextCaps | DocVQA | InfoVQA | Average | Average (excl. ChartQA) |
|--------------|---------|------|---------|--------|-----------|--------------|----------|--------|---------|---------|------------------------|
| Pix2Struct   | 56.0    | 40.9 | 69.4    | 92.2   | 133.1     | 107.0        | 88.0     | 72.1   | 38.2    | 77.4    | 80.1                   |
| MATCHA       | 64.2    | 42.6 | 68.9    | 94.2   | 137.7     | 106.2        | 92.4     | 74.2   | 37.2    | 79.7    | 81.7                   |

Table 4: MATCHA vs. Pix2Struct on Pix2Struct tasks



