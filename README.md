# Midjourney Prompt–Embedding Dataset

This repository contains the dataset associated with our COLM 2024 paper:

**Iteratively Prompting Multimodal LLMs to Reproduce Natural and AI-Generated Images**  
Ali Naseh, Katherine Thai, Mohit Iyyer, Amir Houmansadr  

This dataset is derived from our study of whether multimodal language models can infer prompts that generate images visually similar to target images produced by text-to-image systems or found in stock image collections. By leveraging large-scale prompt–image data and iterative refinement, our work demonstrates that comparable images can be reproduced at a fraction of the original cost, highlighting emerging economic and security implications for digital imagery ecosystems.

To support further research on prompt–image relationships and multimodal representations, we release a large-scale collection of anonymized prompts and corresponding image embeddings. Raw images are not redistributed due to platform terms of service, copyright considerations, and potential misuse risks.

📄 Paper: [PDF](https://people.cs.umass.edu/~amir/papers/2024-COLM-prompt.pdf) | [arXiv](https://arxiv.org/pdf/2404.13784)


## Overview

This dataset is constructed from a large-scale collection of prompt–image pairs gathered from publicly accessible Midjourney Discord channels. From this collection, we release a processed subset of approximately 4.5 million samples. Each sample corresponds to a user-submitted prompt and its associated generated image, reflecting real-world usage of text-to-image systems. The released dataset includes anonymized prompts and corresponding image embeddings derived from the generated images. The data has been preprocessed to remove user-identifying information and to ensure consistency, including filtering prompts based on length and content. The dataset spans multiple Midjourney model versions and reflects a wide range of prompt styles, modifiers, and generation behaviors observed in practice.


## Dataset Contents

Each sample in the dataset includes the following fields:

- **Prompt**: Anonymized user-submitted text prompt used to generate the image. Prompts may include Midjourney-specific parameters (e.g., arguments following `--`) that control aspects of generation (see [Midjourney Parameter List](https://docs.midjourney.com/hc/en-us/articles/32859204029709-Parameter-List)).
- **Midjourney Version**: The version of the Midjourney model used to generate the image.
- **Embedding Mapping**: A reference to the corresponding image embedding stored in chunked files, specified by:
  - `embedding_file`: the file containing the embedding  
  - `embedding_index`: the row index within that file  

Image embeddings are computed using the CLIP model (`laion/CLIP-ViT-g-14-laion2B-s12B-b42K`) and stored separately in chunked files (each containing up to 200,000 embeddings).


## Data Collection

The dataset is constructed from prompt–image pairs collected from publicly accessible Midjourney Discord channels, where users interact with the Midjourney bot to generate images. Each prompt submitted by a user is posted alongside the corresponding generated images, enabling the collection of large-scale prompt–image data from real-world usage.

We collect data from multiple general-purpose channels to capture a broad range of prompts and generation behaviors. The original dataset consists of tens of millions of samples, from which we derive a processed subset for release. All samples are preprocessed to extract the original prompts, remove user-identifying information, and filter out invalid or noisy entries to ensure data quality and consistency.

## Data Preprocessing

The collected data is processed to produce clean and consistent prompt–image pairs suitable for analysis and release. We first extract the original user prompts from raw message content, which may include additional fields such as user identifiers, timestamps, image URLs, and generation parameters. We then remove empty entries and rows with missing or malformed data, and exclude prompts that contain external image URLs to avoid dependency on external content. To improve data quality and consistency, we filter out prompts containing non-English text or excessive noise (e.g., emojis).

We further retain only prompts with tokenized length ≤ 77, as many text encoders used in multimodal models (e.g., CLIP) operate under fixed context length constraints, and longer prompts would otherwise be truncated. The Midjourney model version associated with each sample is inferred using timestamps and known release periods of different versions. Finally, all user-identifying and platform-specific metadata, including user IDs, timestamps, and image URLs, are removed to ensure anonymization in the released dataset. The preprocessing scripts used to perform these steps are included in this repository.

## Data Access

The dataset and embedding files are hosted on Hugging Face:

👉 [Midjourney Prompt–Embedding Dataset](https://huggingface.co/datasets/AliN96/midjourney-prompts-embeddings)

- The main dataset is provided as a Parquet file (`train.parquet`).
- Image embeddings are stored separately as shard files (`.npy`).

Please refer to the Hugging Face page for instructions on downloading and using the dataset.

## Citation

If you use this dataset, please cite our paper:

```bibtex
@inproceedings{
naseh2024iteratively,
title={Iteratively Prompting Multimodal {LLM}s to Reproduce Natural and {AI}-Generated Images},
author={Ali Naseh and Katherine Thai and Mohit Iyyer and Amir Houmansadr},
booktitle={First Conference on Language Modeling},
year={2024},
url={https://openreview.net/forum?id=SwUsFTtM9h}
}
