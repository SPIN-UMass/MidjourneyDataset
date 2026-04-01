# Midjourney Prompt–Embedding Dataset

This repository contains the dataset associated with our COLM 2024 paper:

**Iteratively Prompting Multimodal LLMs to Reproduce Natural and AI-Generated Images**  
Ali Naseh, Katherine Thai, Mohit Iyyer, Amir Houmansadr  

This dataset is derived from our study of whether multimodal language models can infer prompts that generate images visually similar to target images produced by text-to-image systems or found in stock image collections. By leveraging large-scale prompt–image data and iterative refinement, our work demonstrates that comparable images can be reproduced at a fraction of the original cost, highlighting emerging economic and security implications for digital imagery ecosystems.

To support further research on prompt–image relationships and multimodal representations, we release a large-scale collection of anonymized prompts and corresponding image embeddings. Raw images are not redistributed due to platform terms of service, copyright considerations, and potential misuse risks.

📄 Paper: [PDF](https://people.cs.umass.edu/~amir/papers/2024-COLM-prompt.pdf) | [arXiv](https://arxiv.org/pdf/2404.13784)


## Overview

This dataset is constructed from a large-scale collection of prompt–image pairs gathered from publicly accessible Midjourney Discord channels. From this collection, we release a processed subset of approximately 4.5 million samples. Each sample corresponds to a user-submitted prompt and its associated generated image, reflecting real-world usage of text-to-image systems.

The released dataset includes anonymized prompts and corresponding image embeddings derived from the generated images. The data has been preprocessed to remove user-identifying information and to ensure consistency, including filtering prompts based on length and content.

The dataset spans multiple Midjourney model versions and reflects a wide range of prompt styles, modifiers, and generation behaviors observed in practice.
