# 🎨 Multimodal LLMs Can Reason about Aesthetics in Zero-Shot | [Arxiv](https://arxiv.org/abs/2501.09012)
TL;DR :fire::fire: : Can MLLMs evaluate the aesthetics of artworks with inference-time reasoning? If so, how? We analyze these issues in this paper, revealing the challenges and offer a strong baseline technique that boost the MLLMs' alignment with human preference.

---
Abstract: We present the first study on how Multimodal LLMs' (MLLMs) reasoning ability shall be elicited to evaluate the aesthetics of artworks. To facilitate this investigation, we construct MM-StyleBench, a novel high-quality dataset for benchmarking artistic stylization. We then develop a principled method for human preference modeling and perform a systematic correlation analysis between MLLMs' responses and human preference. Our experiments reveal an inherent hallucination issue of MLLMs in art evaluation,  associated with response subjectivity. ArtCoT is proposed, demonstrating that art-specific task decomposition and the use of concrete language boost MLLMs' reasoning ability for aesthetics. Our findings offer valuable insights into a wide range of downstream applications, such as style transfer and artistic image generation. 



## Dataset 

**The MM-StyleBench dataset will be available soon.**

![fig_dataset](asset/fig_dataset.jpg)

--- 


## ArtCoT

We propose ArtCoT to enhance the inference-time reasoning capability of MLLMs. A example conversation is provided below. Detailed quantitative comparison can be found in [paper](https://arxiv.org/abs/2501.09012). The full response from MLLMs in our experiments will also be released to facilitate further research.


![fig_example_style](asset/fig_example_style.jpg)
