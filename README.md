# A Hitchhikers Guide to Fine-Grained Face Forgery Detection Using Common Sense Reasoning

Authors official PyTorch implementation of the **[A Hitchhiker's Guide to Fine-Grained Face Forgery Detection Using Common Sense Reasoning](https://arxiv.org/abs/2410.00485)**. If you use this code for your research, please [**cite**](#citation) our paper.

> **A Hitchhikers Guide to Fine-Grained Face Forgery Detection Using Common Sense Reasoning**<br>
> Niki Maria Foteinopoulou, Enjie Ghorbel and Djamila Aouada<br>
> <br>
> ![summary](figs/method.svg)
>
> **Abstract**: Explainability in artificial intelligence is crucial for restoring trust, particularly in areas like face forgery detection, where viewers often struggle to distinguish between real and fabricated content. Vision and Large Language Models (VLLM) bridge computer vision and natural language, offering numerous applications driven by strong common-sense reasoning. Despite their success in various tasks, the potential of vision and language remains underexplored in face forgery detection, where they hold promise for enhancing explainability by leveraging the intrinsic reasoning capabilities of language to analyse fine-grained manipulation areas.
    For that reason, few works have recently started to frame the problem of deepfake detection as a Visual Question Answering (VQA) task, nevertheless omitting the realistic and informative open-ended multi-label setting. With the rapid advances in the field of VLLM, an exponential rise of investigations in that direction is expected.
    As such, there is a need for a clear experimental methodology that converts face forgery detection to a Visual Question Answering (VQA) task to systematically and fairly evaluate different VLLM architectures. Previous evaluation studies in deepfake detection have mostly focused on the simpler binary task, overlooking evaluation protocols for multi-label fine-grained detection and text-generative models. We propose a multi-staged approach that diverges from the traditional binary evaluation protocol and conducts a comprehensive evaluation study to compare the capabilities of several VLLMs in this context.
    In the first stage, we assess the models' performance on the binary task and their sensitivity to given instructions using several prompts. In the second stage, we delve deeper into fine-grained detection by identifying areas of manipulation in a multiple-choice VQA setting. In the third stage, we convert the fine-grained detection to an open-ended question and compare several matching strategies for the multi-label classification task. Finally, we qualitatively evaluate the fine-grained responses of the VLLMs included in the benchmark.
    We apply our benchmark to several popular models, providing a detailed comparison of binary, multiple-choice, and open-ended VQA evaluation across seven datasets.


## Overview

<p alighn="center">
 In a nutshell, the main novelty of this benchmark compared to previous works is threefold: 1) it converts the multi-label classification task of face forgery detection to a VQA task so that VLLM's common sense reasoning capabilities can be evaluated, 2) it systematically and consistently assesses VLLM capabilities on nine binary and three fine-grained benchmarks and 3) is offering an open source and extendable framework for future zero-shot or task-specific VLLMs, that ensures a fair comparison.
</p>


## Installation

We recommend installing the required packages using python's native virtual environment as follows:

```bash
$ python -m venv venv
$ source venv/bin/activate
(venv) $ pip install --upgrade pip
(venv) $ pip install -r requirements.txt
```

For using the aforementioned virtual environment in a Jupyter Notebook, you need to manually add the kernel as follows:

```bash
(venv) $ python -m ipykernel install --user --name=venv
```


## Acknowledgements

This work is supported by the Luxembourg National Research Fund, under the BRIDGES2021/IS/16353350/FaKeDeTeR, and by POST Luxembourg. 
Experiments were performed on the Luxembourg national supercomputer MeluXina.
The authors gratefully acknowledge the LuxProvide teams for their expert support.

## Citation

```bibtex
@misc{foteinopoulou2024hitchhikersguidefinegrainedface,
      title={A Hitchhikers Guide to Fine-Grained Face Forgery Detection Using Common Sense Reasoning}, 
      author={Niki Maria Foteinopoulou and Enjie Ghorbel and Djamila Aouada},
      year={2024},
      eprint={2410.00485},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.00485}, 
}
