# MobileMEF
Code related to the paper "[MobileMEF: Fast and Efficient Method for Multi-Exposure Fusion](https://arxiv.org/abs/2408.07932)"

## Preview of MobileMEF

<p float="center">
  <img src="assets/model.PNG" width="50%" />
  <img src="assets/overview.PNG" width="48%" />
</p>

## Usage

We recommend using Conda as package manager.

```
conda env create -f environment.yml
```

The ```model.py``` file provides tools for inference and converting the model to TFLITE or ONNX format.

The ```h5/``` folder provides checkpoints for the trained models using EVs 1 and -1 (```sice_ev1.h5```), and most under and over exposed frames (```sice_ev_most.h5```).

The ```data/``` folder provides examples of images for the input pipeline.

The ```utils/``` folder comprises auxiliary code for metrics evaluation and benchmarking with ONNX model format.

## Visual Results

<p align="center">
  <img src="assets/results.png" width="50%">
</p>

## Citation

If this work has been helpful to you, we would appreciate it if you could cite our paper! 

```
@misc{kirsten2024mobilemeffastefficientmethod,
      title={MobileMEF: Fast and Efficient Method for Multi-Exposure Fusion}, 
      author={Lucas Nedel Kirsten and Zhicheng Fu and Nikhil Ambha Madhusudhana},
      year={2024},
      eprint={2408.07932},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2408.07932}, 
}
```
