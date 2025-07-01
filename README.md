<div align=center>
  <h1>
    NeRF: 3D Reconstruction from 2D Images
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs479-fall-2023/ target="_blank"><b>KAIST CS479: Machine Learning for 3D Data (Fall 2023)</b></a><br>
    Programming Assignment 2    
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://dvelopery0115.github.io target="_blank"><b>Seungwoo Yoo</b></a>  (dreamy1534 [at] kaist.ac.kr)      
  </p>
</div>

<div align=center>
  <img src="./media/nerf_blender/lego.gif" width="400" />
</div>

## Abstract

The introduction of [Neural Radiance Fields (NeRF)](https://arxiv.org/abs/2003.08934) was a massive milestone in image-based, neural rendering literature.
Compared with previous works on novel view synthesis, NeRF is a simple, yet powerful idea that combines recently emerging neural implicit representations with traditional volume rendering techniques.
As of today, the follow-up research aiming to scale and extend the idea to various tasks has become one of the most significant streams in the computer vision community thanks to its simplicity and versatility.

In this assignment, we will take a technical deep dive into NeRF to understand this ground-breaking approach which will help us navigate a broader landscape of the field.
We strongly recommend you check out the paper, together with [our brief summary](https://geometry-kaist.notion.site/Tutorial-2-NeRF-Neural-Radiance-Field-ef0c1f3446434162a540e6afc7aeccc8?pvs=4), before, or while working on this assignment.

## Code Structure
This codebase is organized as the following directory tree. We only list the core components for brevity:
```

media
│
├── ckpt             <- Directory containing nerf checkpoint
│
└── render
    ├── test_views             <- Directory containing test view rendering
    └── video                  <- Directory containing train view rendering



torch_nerf
│
├── configs             <- Directory containing config files
│
├── runners
│   ├── evaluate.py     <- Script for quantitative evaluation.
│   ├── render.py       <- Script for rendering (i.e., qualitative evaluation).
│   ├── train.py        <- Script for training.
│   └── utils.py        <- A collection of utilities used in the scripts above.
│
├── src
│   ├── cameras
│   │   ├── cameras.py
│   │   └── rays.py
│   │   
│   ├── network
│   │   └── nerf.py
│   │
│   ├── renderer
│   │   ├── integrators
│   │   ├── ray_samplers
│   │   └── volume_renderer.py
│   │
│   ├── scene
│   │
│   ├── signal_encoder
│   │   ├── positional_encoder.py
│   │   └── signal_encoder_base.py
│   │
│   └── utils
│       ├── data
│       │   ├── blender_dataset.py
│       │   └── load_blender.py
│       │
│       └── metrics
│           └── rgb_metrics.py
│
├── requirements.txt    <- Dependency configuration file.
└── README.md           <- This file.
```
