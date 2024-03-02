# Image-Captioning-using-Faiss-Indices

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Goal](#goal)
  * [Technical Aspect](#technical-aspect)
  * [Technologies Used](#technologies-used)
  * [Results](#results)

## Demo
Example 1:  
![image](https://github.com/Sagnick0907/Image-Captioning-using-Faiss-Indices/assets/76872499/b8a9a61b-1da9-4e7f-bdcb-1748e4de8b24)  
**Ground Truth captions:**  ['This wire metal rack holds several pairs of shoes and sandals', 'A dog sleeping on a show rack in the shoes.', 'Various slides and other footwear rest in a metal basket outdoors.', 'A small dog is curled up on top of the shoes', 'a shoe rack with some shoes and a dog sleeping on them']  
**Predicted caption:**  The two dogs are sitting on a chair.  

Example 2:  
![image](https://github.com/Sagnick0907/Image-Captioning-using-Faiss-Indices/assets/76872499/88d3f583-e6ca-404d-ac82-621fa46c46ff)  
**Ground Truth captions:**  ['A loft bed with a dresser underneath it.', 'A bed and desk in a small room.', 'Wooden bed on top of a white dresser.', 'A bed sits on top of a dresser and a desk.', 'Bunk bed with a narrow shelf sitting underneath it. ']  
**Predicted caption:**  A bedroom with a bed, desk and entertainment console.  

![image](https://github.com/Sagnick0907/Image-Captioning-using-Faiss-Indices/assets/76872499/08f55000-65ca-4f67-afd9-f8e5f892ee29)  
**Ground Truth captions:**  ['A motorcycle parked in a parking space next to another motorcycle.', 'An old motorcycle parked beside other motorcycles with a brown leather seat.', 'Motorcycle parked in the parking lot of asphalt.', 'A close up view of a motorized bicycle, sitting in a rack. ', 'The back tire of an old style motorcycle is resting in a metal stand. ']  
**Predicted caption:**  a motorcycle that is parked in side a buliding  


## Overview  
Although VLMs (Vision Language Models) are the go to tools for image captioning right now, there are interesting works from earlier years that used KNN for captioning and perform surprisingly well enough!

Further, Libraries like [Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) can be used to perform the nearest neighbor computation efficiently and are used in many industrial applications. In this project we implemented an algorithm to perform Image captioning using KNN based on the paper [A Distributed Representation Based Query Expansion Approach for
Image Captioning](https://aclanthology.org/P15-2018.pdf)  

## Motivation and Goal  
The motivation of the project is to explore an alternative method for image captioning using the k-nearest neighbors (KNN) algorithm, particularly leveraging the Faiss library for efficient similarity search. The goal includes implementing and evaluating the KNN-based algorithm, experimenting with different values for k, optimizing the algorithm's performance using Faiss, and conducting a qualitative study by visualizing images and predicted captions. The project aims to assess the viability of the KNN approach in comparison to other methods like Vision Language Models (VLMs).  

## Technical Aspect  
- Dataset: [MS COCO](https://cocodataset.org/#home) 2014 (val set only)  

Algorithm:  
1. Given: Image embeddings and correspond caption embeddings (5 Per image)
2. For every image, findout the k nearest images and compute its query vector as the weighted sum of the captions of the nearest images (k*5 captions per image)
3. The predicted caption would be the caption in the dataset that is closest to the query vector. (for the sake of the assignment use the same coco val set captions as the dataset)

Tasks:
1. Implemented the algorithm and compute the bleu score. Use Faiss for nearest neighbor computation.
2. Tried for a few options for k. Recorded the observations.  
3. For a fixed k, tried a few options in the Faiss index factory to speed the computation in step 2. Recorded the observations.
4. Qualitative study: Visualize five images, their ground truth captions and the predicted caption.

## Technologies Used
- Google Colab  
- Concepts used : Faiss Indices (IndexFlatIP, IndexFlatL2, IndexIVFFlat, IndexHNSWFlat), cosine similarity.    
- Libraries: torchvision.datasets, torchvision.transforms, DataLoader, torch, torch.nn, torch.nn.functional, bleu_score, faiss, np, matplotlib.pyplot

## Results
|    Faiss Index   |    Bleu Score       |     k       |
|:-------------:|:-------------:|:-------------:|
|   IndexFlatIP   |  0.0723923250317784   |  5   |
|   IndexFlatL2   |   0.07252093419068131   |  5  |
|   IndexIVFFlat   |   0.07080728032053231   |  5  |
|   IndexHNSWFlat   |   0.07043850246759856   |  5  |

