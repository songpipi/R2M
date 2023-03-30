# Recurrent Relational Memory Network for Unsupervised Image Captioning



![alt text](./image/framework.png)
<p align="center">The overall framework of R2M.</p>



This is a Tensorflow implementation for [Recurrent Relational Memory Network for Unsupervised Image Captioning, IJCAI2020](https://arxiv.org/abs/2006.13611).

If you use this code in your research, please consider citing:

```text
@InProceedings{Guo_2020_IJCAI,
author = {Guo, Dan and Wang, Yang and Song, Peipei and Wang, Meng},
title = {Recurrent relational memory network for unsupervised image captioning},
booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
month = {January},
year = {2021}
}
```

Requirements
----------------------
* Ubuntu 16.04
* CUDA 9.0
* cuDNN 7.0.5
* Java 8
* Sonnet 1.34
* Python 2.7.13
  * Tensorflow 1.12.0
  * Other python packages specified in environment.yaml


Prepare Data
----------------------
1. Download the Shutterstock/MSCOCO/GCC descriptions from [link](https://www.shutterstock.com/)/[link](http://cocodataset.org/)/[link](https://ai.google.com/research/ConceptualCaptions/download).

2. Extract the descriptions. It seems that NLTK is changing constantly. So 
the number of the descriptions obtained may be different.
    ```
    python -c "import nltk; nltk.download('punkt')"
    python preprocessing/extract_descriptions.py
    ```
    
3. Preprocess the descriptions. 
    ```
    python preprocessing/process_descriptions.py --word_counts_output_file \ 
      data/word_counts.txt --new_dict
    ```

4. Download the MSCOCO/Flickr30k images from [link](http://cocodataset.org/)/[link](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html).

5. Object detection for the training images. You need to first download the
detection model from [here][detection_model] and then extract the model under
tf_models/research/object_detection.
    ```
    python preprocessing/detect_objects.py --image_path\
      ~/dataset/mscoco/all_images --num_proc 2 --num_gpus 1
    ```
    
6. Generate tfrecord files for images.
    ```
    python preprocessing/process_images.py --image_path\
      ~/dataset/mscoco/all_images
    ```

 *You can skip step 1-2 and download below files*
* MSCOCO-Shutterstock: https://drive.google.com/drive/folders/1ay0o0gUe2iaUQScIwoVjv8Nj3KdBp18f
* Flickr30k-MSCOCO: https://drive.google.com/drive/folders/1cIo1O1-_TypJdTAY1p1q3cMwzTPL_4W6
* MSCOCO-GCC: https://drive.google.com/drive/folders/1Ih4XHaQ1zJ85d4p_hciwzXXtiXARYL2g


Training
--------

* Supervision Learning on Text Corpus
1. Train the model with only cross-entropy loss
```
python obj2sen_xe.py
```
2. Fine-tune the model with cross-entropy loss and reconstruction loss
```
python obj2sen_xe+rec.py
```

* Unsupervised Visual Alignment on Images
3. Fine-tune the model with triplet ranking loss
```
python obj2sen_tri.py
```
4. Fine-tune the model with triplet ranking loss and reconstruction loss
```
python obj2sen_tri+rec.py
```

*Pretrained Models - R2M*
* MSCOCO-Shutterstock: https://drive.google.com/drive/folders/1Nqy0Gohhu33k8cgWwRNvISuSu01nWX5u
* Flickr30k-MSCOCO: https://drive.google.com/drive/folders/1JUH2_Aq7u9mwik9maHOKclnjOVqXo1Q9
* MSCOCO-GCC: https://drive.google.com/drive/folders/13BM7PYQMfYaQ6WXMvz9_u0NgotCktydR


Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:
```
python test_obj2sen.py --job_dir [path_to_root]/save/XXXXX.pth
```

Acknowledgements
----------------

* This code began with [fengyang0317/unsupervised_captioning](https://github.com/fengyang0317/unsupervised_captioning). We thank the developers for doing most of the heavy-lifting.

[detection_model]: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz
