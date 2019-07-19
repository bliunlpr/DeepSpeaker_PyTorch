# Speaker Verification

This is an implementation of Speaker Verification on Python 3, Pytorch. The model can be trained by the ge2e loss or classification loss.

# Requirements
Python 3.5, Pytorch 1.0.0.

# Data
### AISHELL
You can download [AISHELL](http://www.aishelltech.com/kysjcp) to run the code.

### Your Own Dataset
You need build train, dev and test directory. Each has ```feats.scp``` and ```utt2spk```. 
The test directory need ```pair.txt```. Each line of ```pair.txt``` is "utt_id0 feats_path0 utt_id1 feats_path1 label". 
You can run ```python3 data/make_pairs.py test 1500 1500``` to randomly build the ```pair.txt```.

# Training
We provide two training methods. One is based on "Generalized End-to-End Loss for Speaker Verfication"[GE2E](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462665). 
Anonther is x-vector(classification loss). The model is LSTM or CNN. We also provide different attention strategies and different margin strategies. 
```

sh run.sh --loss_type class_softmax | ge2e_softmax --model_type lstm | cnn --att_type  base_attention | last_state | multi_attention --margin_type Softmax ```   

```
