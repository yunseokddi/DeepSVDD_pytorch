Harness anomaly detection trainer code by RESNET152
=============

# 1. How to run?

1. Check the **argparser's parameters**
   ~~~
   usage: main.py [-h] [--num_epochs NUM_EPOCHS] [--num_epochs_ae NUM_EPOCHS_AE]
               [--lr LR] [--lr_ae LR_AE] [--weight_decay WEIGHT_DECAY]
               [--weight_decay_ae WEIGHT_DECAY_AE]
               [--lr_milestones LR_MILESTONES] [--batch_size BATCH_SIZE]
               [--pretrain PRETRAIN] [--latent_dim LATENT_DIM]
               [--normal_class NORMAL_CLASS]

    Train Deep SVDD model

    optional arguments:
    -h, --help            show this help message and exit
    --num_epochs NUM_EPOCHS, -e NUM_EPOCHS
                           Num of epochs to Deep SVDD train (default: 50)
    --num_epochs_ae NUM_EPOCHS_AE, -ea NUM_EPOCHS_AE
                        Num of epochs to AE model train (default: 50)
    --lr LR, -lr LR       learning rate for model (default: 0.001)
    --lr_ae LR_AE, -lr_ae LR_AE
                        learning rate for AE model (default: 0.001)
    --weight_decay WEIGHT_DECAY, -wd WEIGHT_DECAY
                        weight decay for model (default: 5e-07)
    --weight_decay_ae WEIGHT_DECAY_AE, -wd_ae WEIGHT_DECAY_AE
                        weight decay for model (default: 0.005)
    --lr_milestones LR_MILESTONES, -lr_mile LR_MILESTONES
                        learning rate milestones (default: [50])
    --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        batch size (default: 1024)
    --pretrain PRETRAIN, -pt PRETRAIN
                        Pretrain to AE model (default: True)
    --latent_dim LATENT_DIM, -ld LATENT_DIM
                        latent dimension (default: 32)
    --normal_class NORMAL_CLASS, -cls NORMAL_CLASS
                        Set the normal class (default: 0)
   ~~~
2. Enter `python3 main.py` **(Default mode)**

    or
    
    Enter `python3 main.py -e 50 -ea 50 -lr 1e-3 -lr_ae 1e-3 -wd 5e-7 -wd_ae 5e-3 -lr_mile [50] -bs 1024 -pt True -ld 32 -cls 0` **(Custom mode)**

# 2. Code detail

- ### A. main.py: main code
- ### B. dataloader.py: Setting the train, test data (MNIST)
- ### C. train.py: train mothod
  - **AUTOENCODER** model train
  - **Deep SVDD** model train
  - Initializing the **c** and **weigths**
  - **AUROC** test code
    
- ### D. model.py: Definition of AE, Deep SVDD model

# 3. Requirements
~~~
python version = 3.6.8 (Recommand)
~~~

~~~
torch                   1.7.1+cu110
torchaudio              0.7.2
torchsummary            1.5.1
torchvision             0.8.2+cu110
tqdm                    4.61.2
imageio                 2.9.0
scikit-learn            0.24.2
numpy                   1.16.6
~~~

# 4. Result

Model | epoch| loss
------|-------|------
AUTOENCODER| 50|2.87
Deep SVDD | 50| 0.0271

**AUROC: 97.39**


###Logs
~~~
Start AUTOENCODER train!
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.09it/s, epoch=0, train loss=69.3]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.47it/s, epoch=1, train loss=35.1]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.39it/s, epoch=2, train loss=19.6]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.43it/s, epoch=3, train loss=13.2]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.41it/s, epoch=4, train loss=10.3]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.28it/s, epoch=5, train loss=8.84]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.13it/s, epoch=6, train loss=7.81]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.38it/s, epoch=7, train loss=7.46]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.25it/s, epoch=8, train loss=7.02]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.54it/s, epoch=9, train loss=6.62]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.52it/s, epoch=10, train loss=6.32]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.55it/s, epoch=11, train loss=6.13]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.32it/s, epoch=12, train loss=5.85]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.48it/s, epoch=13, train loss=5.57]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.43it/s, epoch=14, train loss=5.28]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.52it/s, epoch=15, train loss=5.15]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.47it/s, epoch=16, train loss=5.07]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.31it/s, epoch=17, train loss=4.81]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.30it/s, epoch=18, train loss=4.67]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.59it/s, epoch=19, train loss=4.64]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.39it/s, epoch=20, train loss=4.43]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.65it/s, epoch=21, train loss=4.35]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.62it/s, epoch=22, train loss=4.31]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.48it/s, epoch=23, train loss=4.02]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.48it/s, epoch=24, train loss=4.03]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.05it/s, epoch=25, train loss=3.95]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.55it/s, epoch=26, train loss=4]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.35it/s, epoch=27, train loss=3.82]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.39it/s, epoch=28, train loss=3.67]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.36it/s, epoch=29, train loss=3.56]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.10it/s, epoch=30, train loss=3.66]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.64it/s, epoch=31, train loss=3.39]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.40it/s, epoch=32, train loss=3.49]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.48it/s, epoch=33, train loss=3.6]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.44it/s, epoch=34, train loss=3.38]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.31it/s, epoch=35, train loss=3.41]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.60it/s, epoch=36, train loss=3.32]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.34it/s, epoch=37, train loss=3.19]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.03it/s, epoch=38, train loss=3.25]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 11.90it/s, epoch=39, train loss=3.15]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 11.84it/s, epoch=40, train loss=3.08]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.37it/s, epoch=41, train loss=3.08]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.24it/s, epoch=42, train loss=3.16]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.35it/s, epoch=43, train loss=3.04]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.51it/s, epoch=44, train loss=3.07]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.59it/s, epoch=45, train loss=2.95]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.52it/s, epoch=46, train loss=3.03]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.50it/s, epoch=47, train loss=2.9]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.13it/s, epoch=48, train loss=2.83]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.30it/s, epoch=49, train loss=2.87]
Start Deep SVDD train!
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.66it/s, epoch=0, train loss=7.56]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.97it/s, epoch=1, train loss=2.92]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.10it/s, epoch=2, train loss=1.62]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.81it/s, epoch=3, train loss=1.05]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.04it/s, epoch=4, train loss=0.695]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.84it/s, epoch=5, train loss=0.517]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.05it/s, epoch=6, train loss=0.408]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.76it/s, epoch=7, train loss=0.356]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.77it/s, epoch=8, train loss=0.302]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.97it/s, epoch=9, train loss=0.28]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.05it/s, epoch=10, train loss=0.0706]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.85it/s, epoch=11, train loss=0.0633]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.86it/s, epoch=12, train loss=0.0619]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.18it/s, epoch=13, train loss=0.0527]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.99it/s, epoch=14, train loss=0.0518]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.82it/s, epoch=15, train loss=0.0481]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.09it/s, epoch=16, train loss=0.0528]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.74it/s, epoch=17, train loss=0.0533]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.07it/s, epoch=18, train loss=0.0482]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.11it/s, epoch=19, train loss=0.0484]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.07it/s, epoch=20, train loss=0.0441]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.73it/s, epoch=21, train loss=0.0474]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.01it/s, epoch=22, train loss=0.0482]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.08it/s, epoch=23, train loss=0.0411]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.71it/s, epoch=24, train loss=0.0402]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.00it/s, epoch=25, train loss=0.0397]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.53it/s, epoch=26, train loss=0.0428]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.96it/s, epoch=27, train loss=0.0362]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.10it/s, epoch=28, train loss=0.0403]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.17it/s, epoch=29, train loss=0.0369]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.06it/s, epoch=30, train loss=0.0366]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.01it/s, epoch=31, train loss=0.0363]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.18it/s, epoch=32, train loss=0.0339]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.11it/s, epoch=33, train loss=0.0335]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.93it/s, epoch=34, train loss=0.0354]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.14it/s, epoch=35, train loss=0.0374]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.09it/s, epoch=36, train loss=0.0323]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.68it/s, epoch=37, train loss=0.0341]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.72it/s, epoch=38, train loss=0.0321]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.05it/s, epoch=39, train loss=0.0301]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.47it/s, epoch=40, train loss=0.0314]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.19it/s, epoch=41, train loss=0.0298]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.15it/s, epoch=42, train loss=0.033]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.86it/s, epoch=43, train loss=0.032]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.14it/s, epoch=44, train loss=0.0278]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.22it/s, epoch=45, train loss=0.0283]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.95it/s, epoch=46, train loss=0.0272]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.86it/s, epoch=47, train loss=0.0295]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 12.68it/s, epoch=48, train loss=0.0294]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 13.02it/s, epoch=49, train loss=0.0271]
Start testing
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 16.72it/s]
Test AUROC: 97.39
~~~

###Tensorboard result
![tensorboard_img](./readme_img/tensorboard_result.png)

# 5. Reference
http://proceedings.mlr.press/v80/ruff18a.html [Official paper]

https://github.com/lukasruff/Deep-SVDD-PyTorch [Official paper code]

https://ys-cs17.tistory.com/50 [My paper review blog]

https://wsshin.tistory.com/m/3 [other's code review blog]

