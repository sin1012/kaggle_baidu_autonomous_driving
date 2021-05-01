# Kaggle Baidu Autonomous Driving Baseline
<img src='https://github.com/sin1012/kaggle_baidu_autonomous_driving/blob/main/images/sample_predictions.png'>

## TLDR [Todo]
_placeholder_

## Dependencies [Todo]
_placeholder_

## Directory Settings
┣ 📦src  
┣ ┣ 📜conf.py  
┣ ┣ 📜dataset.py  
┣ ┣ 📜loss.py  
┣ ┣ 📜main.py  
┣ ┣ 📜models.py  
┣ ┣ 📜transforms.py   
┣ ┗ 📜utils.py    
┣ 📦notebooks  
┣ ┣ 📜INFERECE AND VISUALIZATION.ipynb [Todo]  
┣ ┣ 📜EDA.ipynb  
┣ ┗ 📜PUBLIC CENTERNET BASELINE.ipynb  
┣ 📦data (you should download this from kaggle)

## How to run this 
1. Download Dataset Using Kaggle API
```
! pip install kaggle
! mkdir ~/.kaggle/
! mv kaggle.json ~/.kaggle/
! kaggle competitions download -c pku-autonomous-driving
! mkdir data
! unzip pku-autonomous-driving.zip -d data
```
2. git clone this repo and move the data folder under the current directory
3. start training
```
! cd src
! python main.py --config test_config
```

## Training Details 
- architecture: CenterNet
- backbone: ResNet18 with 6 input channels (RGB(3) + Mesh(2) + Mask(1))
- input: `[Image, Mask (we want to ignore this)]` with size (2048, 512)
- output: `[keypoint heatmap, vdiff, udiff, z, yaw, pitch_sin, pitch_cos, roll]`
- loss function: focal loss for keypoint heatmap and l1 loss for regression
- batch size: 2
- data augmentations: hflip(0.5), BrightnessContrast(0.5), Blur(0.2), HSV(0.5), CLAHE(0.2)
- lr scheduling: Cosine Annealing with init lr `1e-3` and eta `1e-5`
- hardware: RTX 3090 (24G), AMD 3900XT, 64GB RAM
- calculate map on the fly
- architecture:
<img src='https://github.com/sin1012/kaggle_baidu_autonomous_driving/blob/main/images/model.png'>

## Sample Training Log
```
python main.py --config test_config
Using config file test_config
[✔️] Setting all seeds to be 888 to reproduce.
[🐻] Bad Image files removed.
[✔️] Full Training Mode...
[🐶] Building Dataset.
[✔️] Dataset initiated in train mode.
[✔️] Dataset initiated in valid mode.
[✔️] Dataset initiated in test mode.
[💻] Training on GPU 1
[✔️] Model Loaded.
[🚀] Start Training...
Train Epoch: 0  LR: 0.001000
Sum Loss: 9.027, Keypoint loss: 13.89, Regression loss: 2.0823: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:31<00:00,  2.20it/s]
Train Loss: 15.7776     Keypoint: 27.8990       Regression: 1.8281
Valid Loss: 8.6186      Keypoint: 14.0348       Regression: 1.6012      mAP: 0.004354
Train Epoch: 1  LR: 0.000999
Sum Loss: 9.447, Keypoint loss: 16.73, Regression loss: 1.0842: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:24<00:00,  2.22it/s]
Train Loss: 8.0584      Keypoint: 13.0582       Regression: 1.5293
Valid Loss: 7.1721      Keypoint: 11.5024       Regression: 1.4209      mAP: 0.027710
Train Epoch: 2  LR: 0.000996
Sum Loss: 10.493, Keypoint loss: 18.07, Regression loss: 1.4595: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:24<00:00,  2.22it/s]
Train Loss: 6.7295      Keypoint: 10.7181       Regression: 1.3704
Valid Loss: 6.1441      Keypoint: 9.6507        Regression: 1.3188      mAP: 0.047497
Train Epoch: 3  LR: 0.000991
Sum Loss: 7.889, Keypoint loss: 13.03, Regression loss: 1.3722: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:28<00:00,  2.21it/s]
Train Loss: 5.7893      Keypoint: 9.0573        Regression: 1.2607
Valid Loss: 5.5381      Keypoint: 8.5008        Regression: 1.2877      mAP: 0.047618
Train Epoch: 4  LR: 0.000984
Sum Loss: 6.206, Keypoint loss: 10.49, Regression loss: 0.9601: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:30<00:00,  2.20it/s]
Train Loss: 5.2423      Keypoint: 8.1331        Regression: 1.1758
Valid Loss: 4.9780      Keypoint: 7.6943        Regression: 1.1309      mAP: 0.071875
Train Epoch: 5  LR: 0.000976
Sum Loss: 2.897, Keypoint loss: 2.88, Regression loss: 1.4579: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:33<00:00,  2.19it/s]
Train Loss: 4.8695      Keypoint: 7.5141        Regression: 1.1124
Valid Loss: 4.5892      Keypoint: 7.0246        Regression: 1.0769      mAP: 0.086466
Train Epoch: 6  LR: 0.000965
Sum Loss: 4.728, Keypoint loss: 8.10, Regression loss: 0.6785: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:31<00:00,  2.20it/s]
Train Loss: 4.5885      Keypoint: 7.0620        Regression: 1.0575
Valid Loss: 4.3607      Keypoint: 6.6453        Regression: 1.0381      mAP: 0.087167
Train Epoch: 7  LR: 0.000953
Sum Loss: 4.502, Keypoint loss: 7.50, Regression loss: 0.7511: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:34<00:00,  2.19it/s]
Train Loss: 4.3330      Keypoint: 6.6322        Regression: 1.0168
Valid Loss: 4.3259      Keypoint: 6.6315        Regression: 1.0101      mAP: 0.088729
Train Epoch: 8  LR: 0.000939
Sum Loss: 3.240, Keypoint loss: 5.06, Regression loss: 0.7087: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:43<00:00,  2.17it/s]
Train Loss: 4.1053      Keypoint: 6.2588        Regression: 0.9759
Valid Loss: 4.0567      Keypoint: 6.1492        Regression: 0.9821      mAP: 0.095301
Train Epoch: 9  LR: 0.000923
Sum Loss: 3.392, Keypoint loss: 4.23, Regression loss: 1.2784: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [15:01<00:00,  2.12it/s]
Train Loss: 3.8921      Keypoint: 5.9104        Regression: 0.9369
Valid Loss: 4.1584      Keypoint: 6.4145        Regression: 0.9511      mAP: 0.085901
Train Epoch: 10         LR: 0.000905
Sum Loss: 1.658, Keypoint loss: 2.23, Regression loss: 0.5417: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:31<00:00,  2.20it/s]
Train Loss: 3.6936      Keypoint: 5.5683        Regression: 0.9095
Valid Loss: 3.8695      Keypoint: 5.9122        Regression: 0.9134      mAP: 0.107742
Train Epoch: 11         LR: 0.000886
Sum Loss: 3.500, Keypoint loss: 5.44, Regression loss: 0.7812: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:28<00:00,  2.21it/s]
Train Loss: 3.5266      Keypoint: 5.2871        Regression: 0.8830
Valid Loss: 3.7986      Keypoint: 5.7909        Regression: 0.9032      mAP: 0.097596
Train Epoch: 12         LR: 0.000866
Sum Loss: 2.863, Keypoint loss: 4.48, Regression loss: 0.6224: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:29<00:00,  2.20it/s]
Train Loss: 3.3418      Keypoint: 4.9607        Regression: 0.8614
Valid Loss: 3.8582      Keypoint: 5.9161        Regression: 0.9002      mAP: 0.091515
Train Epoch: 13         LR: 0.000844
Sum Loss: 1.426, Keypoint loss: 1.82, Regression loss: 0.5162: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1916/1916 [14:30<00:00,  2.20it/s]
Train Loss: 3.1762      Keypoint: 4.6692        Regression: 0.8417
Valid Loss: 3.7894      Keypoint: 5.8360        Regression: 0.8714      mAP: 0.120562
```
After 50 epochs, you should get a leaderboard score of `0.89/0.91` on public lb/private lb.
