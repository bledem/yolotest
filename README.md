# YOLOv2（Chainerバージョン）

This code has been taken here "https://github.com/leetenki/YOLOv2" and modified according to my research in Nakazawa laboratory in Keio University.
The following text is the taken README.md

YOLOv2は、2016年12月25日時点の、速度、精度ともに世界最高のリアルタイム物体検出手法です。

本リポジトリは、YOLOv2の論文をChainer上で再現実装したものです。darknetオリジナルの学習済みパラメータファイルをchainerで読み込むためのパーサと、chainer上でゼロからYOLOv2を訓練するための実装が含まれています。（YOLOv2のtiny版に関してはChainerで読み込む方法が<a href="http://qiita.com/ashitani/items/566cf9234682cb5f2d60">こちら</a>のPPAPの記事で紹介されています。今回は、Full Version のYOLOv2の読込みと、学習ともにChainer実装しています。）



Joseph Redmonさんの元論文はこちら：

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016/12/25)


[You Only Look Once](https://arxiv.org/abs/1506.02640) 


Use of Darknet and YOLOv2 for recognizing Cans and PetBottles

[darknet](http://pjreddie.com/)

[YOLOChainer](https://github.com/leetenki/YOLOv2)




##<img src="data/dance_short.gif">

##<img src="data/drive_short.gif">

##<img src="data/animal_output.gif">


## Conditions of usage
- Ubuntu 16.04.1 LTS (ARMv8 Processor rev 3 (v8l) × 4 / NVIDIA Tegra X2 (nvgpu)/integrated)
- Anaconda 2.4.1
- Python 3.5.2
- OpenCV 3.4.1
- Chainer 3.5.0
- CUDA V8.0



## darknetnet classifier
See the files :
darknet19_train.py
betty_predict.py

#<a href="./YOLOv2_execute.md">訓練済みYOLOの実行手順</a>




## YOLOv2 bbox predictor
See the files :
yolov2_train.py
yolov2_predict.py



