---
title: "[논문] Drone Object Detection Using RGB/IR Fusion"
date: 2024-03-24 17:00:00 +09:00
categories: [PAPER]
tags: [SUMMARY, PAPER, COMPUTER VISION, OBJECT DETECTION]
comments: true
---

![drone](https://cdn.pixabay.com/photo/2017/08/06/03/04/drone-2588156_640.jpg)

## 논문 출처

https://arxiv.org/abs/2201.03786

## Abstract

그동안 drone이라고 불리는 공중 비행 장치를 통한 object detection은 많은 관심을 받아왔다. 그러나, RGB/IR object detection에서 deep learning을 적용할 때 가장 큰 challenge는 **특히나 밤에 사용할 수 있는 training data가 부족하다는 점이다.** 그래서 이 논문에서는 유의미한 IR image를 얻기 위해 여러 가지 기법들을 시도하였는데, 그 중 **AIR Sim simulator engine**과 **CycleGAN**, **illumination-aware fusion framework**를 적용하였다고 한다.

## Introduction

Drone의 장점은 날아다니기 때문에 공간상의 제약이 없다는 점이다. 그렇기 때문에 장점이 매우 많을 것처럼 보이지만, 현재의 drone은 "onboard visible light RGB camera"를 사용하고 있기 때문에 야간에는 성능이 급격하게 떨어진다는 문제점이 존재한다. 그렇기 때문에 lighting condition에 제약을 받지 않는 새로운 방법을 찾아보게 되었고 이로 인해 등장하게 된 것이 **IR(Infrared) image**이다.
IR image의 경우 야간에도 object detection이 가능하지만 RGB image에 비해 pixel수도 적고 해상도도 낮기 때문에 성능은 낮다.
따라서 각각의 장단점이 존재하기 때문에 이 둘을 적절히 조합하여 주간/야간 모두 성능이 좋게 만드려고 시도하였고 이 과정의 매개체가 **IAN (adative Illumination Aware Network)**이다.

Deep learning 방법을 적용하여 training을 시키기 위해서는 IR/RGB 이미지 pair가 필요하다. 이미 존재하는 dataset의 경우 IR이거나 RGB 둘 중 하나로 존재할 뿐더러 drone으로 capture한 이미지가 아닌 경우도 있기 때문에 "paired RGB/IR image"가 부족하다.

이 논문에서는 세 가지 기법들을 통해 위의 문제점을 해결하고자 하였다.

### 1. CycleGAN

![CycleGAN](https://miro.medium.com/max/2692/1*_KxtJIVtZjVaxxl-Yl1vJg.png)

CycleGAN의 경우 RGB image가 제공이 되었을 때 이에 대응하는 IR image를 만들기 위해 사용한다. 이미 labeling된 RGB drone image가 존재하기 때문에 이 과정을 통해 역시 labeling된 IR image를 도출해낼 수 있다. 그러나, CycleGAN의 경우 야간 시간대에 얻은 RGB image의 정보 손실에 대한 회복을 cycleGAN 혼자서는 불가능하다는 점이 단점이다. 따라서 RGB 이미지로 훈련된 합성 IR 이미지가 야간에 촬영된 실제 IR 이미지에 존재했을 정보를 열화상 카메라로 드러낼 수는 없다.

### 2. AIR Sim

![AirSIM](https://cdn2.unrealengine.com/project-airsim-infrastructure-2-2560x1410-e48b37a411d6.png?resize=1&w=1920)

그래서 AIR Sim simulator를 적용하게 되었다. AIR Sim simulator의 경우 가상 환경을 만들기 때문에 도시적인 환ㄱㅇ에 제약을 받을 필요도 없고 현실적인 IR image를 주간/야간 상관 없이 만들어낼 수 있다.

## RGB/IR fusion object detection

### YOLOv4

: **fast inference time** + **high accuracy**를 얻을 수 있는 real time object detection 

### IAN (Illumination Awareness Network)

![image](https://github.com/Hyunjoon83/Hyunjoon83.github.io/assets/141709404/435334bf-c327-47e4-9b54-32e6f909eafa)
IAN의 경우 총 7개의 layer들로 구성되어 있다. 그 중 5개는 convolutional layer이고 2개는 fully-connected layer이다. 이 network의 경우 **MS COCO dataset**으로 학습시켰을 때 EfficientDet보다 두 배 이상의 빠른 속도를 보였다고 한다. 또한 **SPP(Spatial Pyramid Pooling)**을 사용하여 scale variance를 대처했다고 한다. 보통은 MaxPooling이나 AveragePooling을 주로 사용하는데 SPP라는 개념을 사용한 것이 독특한 점인 것 같다.

이 논문에서는 YOLO model을 VisDrone dataset에 대해 다시 training시킨 뒤 이를 RGB detector로 사용하였다.

여기서 VisDrone dataset의 경우 다양한 날씨 환경과 채광조건 등에 대해 이미지를 수집하였다고 한다.

IR model의 경우 RGB 3 channel을 1개의 infrared channel로 줄이는 과정을 통해 생성된다. RGB와 IR DNN을 fuse하기 위해서는 **lightweight IAN**이 필요하다고 한다. 이러한 과정을 통해 빛이 많은 주간 시간대에는 RGB로, 빛이 거의 없는 야간 시간대에는 IR로 이미지를 수집해서 RGB/IR image pair를 구성한 다음 학습을 시킨다.

![image2](https://github.com/Hyunjoon83/Hyunjoon83.github.io/assets/141709404/e11577b1-11aa-4a6c-bf87-b8dfde032470)

그리고 RGB와 IR detector의 장점을 최대한 활용하기 위해 2개의 YOLOv4 detector와 IAN Network로 model을 구성하여 RGB/IR unage 쌍이 각각의 detector를 통과하여 object detection 결과와 confidence weight를 return 하는 메커니즘이고 decision layer는 어떤 model이 믿을만하고 결과를 도출하는지 결정한다고 한다.

### Synthetic IR data generation

![cycc](https://github.com/Hyunjoon83/Hyunjoon83.github.io/assets/141709404/05deb17f-1458-41aa-a78c-60b37a7f597a)

그럼에도 deep learning object detection model을 train 시키기 위해서는 많은 양의 labeling된 RGB/IR data 쌍이 필요하다. 이 data는 흔치 않고 존재하더라도 매우 적은 양이 존재한다. 이를 해결하기 위해 이 논문에서는 세 가지 방법을 제안한다.

#### CycleGAN with Mask R-CNN Segmentation

RGB image에 대응하는 IR image를 얻기 위해 드론으로 캡쳐한 쌍으로 구성되어 있지 않은 RGB/IR image data와 기존에 존재한 labeling되지 않은 data로 CycleGAN을 train 시켜 **RGB-to-IR style adapter**를 만들었다.

그 다음 위의 이미지 처럼 bounding box로 RGB image와 IR image를 매칭시켜 의미적으로 분할을 하였다. CycleGAN의 장점은 많은 양의 합성된 IR image를 생성하기 위해 기존의 RGB drone image dataset의 장점을 활용할 수 있다.

![11](https://github.com/Hyunjoon83/Hyunjoon83.github.io/assets/141709404/135a5dda-ff1a-4ef4-a6bd-9c1ab45ad777)

그러나 이 방법의 경우 몇가지 결함들이 존재하는데, CycleGAN이 많은 수의 input을 요구한다는 점과 pixel의 개수가 적은 점, 그리고 단순한 heat signature을 pixel 값으로 바꾸는것이 부자연스러운 결과를 도출한다는 점이다. 특히 두 번째 case의 경우 Mask R-CNN의 성능에 역효과를 야기할 수도 있다.

#### Simulation Rendering of Environment

![image](https://github.com/Hyunjoon83/Hyunjoon83.github.io/assets/141709404/4c77bcf0-9a89-43b8-9524-7fe67264c57c)

위의 결함들을 해결하고자 **labeling된 IR/GRB image pair**를 만들 새로운 방법을 찾고자 Unreal engine으로 AIRSim을 통해 image 쌍을 만들었다.

#### Combining CycleGAN with Simulation Based Rendering

Simulation과 현실 세계의 간극을 줄이기 위해 Cycle-GAN만 쓰는 기법의 RGB dataset의 절반을 AIRSim으로 만든 RGB dataset으로 바꾸고 나머지 절반은 VisDrone dataset에서 가져온 RGB training image를 사용했다고 한다. 
이 방식은 현실의 data가 전송될 때의 quality를 유지시키고 synthetic RGB image 역시 IR pair에게 성공적으로 보내진다고 한다.

## Conclusion

이렇게 Drone으로 object detection을 하는 여러 기법들 중 하나인 RGB/IR fusion 기법에 대해 살펴보았다. 사실 필자의 경우 올해 4학년이라 졸업프로젝트를 진행중인데 이번에 하게 된 주제가 드론 자율주행이다보니 Object detection 기법을 드론에 적용할 방법에 대해 배우고자 이번 논문을 읽게 되었다. Object detection 분야는 이미 많이 발전했지만, 딥러닝 기법들이 점점 발전해가고 특히나 YOLO 알고리즘이 등장하면서 부터 더 빠르게 발전해온 것 같다.