# HW5 - Recognition

---


## 01. 간단한 이미지 분류기 구현

### 📘 설명

손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현합니다.

### ✅ 구현 요구사항

1. MNIST 데이터셋을 로드합니다.
2. 데이터를 훈련 세트와 테스트 세트로 분할합니다.
3. 간단한 신경망 모델을 구축합니다.
4. 모델을 훈련시키고 정확도를 평가합니다.


### 💡 힌트
- tensorflow.keras.datasets에서 MNIST 데이터셋을 불러올 수 있습니다.
- Sequential 모델과 Dense 레이어를 활용하여 신경망을 구성해보세요.
- 손글씨 숫자 이미지는 28x28 픽셀 크기의 흑백 이미지입니다.

---

### **📌 코드 설명**

#### **1. 라이브러리 import**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```

- **torch**: PyTorch의 핵심 라이브러리
- **torch.nn**: 신경망 구성에 필요한 함수와 클래스를 제공
- **torch.optim**: 최적화 알고리즘을 제공
- **datasets, transforms**: torchvision을 사용하여 데이터셋을 로드하고 전처리
- **DataLoader**: 데이터를 배치로 로드하는 데 사용
- **matplotlib.pyplot**: 훈련 과정에서 손실 및 정확도를 시각화.

<br>
<br>

#### **2.데이터셋 로드 및 전처리**

```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
```



- **데이터 전처리** 
- > ```ToTensor()```: 이미지를 텐서로 변환. <br>
  > ```Normalize((0.5,), (0.5,))```: 이미지의 픽셀 값을 [-1, 1]로 정규화 <br>

- **MNIST 데이터셋**
- > ```train=True```로 훈련 세트를, train=False로 테스트 세트를 불러옴 <br>
- **DataLoader**
- > 훈련 세트는 ```batch_size=64```로 배치 크기 설정, 데이터가 무작위로 섞이도록 ```shuffle=True```로 설정 <br>

<br>
<br>

#### **3.신경망 모델 정의**
```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```


- **SimpleNN 클래스**
- ```nn.Linear```: 완전 연결층을 정의
  > 첫 번째 층: 28x28 이미지를 128차원으로 변환 <br>
  > 두 번째 층: 128차원을 64차원으로 변환 <br> 
  > 세 번째 층: 64차원을 10개의 출력 값(클래스)으로 변환 <br>
- **활성화 함수**: ```torch.relu()```를 사용하여 ReLU 활성화 함수 적용

<br>
<br> 

#### **4.모델, 손실함수, 옵티마이저 설정**
```python
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- **모델**: ```SimpleNN```을 인스턴스로 생성
- **손실 함수**: ```CrossEntropyLoss()```는 다중 클래스 분류 문제에 적합한 손실 함수.
- **옵티마이저**: ```Adam``` 옵티마이저를 사용하여 모델의 파라미터를 최적화.


<br>
<br>

### 구현 결과
<img src="https://github.com/user-attachments/assets/7345c4e3-84a9-4ef9-926d-5277e8e082b5" width="400">

![image](https://github.com/user-attachments/assets/d8655900-e058-4cbe-9415-f34cc15a7034)


#### GITHUB
#### [HW5_01파일로 이동](https://github.com/daanhaa/Computer-Vision/blob/main/HW5_0409/HW5_01.py)
---



## 02. 간단한 이미지 분류기 구현

### 📘 설명

CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행

### ✅ 구현 요구사항

1. CIFAR-10 데이터셋을 로드합니다.
2. 데이터 전처리(정규화 등)를 수행합니다.
3. CNN 모델을 설계하고 훈련시킵니다.
4. 모델의 성능을 평가하고, 테스트 이미지에 대한 예측을 수행합니다.

### 💡 힌트
- tensorflow.keras.datasets에서 CIFAR-10 데이터셋을 불러올 수 있습니다.
- Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN을 구성해보세요.
- 데이터 전처리 시 픽셀 값을 0~1 범위로 정규화하면 모델의 수렴이 빨라질 수 있습니다.
---

### **📌 코드 설명**

#### **1.데이터셋 로드 및 전처리**
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # CIFAR-10에 맞는 평균과 표준편차로 정규화
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
```

**데이터 전처리**
- ```ToTensor()```: 이미지를 텐서로 변환
-  ```Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))```: CIFAR-10에 맞는 평균과 표준편차로 픽셀 값을 정규화

**CIFAR-10 데이터셋**
- ```train=True```: 훈련 세트를, ```train=False```: 테스트 세트를 불러옵니다.

**DataLoader**
- 훈련 세트는 ```batch_size=64```로 배치 크기 설정, 데이터가 무작위로 섞이도록 ```shuffle=True```로 설정


<br>
<br>


#### **2.CNN 모델 정의**
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

```

**CNN 클래스**
- ```nn.Conv2d```: 합성곱 층을 정의
- > 첫 번째 층: 입력 채널 3, 출력 채널 32, 커널 크기 3x3 <br>
  > 두 번째 층: 입력 채널 32, 출력 채널 64, 커널 크기 3x3

- ```nn.MaxPool2d```: 풀링 층을 정의
- > 풀링 크기 2x2 <br>

- ```nn.Linear```: 완전 연결층을 정의
- > 첫 번째 완전 연결층: 64 * 8 * 8 크기를 512로 변환 <br>
  >두 번째 완전 연결층: 512 차원을 10개의 출력 클래스로 변환 <br>
  >활성화 함수: torch.relu()를 사용하여 ReLU 활성화 함수 적용<br>

<br>
<br>

#### **3.모델, 손실함수, 옵티마이저 설정**
```python
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```
- **모델**: ```CNN```을 인스턴스로 생성
- **손실 함수**: ```CrossEntropyLoss()```는 다중 클래스 분류 문제에 적합한 손실 함수.
- **옵티마이저**: ```Adam``` 옵티마이저를 사용하여 모델의 파라미터를 최적화.


<br>
<br>

### 구현 결과
<img src="https://github.com/user-attachments/assets/d25354ea-2464-41d7-a47f-70774e48459a" width="400">

<img src="https://github.com/user-attachments/assets/5b9c3c51-3072-4ee3-8da8-661c822388ba" width="600">
<img src="https://github.com/user-attachments/assets/f60f5bd9-bb86-4748-8840-7753de35d8d3" width="800">



#### GITHUB
#### [HW5_02파일로 이동](https://github.com/daanhaa/Computer-Vision/blob/main/HW5_0409/HW5_02.py)
---



