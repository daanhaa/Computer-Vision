import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 이미지 출력용 라이브러리

# 데이터 전처리 및 로드 (정규화 수정)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # CIFAR-10에 맞는 평균과 표준편차로 정규화
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# CNN 모델
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

# 모델, 손실 함수, 옵티마이저 설정
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 손실 값을 기록할 리스트
train_losses = []
test_losses = []

# 훈련
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 에포크마다 평균 훈련 손실 기록
    avg_train_loss = running_loss / len(trainloader)
    train_losses.append(avg_train_loss)

    # 테스트 손실 계산
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

    avg_test_loss = running_test_loss / len(testloader)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}")

# 모델 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')

# 손실 그래프 출력
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epochs+1), test_losses, label='Test Loss', marker='x')
plt.title('Training and Testing Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 테스트 이미지 출력 및 예측값 출력
data_iter = iter(testloader)
images, labels = next(data_iter)

# 예측값 계산
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 이미지 출력 (배치에서 첫 번째 이미지를 출력)
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):  # 첫 5개 이미지를 출력
    ax = axes[i]
    ax.imshow(images[i].permute(1, 2, 0))  # (C, H, W) -> (H, W, C)로 변환
    # 예측값과 실제값을 비교
    predicted_label = testset.classes[predicted[i]]  # testset에서 클래스 이름을 가져옵니다
    true_label = testset.classes[labels[i]]  # testset에서 실제 클래스 이름을 가져옵니다
    ax.set_title(f"True: {true_label}\nPred: {predicted_label}")  # 타이틀에 실제값과 예측값 표시
    ax.axis('off')

plt.show()
