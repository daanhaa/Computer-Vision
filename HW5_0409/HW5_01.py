import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 그래프를 그리기 위한 라이브러리

# 데이터 전처리 및 로드
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 신경망 모델
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 입력 크기 28x28=784
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10개의 클래스 (0~9)

    def forward(self, x):
        x = x.view(-1, 28*28)  # 28x28 이미지를 1D 벡터로 변환
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델, 손실 함수, 옵티마이저 설정
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 손실 값을 기록할 리스트
losses = []
accuracies = []  # 정확도 저장 리스트

# 훈련
epochs = 10
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

    # 에포크마다 평균 손실 기록
    avg_loss = running_loss / len(trainloader)
    losses.append(avg_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_loss}")
    
    # 테스트 정확도 계산
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    accuracies.append(accuracy)

# 최종 모델 평가 및 정확도 출력
print(f'Final Accuracy: {accuracies[-1]:.2f}%')

# 손실 및 정확도 그래프 출력
plt.figure(figsize=(12, 5))

# 손실 그래프 출력
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 정확도 그래프 출력
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), accuracies, marker='s', color='orange')
plt.title('Test Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.show()
