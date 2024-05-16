import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.log_var = nn.Parameter(torch.randn(output_dim))  # 출력의 로그 분산을 학습 가능한 파라미터로 정의합니다.

    def forward(self, x):
        x = self.relu(self.fc1(x))
        mean = self.fc2(x)
        return mean, self.log_var.exp()  # 출력의 로그 분산을 exp()를 통해 양수로 만듭니다.

# 훈련 데이터 생성
def generate_data(text, vocab):
    # 각 토큰을 원-핫 인코딩으로 변환합니다.
    data = [torch.tensor([vocab.index(token)], dtype=torch.float32) for token in text]
    X = torch.stack(data[:-1])  # 입력 데이터는 문장에서 마지막 토큰을 제외한 것입니다.
    y = torch.stack(data[1:])  # 출력 데이터는 문장에서 첫 번째 토큰을 제외한 것입니다.
    return X, y

text = ["I", "am", "a", "dog", "I", "am", "a", "cat"]
vocab = sorted(set(text))

# 모델 초기화
input_dim = len(vocab)  # 단어장의 크기
hidden_dim = 10
output_dim = len(vocab)  # 다음 토큰을 예측하기 때문에 출력 차원은 단어장의 크기와 같습니다.
model = BayesianNN(input_dim, hidden_dim, output_dim)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()  # 다음 토큰을 예측하기 때문에 CrossEntropyLoss를 사용합니다.
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 훈련 함수 정의
def train(model, X_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        mean, log_var = model(X_train)
        loss = criterion(mean, y_train) + 0.5 * torch.mean(log_var - torch.exp(log_var) + mean**2 - 1)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 예측 함수 정의
def predict(model, X_test, vocab):
    with torch.no_grad():
        predictions = []
        for x in X_test:
            mean, _ = model(x.unsqueeze(0))
            _, predicted_index = mean.max(1)
            predictions.append(predicted_index.item())
        predicted_tokens = [vocab[i] for i in predictions]
        return predicted_tokens

# 훈련 데이터 생성
text = ["I", "am", "a", "dog", "I", "am", "a", "cat"]
vocab = sorted(set(text))
X_train, y_train = generate_data(text, vocab)

# 모델 훈련
train(model, X_train, y_train, num_epochs=1000)

# 예측 수행
predicted_tokens = predict(model, X_train, vocab)
print("Original Text:", text)
print("Predicted Text:", predicted_tokens)
