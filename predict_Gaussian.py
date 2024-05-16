import numpy as np

class GaussianProcess:
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.K = self.kernel(X, X)
        self.K_inv = np.linalg.inv(self.K + 1e-8 * np.eye(len(X)))
        # print(self.K_inv)

    def predict(self, X_test):
        K_star = self.kernel(X_test, self.X_train)
        mu = (K_star.dot(self.K_inv)).T.dot(self.y_train)
        cov = self.kernel(X_test, X_test) - K_star.dot(self.K_inv).dot(K_star.T)
        return mu, np.diag(cov)

# RBF 커널 함수 정의
def rbf_kernel(X1, X2, sigma=1.0, lengthscale=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma**2 * np.exp(-0.5 / lengthscale**2 * sqdist)

# 훈련 데이터 예시
train_text = ["I", "am", "a", "cat", "you", "are", "a", "dog"]
train_targets = ["am", "a", "cat", "you", "are", "a", "dog", "<END>"]

# 특성 벡터로 변환 (여기서는 단어를 one-hot 인코딩으로 변환합니다)
vocab = sorted(set(train_text))
feature_vectors = np.eye(len(vocab))

# 가우시안 프로세스 모델 구성
kernel = lambda X1, X2: rbf_kernel(X1, X2, sigma=1.0, lengthscale=1.0)  # RBF 커널 사용
model = GaussianProcess(kernel)

# 모델 훈련
model.fit(feature_vectors, np.arange(len(train_text)).reshape(-1, 1))

# 테스트 데이터 준비
test_text = ["I", "am", "a", "dog", "I", "am", "a", "cat"]
test_features = np.eye(len(vocab))[[vocab.index(token) for token in test_text]]

# 테스트 데이터에 대한 예측 및 불확실성 추정
means, variances = model.predict(test_features)

# 예측된 다음 토큰 및 해당 예측의 표준 편차 출력
for token, mean, variance in zip(test_text, means, variances):
    print(f"Token: {token}, Predicted Next Token: {vocab[np.argmax(mean)]}, Uncertainty: {variance}")
