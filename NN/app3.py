import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label='Справжні ціни', marker='o')
plt.plot(y_pred[:100], label='Передбачені ціни', marker='x')
plt.title("Справжні vs Передбачені")
plt.ylabel("Ціна")
plt.legend()
plt.grid()
plt.show()
