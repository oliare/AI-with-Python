import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

speed = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 140])
consumption = np.array([9, 7.8, 6.9, 6.5, 6.3, 6.4, 6.7, 7.3, 8, 9.2, 12.8])

X = speed.reshape(-1, 1)
y = consumption

X_plot = np.linspace(20, 110, 1000).reshape(-1, 1)

results = {}

for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_plot_poly = poly.transform(X_plot)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    y_plot = model.predict(X_plot_poly)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    results[degree] = {'model': model, 'mae': mae, 'mse': mse, 'y_plot': y_plot, 'poly': poly}

    print(f"Ступінь {degree}:")
    print(f"Середня абсолютна помилка: {mae:.4f}")
    print(f"Середня квадратична помилка: {mse:.4f}\n")

best_degree = min(results, key=lambda d: results[d]['mse'])
best_model = results[best_degree]['model']
best_poly = results[best_degree]['poly']

predict_speeds = np.array([35, 95, 110, 140]).reshape(-1, 1)
predict_poly = best_poly.transform(predict_speeds)
predicted_values = best_model.predict(predict_poly)

for s, p in zip(predict_speeds.flatten(), predicted_values):
    print(f"Передбачені витрати пального при {s} км/год: {p:.2f} л/100 км")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Реальні дані')

for degree in results:
    plt.plot(X_plot, results[degree]['y_plot'], label=f'Поліном ступеня {degree}')

plt.xlabel('Швидкість (км/год)')
plt.ylabel('Витрати пального (л/100 км)')
plt.title('Поліноміальна регресія для витрат пального')
plt.legend()
plt.grid(True)
plt.show()
