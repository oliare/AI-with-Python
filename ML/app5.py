import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

x = np.linspace(-20, 20, 400).reshape(-1, 1)
y = np.sin(x) + 0.1 * x**2

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=5)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
x_poly = poly.transform(x)

model = LinearRegression()
model.fit(x_train_poly, y_train)

predicted = model.predict(x_test_poly)
value = model.predict(x_poly)

mae = mean_absolute_error(y_test, predicted)
mse = mean_squared_error(y_test, predicted)
print(f"MAE: {mae:.2f}\nMSE: {mse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Реальна функція', color='blue')
plt.plot(x, value, label='Прогноз моделі', color='red')
plt.title('Реальна функція та передбачена')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
