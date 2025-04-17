import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# load data
# df = pd.read_csv("assets/cars.csv")
df = pd.read_csv("assets/cars_plus.csv")

df = pd.get_dummies(df, columns=['brand', 'model'], drop_first=True)

# selection of features and target variable
X = df.drop('price', axis=1)

# selection of features and target variable
# X = df[['year', 'engine_volume', 'mileage', 'horsepower']] # features
y = df['price']                                            # target

# print("Features: ", X)
# print("Target:", y)

# train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# price forecast
y_pred = model.predict(X_test)

# evaluate the model
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")

# actual prices vs forecast
plt.scatter(y_test, y_pred)
plt.xlabel("Справжня ціна")
plt.ylabel("Прогнозована ціна")
plt.title("Справжня vs Прогнозована ціна")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()

mae = mean_absolute_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred)

print(f"Mean abs error: {mae:.2f} grn")
print(f"Model quality: {r2:.2f}")
