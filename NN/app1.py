import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# mock data
def generate_trip_data(seed=42, num_samples=500):
    np.random.seed(seed)

    date = np.random.uniform(0, 24, num_samples).reshape(-1, 1)
    trip_duration = 30 + 10 * np.sin(np.pi * date / 12) + 5 * np.random.normal(0, 1, size=date.shape)
    return date, trip_duration

# nn
def build_nn_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

# data for poly
def polynomial_regression(x_train, y_train, degree=3):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_poly, y_train)
    return model, poly

def visualize_results(x_train, y_train, x_test, y_pred_nn, y_pred_poly):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, label="Training data", alpha=0.3)
    plt.plot(x_test, y_pred_nn, label="NN prediction", color="red")
    plt.plot(x_test, y_pred_poly, label="Poly prediction", color="blue", linestyle="dashed")
    plt.legend()
    plt.title("Trip duration prediction")
    plt.xlabel("Time of day")
    plt.ylabel("Trip duration")
    plt.show()

if __name__ == "__main__":
    x_train, y_train = generate_trip_data()
    x_test = np.linspace(0, 24, 100).reshape(-1, 1) 
    
    model_nn = build_nn_model()
    model_nn.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)
    pred_nn = model_nn.predict(x_test)
    
    # poly
    model_poly, poly = polynomial_regression(x_train, y_train, degree=3)
    y_pred_poly = model_poly.predict(poly.transform(x_test))
    
    visualize_results(x_train, y_train, x_test, pred_nn, y_pred_poly)
    
    test_times = np.array([[10.5], [0.0], [2.666]])  # 10:30, 00:00, 02:40
    y_pred_nn = model_nn.predict(test_times)
    y_pred_poly = model_poly.predict(poly.transform(test_times))
    
    print(f"NN prediction for 10:30: {y_pred_nn[0][0]:.2f} minutes")
    print(f"Poly regression prediction for 10:30: {y_pred_poly[0]:.2f} minutes")

    print(f"NN prediction for 00:00: {y_pred_nn[1][0]:.2f} minutes")
    print(f"Poly regression prediction for 00:00: {y_pred_poly[1]:.2f} minutes")

    print(f"NN prediction for 02:40: {y_pred_nn[2][0]:.2f} minutes")
    print(f"Poly regression prediction for 02:40: {y_pred_poly[2]:.2f} minutes")
