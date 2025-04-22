import matplotlib.pyplot as plt
import numpy as np

# 1
x = np.linspace(-10, 10, 500)

y = x**2 * np.sin(x)

plt.plot(x, y)
plt.title("Function f(x) = x^2·sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2
data = np.random.normal(loc=5, scale=2, size=1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 3
labels = ['Piano', 'Boxing', 'Coding', 'Cycling', 'Tennis']
sizes = [20, 20, 20, 20, 20]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Hobbies')
plt.axis('equal')
plt.show()

# 4
a = np.random.normal(loc=1, scale=2, size=100)
b = np.random.normal(loc=2, scale=2, size=100)
c = np.random.normal(loc=3, scale=2, size=100)
d = np.random.normal(loc=4, scale=2, size=100)

plt.boxplot([a, b, c, d], labels=['Kiwi', 'Strawberry', 'Raspberry', 'Orange'])
plt.title('Fruit weight')
plt.xlabel("Fruit")         
plt.ylabel("Weight (kg)")
plt.show()

# 5*
x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)

plt.scatter(x, y, color='green', alpha=0.6)
plt.title('Distributed points')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 6*
x = np.linspace(-10, 10, 500)

f_1 = np.sin(x)
f_2 = np.cos(x)
f_3 = f_1 + f_2

plt.plot(x, f_1, color='red', label='sin(x)')
plt.plot(x, f_2, color='green', label='cos(x)')
plt.plot(x, f_3, color='blue', label='sin(x) + cos(x)')

plt.title("Graphs of sin(x), cos(x), and their sum")
plt.xlabel("x")  
plt.ylabel("Function value")
plt.legend(loc='upper right')
plt.grid()
plt.show()