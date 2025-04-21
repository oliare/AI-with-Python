import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('assets/orders.csv')  

# 1
df['OrderDate'] = pd.to_datetime(df['OrderDate']) 

# 2
df['TotalAmount'] = df['Quantity'] * df['Price']

#3
print("> total store income:", df['TotalAmount'].sum())
print("> average TotalAmount:", df['TotalAmount'].mean().round(2))
print("> number of customer orders:\n", df['Customer'].value_counts())

# 4
print("\n> orders with TotalAmount > $500:\n", df[df['TotalAmount'] > 500])

# 5
sorted = df.sort_values('OrderDate', ascending=False)
print("\n> sorted by OrderDate by descending:\n", sorted)

# 6
start_date = pd.to_datetime('2023-06-05')
end_date = pd.to_datetime('2023-06-10')
filtered = df[(df['OrderDate'] >= start_date) & (df['OrderDate'] <= end_date)]
print("\n> orders from June 5 to June 10:\n", filtered)

# 7
#>>> a
quantity = df.groupby('Category')['Quantity'].count()
print("\n> quantity by Category:\n", quantity)

#>>> b
sales = df.groupby('Category')['TotalAmount'].sum()
print("\n> sales by Category:\n", sales)

# 8
top3 = df.groupby('Customer')['TotalAmount'].sum().sort_values(ascending=False).head(3)
print("\n> TOP 3 customers by TotalAmount:\n", top3)

# *
orders = df.groupby('OrderDate').size()
orders.plot(kind='line', title="Orders count by date")

plt.xlabel("Date")
plt.ylabel("Orders count")
plt.grid(True)
plt.tight_layout()
plt.show()

# 
cat = df.groupby('Category')['TotalAmount'].sum()
cat.plot(kind='bar', title="Total amount by category")
plt.show()
