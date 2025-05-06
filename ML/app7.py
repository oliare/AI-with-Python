import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1.
df = pd.read_csv('assets/internship_candidates_cefr_final.csv')

# 2.
english_mapping = {
    'Elementary': 1,
    'Pre-Intermediate': 2,
    'Intermediate': 3,
    'Upper-Intermediate': 4,
    'Advanced': 5
}
df['EnglishLevel'] = df['EnglishLevel'].map(english_mapping)

# 3.
X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y = df['Accepted']

# 4. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. 
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7.
english_levels = sorted(df['EnglishLevel'].unique())
entry_scores = sorted(df['EntryTestScore'].unique())

mean_experience = df['Experience'].mean()
mean_grade = df['Grade'].mean()
mean_age = df['Age'].mean()

data_grid = []
for el in english_levels:
    for es in entry_scores:
        data_grid.append({
            'Experience': mean_experience,
            'Grade': mean_grade,
            'EnglishLevel': el,
            'Age': mean_age,
            'EntryTestScore': es
        })

grid_df = pd.DataFrame(data_grid)
grid_df['Probability'] = model.predict_proba(grid_df)[:, 1]

english_level_names = {
    1: 'Elementary',
    2: 'Pre-Intermediate',
    3: 'Intermediate',
    4: 'Upper-Intermediate',
    5: 'Advanced'
}

plt.title('Probability of acceptance')
plt.xlabel('Entry test score')
plt.ylabel('Probability of acceptance')

for el in english_levels:
    prob = grid_df[grid_df['EnglishLevel'] == el]['Probability']
    level = grid_df[grid_df['EnglishLevel'] == el]['EntryTestScore']
    
    plt.plot(level, prob, label=f'{english_level_names[el]}')

plt.legend(title='English Level')
plt.tight_layout()
plt.show()
