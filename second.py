import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Дані
data = {
    "Діаметр зразка, см": [3, 2, 1, 3, 10, 2, 4, 4, 5, 5, 6, 3, 7, 6, 7, 4, 8, 8, 5, 9],
    "Колір зразка": [
        "жовтий", "зелений", "зелений", "жовтий", "червоний", "зелений", "жовтий",
        "червоний", "червоний", "жовтий", "фіолетовий", "зелений", "червоний",
        "жовтий", "червоний", "зелений", "жовтий", "рожевий", "зелений", "жовтий"
    ],
    "Вага зразка, г": [34, 54, 120, 67, 112, 145, 45, 76, 89, 57, 68, 204, 122, 145, 75, 256, 89, 167, 340, 92],
    "Товщина зразка, мм": [2, 3, 1, 1, 2, 1, 2, 3, 3, 2, 2, 1, 3, 3, 2, 1, 2, 3, 1, 2],
    "Матеріал зразка": [
        "пластик", "дерево", "метал", "дерево", "пластик", "метал", "пластик",
        "дерево", "дерево", "пластик", "пластик", "метал", "дерево", "дерево",
        "пластик", "метал", "пластик", "дерево", "метал", "пластик"
    ],
    "Тип зразка": [1, 2, 3, 2, 1, 3, 1, 2, 2, 1, 1, 3, 2, 2, 1, 3, 1, 2, 3, 1]
}

df = pd.DataFrame(data)

# Новий зразок
sample = pd.DataFrame({
    "Діаметр зразка, см": [3],
    "Колір зразка": ["зелений"],
    "Вага зразка, г": [112],
    "Товщина зразка, мм": [5],
    "Матеріал зразка": ["пластик"]
})

# Кодування категоріальних змінних
encoder = LabelEncoder()
for column in ["Колір зразка", "Матеріал зразка"]:
    df[column] = encoder.fit_transform(df[column])
    sample[column] = encoder.transform(sample[column])

# Розділення на ознаки та цільову змінну
X = df.drop(columns=["Тип зразка"])
y = df["Тип зразка"]

# Масштабування даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sample_scaled = scaler.transform(sample)

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# k-NN модель
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(sample_scaled)
print(f"Тип зразка за методом k-NN: {knn_pred[0]}")

# SVM модель
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
svm_pred = svm.predict(sample_scaled)

print(f"Тип зразка за методом SVM: {svm_pred[0]}")



