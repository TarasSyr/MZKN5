import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Створення таблиці з даними
data = {
    'Діаметр': [3, 2, 1, 3, 10, 2, 4, 4, 5, 5, 6, 3, 7, 6, 7, 4, 8, 8, 5, 9],
    'Колір': ['жовтий', 'зелений', 'зелений', 'жовтий', 'червоний', 'зелений', 'жовтий', 'червоний',
              'червоний', 'жовтий', 'фіолетовий', 'зелений', 'червоний', 'жовтий', 'червоний', 'зелений',
              'жовтий', 'рожевий', 'зелений', 'жовтий'],
    'Вага': [34, 54, 120, 67, 112, 145, 45, 76, 89, 57, 68, 204, 122, 145, 75, 256, 89, 167, 340, 92],
    'Товщина': [2, 3, 1, 1, 2, 1, 2, 3, 3, 2, 2, 1, 3, 3, 2, 1, 2, 3, 1, 2],
    'Матеріал': ['пластик', 'дерево', 'метал', 'дерево', 'пластик', 'метал', 'пластик', 'дерево',
                 'дерево', 'пластик', 'пластик', 'метал', 'дерево', 'дерево', 'пластик', 'метал',
                 'пластик', 'дерево', 'метал', 'пластик'],
    'Тип': [1, 2, 3, 2, 1, 3, 1, 2, 3, 2, 1, 3, 2, 2, 1, 3, 1, 2, 3, 1]
}

# Створення DataFrame
df = pd.DataFrame(data)

# Конвертація текстових колонок в числові (Label Encoding)
label_encoder_color = LabelEncoder()
df['Колір_числовий'] = label_encoder_color.fit_transform(df['Колір'])

label_encoder_material = LabelEncoder()
df['Матеріал_числовий'] = label_encoder_material.fit_transform(df['Матеріал'])

# Вибір ознак та цільової змінної
X = df[['Діаметр', 'Вага', 'Товщина', 'Матеріал_числовий']]  # Особливості
y = df['Колір_числовий']  # Цільова змінна (колір)

# Розбиття на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# k-найближчих сусідів (k-NN)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Оцінка точності для k-NN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Точність для k-NN: {accuracy_knn:.2f}")

# Опорні вектори (SVM)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Оцінка точності для SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Точність для SVM: {accuracy_svm:.2f}")

# Прогнозування кольору для нового зразка (введемо нові дані для передбачення)
new_sample = [[4, 100, 2, label_encoder_material.transform(['пластик'])[0]]]  # новий зразок
predicted_color_knn = label_encoder_color.inverse_transform(knn.predict(new_sample))
predicted_color_svm = label_encoder_color.inverse_transform(svm.predict(new_sample))

print(f"Прогнозований колір (k-NN): {predicted_color_knn[0]}")
print(f"Прогнозований колір (SVM): {predicted_color_svm[0]}")
