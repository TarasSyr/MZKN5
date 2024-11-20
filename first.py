import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# Дані
data = {
    "Profit": [260, 140, 190, 240, 150, 140, 170, 136, 90, 145, 118, 107, 92, 89, 46, 67],
    "Population": [729.1, 227.1, 217.1, 216.1, 115.9, 85.5, 76.9, 67.9, 69.1, 67.6, 61.4, 59.9, 53.3, 38.9, 29.4, 28.6],
    "Investment": [750, 540, 580, 430, 350, 320, 470, 420, 390, 500, 310, 420, 340, 390, 320, 250],
    "Employees": [5, 4, 5, 4, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2]
}
df = pd.DataFrame(data)

def task1():
    X = df[['Investment', 'Population', 'Employees']]
    x = df[["Investment"]]
    y = df['Profit']

    model = LinearRegression()
    model.fit(x, y)
    profit_for_graph = model.predict(x)

    plt.figure(figsize=(15, 10))
    plt.scatter(x, y)
    plt.plot(x, profit_for_graph, "green")
    plt.xlabel("Вісь Х по інвестиціях")
    plt.ylabel("Вісь У по прибутку")
    plt.title("Порівняння реальних і передбачених значень прибутку")
    plt.grid(True)
    plt.show()

    model.fit(X, y)
    full_profit_predict = model.predict(X)
    return model, full_profit_predict

def task2():
    model, full_profit_predict = task1()

    v12_task2_data = {"Investment": [250], "Population": [84], "Employees": [3]}
    new_sample = np.array([v12_task2_data['Investment'][0], v12_task2_data['Population'][0], v12_task2_data['Employees'][0]]).reshape(1, -1)
    v12_profit_predict = model.predict(new_sample)

    print(f"Передбачений прибуток для міста варіанту 12: {v12_profit_predict}","\n")

def task3():
    model, full_profit_predict = task1()

    data = {
        "Investment": [250] * 15,
        "Population": [220, 210, 120, 85, 34, 67, 54, 90, 110, 75, 91, 84, 54, 35, 37],
        "Employees": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 2, 4, 5],
        "Location": ["Львівська", "Івано-Франківська", "Тернопільська", "Волинська", "Закарпатська", "Закарпатська", "Львівська", "Львівська", "Волинська", "Івано-Франківська", "Івано-Франківська", "Львівська", "Волинська", "Волинська", "Львівська"]
    }

    df = pd.DataFrame(data)
    new_profit_predict = model.predict(df[["Investment", "Population", "Employees"]])
    df["Predicted profit"] = new_profit_predict
    sorted_df = df.sort_values(by="Predicted profit", ascending=False)
    print(sorted_df.head(3), "\n")
    return new_profit_predict

def task4():
    model, full_profit_predict = task1()
    new_profit_predict = task3()

    df = pd.DataFrame({
        "Investment": [250] * 15,
        "Population": [220, 210, 120, 85, 34, 67, 54, 90, 110, 75, 91, 84, 54, 35, 37],
        "Employees": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 2, 4, 5],
        "Location": ["Львівська", "Івано-Франківська", "Тернопільська", "Волинська", "Закарпатська", "Закарпатська", "Львівська", "Львівська", "Волинська", "Івано-Франківська", "Івано-Франківська", "Львівська", "Волинська", "Волинська", "Львівська"]
    })

    df["ROI"] = df["Investment"] / new_profit_predict
    df_sorted = df.sort_values(by="ROI", ascending=True)
    print(df_sorted.head(3),"\n")

# Викликаємо функції
task1()
task2()
task3()
task4()
