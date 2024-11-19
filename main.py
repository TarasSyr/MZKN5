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
    plt.legend()
    plt.grid(True)
    plt.show()

    model.fit(X,y)                              #Роблю модель регресії по троьох параметрах(навідміну від моделі для графіка, котра є по 1 параметру)
    full_profit_predict=model.predict(X)

    return model,full_profit_predict

task1()

def task2():

    model, full_profit_predict=task1()

    v12_task2_data=({"Investment": [250],
                     "Population": [84],
                     "Employees": [3]}  )    #дані мого варіанту

    new_sample = np.array([v12_task2_data['Investment'][0], v12_task2_data['Population'][0], v12_task2_data['Employees'][0]]).reshape(1,-1)
    v12_profit_predict=model.predict(new_sample)

    print(f"Передбачений прибуток до міста варіанту 12 становить: {v12_profit_predict}")

#task2()

def task3():
    model, full_profit_predict = task1()
    """data = {
        "Місто": ["В1", "В2", "В3", "В4", "В5", "В6", "В7", "В8", "В9", "В10", "В11", "В12", "В13", "В14", "В15"],
        "Населення (тис. осіб)": [220, 210, 120, 85, 34, 67, 54, 90, 110, 75, 91, 84, 54, 35, 37],
        "Початкові інвестиції (тис. грн)": [250] * 15,
        "Кількість працівників": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 2, 4, 5],
        "Область": [
            "Львівська", "Івано-Франківська", "Тернопільська", "Волинська", "Закарпатська",
            "Закарпатська", "Львівська", "Львівська", "Волинська", "Івано-Франківська",
            "Івано-Франківська", "Львівська", "Волинська", "Волинська", "Львівська"
        ]
    }"""

    data = ({
        "Investment": [250] * 15,
        "Population": [220, 210, 120, 85, 34, 67, 54, 90, 110, 75, 91, 84, 54, 35, 37],
        "Employees": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 2, 4, 5],
        "Location": [
            "Львівська", "Івано-Франківська", "Тернопільська", "Волинська", "Закарпатська",
            "Закарпатська", "Львівська", "Львівська", "Волинська", "Івано-Франківська",
            "Івано-Франківська", "Львівська", "Волинська", "Волинська", "Львівська"
        ]
    })

    df = pd.DataFrame(data)
    new_profit_predict=model.predict(df[["Investment", "Population", "Employees"]])
    df["Predicted profit"]=new_profit_predict
    sorted_df=df.sort_values(by="Predicted profit", ascending=False)
    print(sorted_df.head(3))
    return new_profit_predict

task3()

def task4():
    model, full_profit_predict = task1()
    new_profit_predict=task3()
    """data = {
        "Population": [729.1, 227.1, 217.1, 216.1, 115.9, 85.5, 76.9, 67.9, 69.1, 67.6, 61.4, 59.9, 53.3, 38.9, 29.4, 28.6],
        "Investment": [750, 540, 580, 430, 350, 320, 470, 420, 390, 500, 310, 420, 340, 390, 320, 250],
        "Employees": [5, 4, 5, 4, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2],
        "Location": [
            "Львівська", "Івано-Франківська", "Тернопільська", "Волинська", "Закарпатська", "Закарпатська",
            "Львівська", "Львівська", "Волинська", "Івано-Франківська", "Івано-Франківська", "Львівська",
            "Волинська", "Волинська", "Львівська", "Тернопільська"
        ]
    }"""

    data = ({
        "Investment": [250] * 15,
        "Population": [220, 210, 120, 85, 34, 67, 54, 90, 110, 75, 91, 84, 54, 35, 37],
        "Employees": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3, 2, 4, 5],
        "Location": [
            "Львівська", "Івано-Франківська", "Тернопільська", "Волинська", "Закарпатська",
            "Закарпатська", "Львівська", "Львівська", "Волинська", "Івано-Франківська",
            "Івано-Франківська", "Львівська", "Волинська", "Волинська", "Львівська"
        ]
    })

    df = pd.DataFrame(data)
    df["ROI"]=df["Investment"]/new_profit_predict
    df_sorted=df.sort_values(by="ROI", ascending=True)
    print(df_sorted.head(3))

task4()