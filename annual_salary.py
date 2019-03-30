import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

'''
打者の成績から年俸を推定する
'''
def main():

    df = pd.read_csv('./data/annual_salary.csv')

    # reshapeを使って1次配列を2次配列に変換
    homerun = df['homerun'].values.reshape(-1, 1)
    salary = df['salary'].values.reshape(-1, 1)

    # 学習
    model = linear_model.LinearRegression()
    model.fit(homerun, salary)

    # 決定係数
    print(model.score(homerun, salary))

    # グラフの表示
    plt.scatter(homerun, salary, marker='+')
    plt.scatter(homerun, model.predict(homerun), marker='o')
    plt.show()

if __name__ == '__main__':
    main()