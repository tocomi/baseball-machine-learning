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
    results = df.loc[ :, [ 'game', 'homerun', 'steel' ]].values
    salary = df['salary'].values.reshape(-1, 1)

    # 学習
    model = linear_model.LinearRegression()
    model.fit(results, salary)

    # 係数と切片
    print('coef: ', model.coef_)
    print('intercept: ', model.intercept_)

    # 決定係数
    print('score: ', model.score(results, salary))

if __name__ == '__main__':
    main()