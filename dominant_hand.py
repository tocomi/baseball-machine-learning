import pandas as pd
from sklearn import datasets, tree, metrics, ensemble

'''
投手の左右別被打率をもとに投手の利き腕を判別する
'''
def main():

    df = pd.read_csv('./data/dominant_hand.csv')

    # df.locで列を抽出
    # df.valuesでDataFrameを配列に変換
    averages = df.loc[ :, [ 'right', 'left' ]].values
    hands = df[ 'hand' ].values

    # 教師データの割合を設定
    sample_size = len(df.values)
    train_size = int(sample_size * 3 / 5)

    # 識別器の作成と学習
    classifier = ensemble.RandomForestClassifier(n_estimators=20, max_depth=3, criterion='gini')
    classifier.fit(averages[:train_size], hands[:train_size])

    # 利き腕を判別する
    expected = hands[train_size:]
    predicted = classifier.predict(averages[train_size:])

    # 分類器の性能を表示
    print('Accuracy:\n', metrics.accuracy_score(expected, predicted))
    print('\nConfusion matrix:\n', metrics.confusion_matrix(expected, predicted))
    print('\nPrecision:\n', metrics.precision_score(expected, predicted, pos_label='right'))
    print('\nRecall:\n', metrics.recall_score(expected, predicted, pos_label='right'))
    print('\nF-measure:\n', metrics.f1_score(expected, predicted, pos_label='right'))

if __name__ == '__main__':
    main()