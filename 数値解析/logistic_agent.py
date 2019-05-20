#ロジスティック写像のエージェント
#モジュールインポート
import numpy as np
import matplotlib.pyplot as plt 
import math
import random

#イテレーション回数
T = 100

#クラス定義
#Agentクラス
class Agent:
    def __init__(self, cat):
        self.category = cat
        #self.x = 0.3
        #self.y = 0.3
        self.x = (random.random())/10000 #初期値は0~1でランダム
        self.y = (random.random())/10000
        self.c = 4.0
        
    #ロジスティック関数定義 
    def logistic(self, z):
        return self.c*z*(1-z)
    def calcnext(self): #次時刻の状態の計算
        if self.category == 0: #categoryが0ならcat0へスイッチ
            self.cat0() 
        else:
            print("ERROR カテゴリがありません\n")
    def cat0(self): #カテゴリ0のエージェントの計算
        self.x = self.logistic(self.x) #ロジステック写像で次の座標を決める
        self.y = self.logistic(self.y)
        
    def pustate(self): #状態の出力
        print(self.x, self.y)

#1イテレーションでの計算
def calcn(a): #aはagentが格納されたlist
    for i in range(len(a)):
        xlist.append(a[i].x)
        ylist.append(a[i].y)
        a[i].calcnext()
        a[i].pustate()
       

#実行部分
a = [Agent(0) for i in range(10)] #agent生成

xlist = []
ylist = []

#シミュレーション
for t in range(T):
    calcn(a)
    #グラフの表示
    plt.clf() #1つ前の点を消去
    plt.axis([0,1,0,1])
    plt.plot(xlist, ylist, ".")
    plt.pause(0.01)
    xlist.clear() 
    ylist.clear()
plt.show()

