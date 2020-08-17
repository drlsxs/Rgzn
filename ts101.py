# 猜数字
import random
print("这是一个猜数字游戏，数值在0-9之间")
a = random.randint(0, 9)
def f():
    temp = input("请输入你猜测的数字：")
    num = int(temp)
    while True:
        if (num > 9) or (num < 0):
            print("你输入的数值超出范围啦！！！")
            temp = input("请重新输入你猜测的数字：")
            num = int(temp)
        elif num == a:
            print("恭喜你猜对啦")
            print("我的数字也是"+str(a)+"呢")
            break
        elif num < a:
            print("你输入的数字小了")
            temp = input("请重新输入你猜测的数字：")
            num = int(temp)
        elif num > a:
            print("你输入的数字大了")
            temp = input("请重新输入你猜测的数字：")
            num = int(temp)
f()
print("游戏结束")
