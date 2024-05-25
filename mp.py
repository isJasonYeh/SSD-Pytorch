import matplotlib.pyplot as plt
import pandas as pd

# 讀取 csv 文件
df = pd.read_csv(r'SSD-Pytorch\result (8).csv')

# 設定 Iter 為 x 軸
x = df['Iter']

# 繪製 Loss、cls_loss、reg_loss 的變化趨勢
plt.figure(figsize=(10, 6)) 

plt.subplot(3, 1, 1)  # 3 行 1 列，第一個圖表
plt.plot(x, df['Loss'], label='Loss')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 1, 2)  # 3 行 1 列，第二個圖表
plt.plot(x, df['cls_loss'], label='Classification Loss')
plt.ylabel('Classification Loss')
plt.legend()

plt.subplot(3, 1, 3)  # 3 行 1 列，第三個圖表
plt.plot(x, df['reg_loss'], label='Regression Loss')
plt.xlabel('Iter')
plt.ylabel('Regression Loss')
plt.legend()

plt.tight_layout()  # 調整子圖佈局
plt.show() 