import pandas as pd
import matplotlib.pyplot as plt

HD = pd.read_csv('C:\\Users\\Cobal\\Desktop\\Machine-Learning-Exam\\test\\01-14-09-52\\record.csv')

x = list(HD['epoch'])
y1 = list(HD['trn_loss'])
y2 = list(HD['val_loss'])

y3 = list(HD['trn_auc'])
y4 = list(HD['val_auc'])

y5 = list(HD['trn_lr'])
y6 = list(HD['val_lr'])

plt.plot(x,y1, label='Training loss')
plt.plot(x,y2, label='Validation loss')
plt.xlabel('Number of epochs')
plt.title('Original TrimNet model')
plt.legend()
plt.show()

plt.plot(x,y3, label='Training area under the ROC curve')
plt.plot(x,y4, label='Validation area under the ROC curve')
plt.xlabel('Number of epochs')
plt.title('Original TrimNet model')
plt.legend()
plt.show()

plt.plot(x,y5, label='Training learning rate')
plt.plot(x,y6, label='Validation learning rate')
plt.xlabel('Number of epochs')
plt.title('Original TrimNet model')
plt.legend()
plt.show()


