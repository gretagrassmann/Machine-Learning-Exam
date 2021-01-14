import pandas as pd
import matplotlib.pyplot as plt

HD_original = pd.read_csv('C:\\Users\\Cobal\\Desktop\\comparison\\original.csv')
HD_compare = pd.read_csv('C:\\Users\\Cobal\\Desktop\\comparison\\focalloss.csv')
x = list(HD_original['epoch'])
t_loss_original = list(HD_original['trn_loss'])
v_loss_original = list(HD_original['val_loss'])

t_auc_original = list(HD_original['trn_auc'])
v_auc_original = list(HD_original['val_auc'])

t_lr_original = list(HD_original['trn_lr'])
v_lr_original = list(HD_original['val_lr'])

t_loss_compare = list(HD_compare['trn_loss'])
v_loss_compare = list(HD_compare['val_loss'])

t_auc_compare = list(HD_compare['trn_auc'])
v_auc_compare = list(HD_compare['val_auc'])

t_lr_compare = list(HD_compare['trn_lr'])
v_lr_compare = list(HD_compare['val_lr'])

a = "with a Focal loss"

plt.plot(x,t_loss_original, alpha=0.4,label='Original training loss')
plt.plot(x,v_loss_original,alpha=0.4, label='Original validation loss')
plt.plot(x,t_loss_compare, label='Training loss {}'.format(a))
plt.plot(x,v_loss_compare,  label='Validation loss {}'.format(a))
plt.xlabel('Number of epochs')
plt.title('Comparison between the original model and the model with {}'.format(a))
plt.legend()
plt.show()

plt.plot(x,t_auc_original, alpha=0.4,label='Original training area under the ROC curve')
plt.plot(x,v_auc_original, alpha=0.4,label='Original validation area under the ROC curve')
plt.plot(x,t_auc_compare, label='Training area under the ROC curve {}'.format(a))
plt.plot(x,v_auc_compare,   label='Validation area under the ROC curve {}'.format(a))
plt.xlabel('Number of epochs')
plt.title('Comparison between the original model and the model with {}'.format(a))
plt.legend()
plt.show()

plt.plot(x,t_lr_original,alpha=0.4, label='Original training learning rate')
plt.plot(x,v_lr_original,alpha=0.4, label='Original validation learning rate')
plt.plot(x,t_lr_compare,  label='Training learning rate {}'.format(a))
plt.plot(x,v_lr_compare,  label='Validation learning rate {}'.format(a))
plt.xlabel('Number of epochs')
plt.title('Comparison between the original model and the model with {}'.format(a))
plt.legend()
plt.show()




