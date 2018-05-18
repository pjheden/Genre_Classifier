import matplotlib.pyplot as plt
import numpy as np

files = 4

acc = np.load('acc0.npy')
acc_val = np.load('accVal0.npy')
loss_val = np.load('lossVal0.npy')
loss = np.load('loss0.npy')

for i in range(1, files):
    acc = np.concatenate( (acc, np.load('acc'+str(i)+'.npy') ))
    acc_val = np.concatenate( (acc_val, np.load('accVal'+str(i)+'.npy') ))
    loss_val = np.concatenate( (loss_val, np.load('lossVal'+str(i)+'.npy') ))
    loss = np.concatenate( (loss, np.load('loss'+str(i)+'.npy') ))

print(acc.shape, acc_val.shape)
print(loss.shape, loss_val.shape)

plt.subplot(1,2,1)
plt.plot(loss_val, label='Validation')
plt.plot(loss, label='Training')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(acc_val, label='Validation')
plt.plot(acc, label='Training')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Plot accuracy per class?
# plt.pcolormesh(arr)
# plt.title('Accuracy for each class')
# plt.xlabel('Predicted class')
# plt.ylabel('True class')
# plt.show()


''' Shows the precision and recall for a model
from sklearn.metrics import classification_report
import numpy as np

Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(x_test)
print(classification_report(Y_test, y_pred))
'''
