from SVM import SVM
import numpy as np  
w = np.array([1, 2])
print(w)
objeto = SVM()
print(SVM.lossRigid(objeto, w))

b= np.array([1])
x = np.array([3, 6])
y = np.array([0, 1])

print(SVM.hingeLoss(objeto, w, b, x, y))
