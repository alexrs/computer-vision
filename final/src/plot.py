import matplotlib.pyplot as plt
import numpy as np
import cv2

def variance(eig_vals):
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        plt.bar(range(80), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(80), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


def shape(X):
    x = []
    y = []
    for i, elem in enumerate(X):
        if i % 2 == 0:
            x.append(elem)
        else:
            y.append(elem)

    plt.plot(x, y, '.')
    plt.show()
            
def landmarks(X):
    x = []
    y = []
    for i, Y in enumerate(X):
        for j, elem in enumerate(Y):
            if j % 2 == 0:
                x.append(elem)
            else:
                y.append(elem)

    plt.plot(x, y, '-')
    plt.show()

def image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.show()
