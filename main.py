from Gen import GenAl
import random 
import numpy as np
import skimage.io
import os
import os.path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ga = GenAl()
    save_dir = 'img'
    x_values = []
    y_values = []
    for i, (out, fit) in enumerate(ga.view(ga.evolve)()):
        j = np.argmax(fit)
        per = out[j]
        per_fit = ga.fitness(per)
        x_values.append(i)
        y_values.append(per_fit)
        print(f'{i:0>6} {per_fit}')
        if i == 0 or (i+1)%200 == 0:
            skimage.io.imsave(os.path.join(save_dir, f'{i:0>6}.jpg'), ga.decode(per))
    
    plt.plot(x_values,y_values)
    plt.savefig("result.png")