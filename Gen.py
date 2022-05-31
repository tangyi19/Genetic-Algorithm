from property import prop 
from property import Gene
import copy
import os
import os.path
from random import random
import random
import numpy as np
import skimage.draw
import skimage.io
import skimage.transform


s = random.randint(190000,200000)
class GenAl:
    def __init__(self):
        self.max_iter = 100000 #迭代轮数
        self.cro = 0.7  #交叉率
        self.mu = 0.005 #变异率
        self.population = 80 #种群数量
        self.dna_size = 100 #染色体数量
        imagee_path = 'firefox.jpg'
        image = skimage.io.imread(imagee_path)
        if image.shape[2] == 4:
            image = skimage.color.rgba2rgb(image)
            image = (255 * im).astype(np.uint8)
        self.imagee = skimage.transform.resize(
            image, (128, 128), mode='reflect', preserve_range=True).astype(np.uint8)

    def genpopula(self):
        popula = []
        for _ in range(self.population):
            indivi = Gene()
            for _ in range(self.dna_size):
                r = np.random.randint(0, self.imagee.shape[0], 3, dtype=np.uint8)
                c = np.random.randint(0, self.imagee.shape[1], 3, dtype=np.uint8)
                color = np.random.randint(0, 256, 3)
                alpha = np.random.random() * 0.5
                indivi.prop.append(prop(r, c, color, alpha))
            popula.append(indivi)
        return popula

    def decode(self, indivi):
        image = np.ones(self.imagee.shape, dtype=np.uint8) * 255
        for e in indivi.prop:
            rr, cc = skimage.draw.polygon(e.r, e.c)
            skimage.draw.set_color(image, (rr, cc), e.color, e.alpha)
        return image

    def fitness(self, indivi):
        image = self.decode(indivi)#
        assert image.shape == self.imagee.shape
        n = np.linalg.norm(np.where(self.imagee > image, self.imagee - image, image - self.imagee))#欧式距离
        return (self.imagee.size * s) ** 0.5 - n

    def obtfitness(self, popula): #获取适应度
        fit = np.zeros(self.population)
        for i, indivi in enumerate(popula):
            fit[i] = self.fitness(indivi)
        return fit


    def select(self, popula, fit):
        fit = fit - np.min(fit)
        fit = fit + np.max(fit) / 2 + 0.01
        idx = np.random.choice(np.arange(self.population), size=self.population, replace=True, p=fit / fit.sum())
        son = []
        for i in idx:
            son.append(popula[i].copy())
        return son

    def view(self, f):
        def vision(*args, **kwargs):
            opt = None
            opf = None
            for popula, fit in f(*args, **kwargs):
                max_idx = np.argmax(fit)
                min_idx = np.argmax(fit)
                if opf is None or fit[max_idx] >= opf:
                    opt = popula[max_idx]
                    opf = fit[max_idx]
                else:
                    popula[min_idx] = opt
                    fit[min_idx] = opf
                yield popula, fit
        return vision

    def crossover(self, popula):
        for i in range(0, self.population, 2):
            if np.random.random() < self.cro:
                father = popula[i]
                mathor = popula[i + 1]
                p = np.random.randint(1, self.dna_size)
                father.prop[p:], mathor.prop[p:] = mathor.prop[p:], father.prop[p:]
                popula[i] = father
                popula[i + 1] = mathor
        return popula

    def mutate(self, popula):
        for indivi in popula:
            for prop in indivi.prop:
                if np.random.random() < self.mu:
                    prop.r = np.random.randint(0, self.imagee.shape[0], 3, dtype=np.uint8)
                    prop.c = np.random.randint(0, self.imagee.shape[1], 3, dtype=np.uint8)
                    prop.color = np.random.randint(0, 256, 3)
                    prop.alpha = np.random.random() * 0.5 #透明度
        return popula

    def evolve(self):
        popula = self.genpopula()
        popula_fit = self.obtfitness(popula)
        for _ in range(self.max_iter):
            chd = self.select(popula, popula_fit)
            chd = self.crossover(chd)
            chd = self.mutate(chd)
            chd_fit = self.obtfitness(chd)
            yield chd, chd_fit
            popula = chd
            popula_fit = chd_fit