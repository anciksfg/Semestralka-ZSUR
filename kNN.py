import matplotlib.pyplot as plt
import numpy as np
import math
import time


def NN_klasifikuj(Y, labels, grid):
    m = len(Y)  # pocet obrazu v trenovaci mnozine
    grid_labels = np.zeros(len(grid))

    # urcit tridu pro vsechny urcovane body
    for i in range(len(grid)):
        # spocitat vzdalenost ke vsem obrazum
        vzdalenosti = np.zeros(m)
        for j in range(m):
            diff = grid[i] - Y[j]
            vzdalenosti[j] = diff.dot(diff.T)

        min_vzor = np.argmin(vzdalenosti)
        trida = labels[min_vzor]
        grid_labels[i] = trida

    return grid_labels


def kNN_klasifikuj(Y, labels, grid, k=1):
    grid_labels = np.zeros(len(grid))
    pocet_trid = len(np.unique(labels))
    mnoziny = [Y[labels == i] for i in range(pocet_trid)]

    # urcit tridu pro vsechny urcovane body
    for i in range(len(grid)):
        # spocitat vzdalenost k obrazův v dané třídě
        min_vzdalenosti = []
        for mnozina in mnoziny:
            m = len(mnozina)
            vzdalenosti = np.zeros(m)
            for j in range(m):
                diff = grid[i] - mnozina[j]
                vzdalenosti[j] = diff.dot(diff.T)
            vzdalenosti = np.sort(vzdalenosti)
            min_vzd = np.average(vzdalenosti[0:k])
            min_vzdalenosti.append(min_vzd)
        min_index = np.argmin(min_vzdalenosti)
        grid_labels[i] = min_index
    return grid_labels


def zobraz_rozdeleni(stredy, grid, grid_labels, Y, labels, k):
    """
    Zobrazení označených bodů podle příslušných tříd
    """
    pocet_trid = len(stredy)
    posl = np.array(Y)  # kvuli zkracenemu indexovani
    labels = np.array(labels)  # kvuli zkracenemu indexovani
    grid = np.array(grid)
    grid_labels = np.array(grid_labels)

    if k == 1:
        plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, k)
    barvy = ['red', 'blue', 'green', 'purple', 'pink', 'orange', 'firebrick', 'magenta']

    # zobrazeni gridu
    for i in range(0, pocet_trid):
        plt.scatter(grid[grid_labels == i, 0], grid[grid_labels == i, 1], s=5, marker='o', color=barvy[i])

    # zobrazení rozdělených bodů
    for i in range(pocet_trid):
        plt.scatter(posl[labels == i, 0], posl[labels == i, 1], s=50, marker='o', color=barvy[i])

    # zobrazeni středů
    for i in range(0, pocet_trid):
        plt.scatter(stredy[i][0], stredy[i][1], s=100, marker='o', color='orange')

    plt.xlabel('x')
    plt.ylabel('y')

    if k == 1:
        plt.title('Klasifikátor podle 1-nejbližšího souseda')
    if k == 2:
        plt.title('Klasifikátor podle 2-nejbližších sousedů')
        plt.show()
    return


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    Y = np.loadtxt('Y.txt', dtype=float)
    labels = np.loadtxt('labels.txt', dtype=int)
    stredy = np.loadtxt('stredy.txt', dtype=float)

    # vytvorit rast pro zobrazeni rozdeleni prostoru
    A, B = np.mgrid[-10:10.5:0.5, -5:10.5:0.5]
    # A, B = np.mgrid[-10:10.5:2, -5:10.5:2]
    grid = np.vstack((A.flatten(), B.flatten())).T

    start = time.time()
    grid_labels = NN_klasifikuj(Y, labels, grid)
    print('Čas běhu podle 1-nejbližšího souseda:', time.time() - start)

    zobraz_rozdeleni(stredy, grid, grid_labels, Y, labels, k=1)

    start = time.time()
    grid_labels = kNN_klasifikuj(Y, labels, grid, k=2)
    print('Čas běhu podle 2-nejbližších sousedů:', time.time() - start)

    zobraz_rozdeleni(stredy, grid, grid_labels, Y, labels, k=2)
