import time

import matplotlib.pyplot as plt
import numpy as np
import binarni_deleni as bd


def vektorova_kvantizace(X, pocet_trid):
    """
    Vektorová kvantizace, hledá středy pomocí nerovnoměřného binárního dělení
    """
    # urceni stredu pomoci nerovnomerneho binarniho deleni
    Y, labels, stredy, J, ceny_trid = bd.nerovnomerne_binarni_deleni(X, pocet_trid=4)
    # stredy = kodova kniha

    # vytvorit rast pro zobrazeni rozdeleni prostoru
    x_min = np.min(X[:, 0]) - 1
    x_max = np.max(X[:, 0]) + 1
    x_step = (x_max - x_min)/50
    y_min = np.min(X[:, 1]) - 1
    y_max = np.max(X[:, 1]) + 1
    y_step = (y_max - y_min)/50
    A, B = np.mgrid[x_min:x_max:x_step, y_min:y_max:y_step]
    grid = np.vstack((A.flatten(), B.flatten())).T

    grid_labels, grid_ceny = rozdel_grid(grid, stredy)

    return stredy, grid, grid_labels, Y, labels


def rozdel_grid(grid, stredy):
    m = len(grid)
    pocet_trid = len(stredy)
    grid_labels = np.zeros(m)
    grid_ceny = np.zeros(m)

    # pro vsechny body gridu
    for i in range(m):
        bod = grid[i]
        vzdalenosti = np.zeros(pocet_trid)

        # spocitat vzdalenosti k bodum kodove knihy
        for j in range(pocet_trid):
            vzdalenosti[j] = dist(bod, stredy[j])

        # urceni tridy, ke ktere ma bod nejblize -> bod kterym by byl v kodove knize reprezentovan
        trida = np.argmin(vzdalenosti)
        grid_labels[i] = trida
        grid_ceny[i] = vzdalenosti[trida]

    return grid_labels, grid_ceny


def zobraz_kvantizaci(kodova_kniha, grid, grid_labels, Y, labels):
    """
    Zobrazení označených bodů podle příslušných tříd
    """
    pocet_trid = len(kodova_kniha)
    posl = np.array(Y)  # kvuli zkracenemu indexovani
    labels = np.array(labels)  # kvuli zkracenemu indexovani
    grid = np.array(grid)
    grid_labels = np.array(grid_labels)

    barvy = ['red', 'blue', 'green', 'purple', 'pink', 'orange', 'firebrick', 'magenta']

    # zobrazeni gridu
    for i in range(0, pocet_trid):
        plt.scatter(grid[grid_labels == i, 0], grid[grid_labels == i, 1], s=5, marker='o', color=barvy[i])

    # zobrazení rozdělených bodů
    for i in range(pocet_trid):
        plt.scatter(posl[labels == i, 0], posl[labels == i, 1], s=50, marker='o', color=barvy[i])

    # zobrazeni středů
    for i in range(0, pocet_trid):
        plt.scatter(kodova_kniha[i][0], kodova_kniha[i][1], s=100, marker='o', color='orange')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Vektorová kvantizace')
    plt.show()
    return


def dist(a, b):
    """
    Kvadrat eukleidovske vzdalenosti bodů a a b
    """
    a = np.array(a)
    b = np.array(b)
    diff = a - b
    return np.dot(diff, diff.T)


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    X = bd.load('data.txt')

    # Rozdělení metodou rovnomerneho binarniho deleni
    start = time.time()
    kodova_kniha, grid, grid_labels, Y, labels = vektorova_kvantizace(X, pocet_trid=4)
    print("Čas běhu:", time.time() - start, 's')

    zobraz_kvantizaci(kodova_kniha, grid, grid_labels, Y, labels)
