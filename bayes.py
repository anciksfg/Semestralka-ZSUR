import matplotlib.pyplot as plt
import numpy as np
import math
import time


def bayes(X, labels):
    """
    Natrénuje bayesův klasifikátor a vrátí parametry rozdělení tříd
    """
    m = len(X) # počet prvků v trénovací množin+
    tridy, pocty_ve_tridach = np.unique(labels, return_counts=True)
    pocet_trid = len(tridy)
    mnoziny = [X[labels == i] for i in range(pocet_trid)]

    # parametry normálního rozložení pro jednotlivé třídy (odhad z trénovacích dat)
    means = [np.mean(mnozina, axis=0) for mnozina in mnoziny]
    covs = [np.cov(mnozina.T) for mnozina in mnoziny]

    # p(c) priorni ppsi trid
    pc = [len(mnozina)/m for mnozina in mnoziny]

    return means, covs, pc


def klasifikuj(posl, means, covs, pc):
    """
    Klasifikuje body posloupnosti do trid
    """
    m = len(posl)
    pocet_trid = len(means)

    labels = np.zeros(m)
    for i in range(m):
        # ppsti p(c|x) pro kazdou tridu
        pcx = np.zeros(pocet_trid)
        for trida in range(pocet_trid):
            mean = means[trida]
            cov = covs[trida]
            # ppst p(c|x) = p(x|c) * p(c) ... nedelime p(x), nehraje roli pro maximum
            pcx[trida] = pxc(posl[i, :], mean, cov) * pc[trida]
        # zarazeni do tridy, do ktere ma nejvetsi ppst
        labels[i] = np.argmax(pcx)
    return labels


def pxc(bod, mean, cov):
    """
    Spočítá ppst bodu za předpokladu dané třídy (s param mean a std)
    p(x|c)
    """
    n = len(bod)
    diff = bod - mean
    invcov = np.linalg.inv(cov)
    detcov = np.linalg.det(cov)
    exponent = math.exp( -1/2 * diff.T.dot(invcov).dot(diff) )
    ppst = (1 / (math.sqrt(2 * math.pi)**n * detcov)) * exponent
    return ppst


def zobraz_rozdeleni(stredy, grid, grid_labels, Y, labels):
    """
    Zobrazení označených bodů podle příslušných tříd
    """
    pocet_trid = len(stredy)
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
        plt.scatter(stredy[i][0], stredy[i][1], s=100, marker='o', color='orange')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bayesův klasifikátor')
    plt.show()
    return


def vytvor_grid_klasifikuj_zobraz(Y, labels, stredy, means, covs, pc):
    # vytvorit rast pro zobrazeni rozdeleni prostoru
    x_min = np.min(Y[:, 0]) - 1
    x_max = np.max(Y[:, 0]) + 1
    x_step = (x_max - x_min)/50
    y_min = np.min(Y[:, 1]) - 1
    y_max = np.max(Y[:, 1]) + 1
    y_step = (y_max - y_min)/50
    A, B = np.mgrid[x_min:x_max:x_step, y_min:y_max:y_step]
    grid = np.vstack((A.flatten(), B.flatten())).T

    # klasifikace bodů
    grid_labels = klasifikuj(grid, means, covs, pc)

    zobraz_rozdeleni(stredy, grid, grid_labels, Y, labels)


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    Y = np.loadtxt('Y.txt', dtype=float)
    labels = np.loadtxt('labels.txt', dtype=int)
    stredy = np.loadtxt('stredy.txt', dtype=float)

    start = time.time()
    # paramtery rozlozeni trid a jejich apriorni ppsti p(c)
    means, covs, pc = bayes(Y, labels)
    print("Čas běhu:", time.time() - start, 's')

    vytvor_grid_klasifikuj_zobraz(Y, labels, stredy, means, covs, pc)



