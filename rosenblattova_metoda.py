import matplotlib.pyplot as plt
import numpy as np
import time

def train_multiple_fcns(Y, labels, num_epochs=20):
    """
    Vraci n diskriminacnich fci, kde n je pocet trid
    """
    print("Training multiple diskr functions...")
    diskr_fce = []
    pocet_trid = len(np.unique(labels))
    vyvoje_cen = []

    # natrenovat n - 1 diskretnich fci
    for i in range(pocet_trid):
        # v i-tem kroku rozlisovat jen dve tridy, aktualni a vsechny ostatni
        labels_i = labels.copy()
        labels_i[labels == i] = 1
        labels_i[labels != i] = 0
        q, vyvoj_ceny = train(Y, labels_i, num_epochs=num_epochs)
        diskr_fce.append(q)

        vyvoje_cen.append(vyvoj_ceny)

    return diskr_fce, vyvoje_cen


def train(X, labels, num_epochs=20, c=1):
    """
    Natrénuje vektor parametrů q pro rozhodnutí mezi dvěma třídami
    """
    print("Training weight vector...")
    # pocet bodu v datasetu
    m = len(labels)
    # inicializovat váhový vektor vcetne extra sloupce pro bias
    dim = X.shape[1]
    q = np.random.rand(dim + 1)
    vyvoj_ceny = []

    # trénování v epochách
    for epoch in range(num_epochs):
        cena = 0

        # pro každý bod
        for i in range(m):
            bod = X[i, :]
            bod = np.insert(bod, 0, 1)
            # spravna trida
            label = labels[i]
            # urcena trida klasifikatorem
            predicted = predict(bod, q)

            # pokud tridy souhlasi, gradient nulovy, jinak ve smeru -x
            grad = -bod * (label - predicted)
            # updatovat vahovou matici
            q = q - c * grad

            if label - predicted != 0:
                cena += 1
        vyvoj_ceny.append(cena)
    return q, vyvoj_ceny



def predict(bod, q):
    if len(bod) != len(q):
        bod = np.insert(bod, 0, 1)

    hodnota_diskr_fce = q.T.dot(bod)
    if hodnota_diskr_fce > 0:
        label = 1
    else:
        label = 0
    return label


def prehazet(X, labels, pocet_prvku=0):
    m = len(X)
    if pocet_prvku == 0:
        pocet_prvku = m
    indexes = np.random.choice(m, replace=False, size=pocet_prvku)
    return X[indexes, :], labels[indexes]


def zobraz_rozdeleni(stredy, grid, grid_labels, Y, labels):
    """
    Zobrazení označených bodů podle příslušných tříd
    """
    barvy = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'firebrick', 'magenta']

    # Zobrazení vývoje ceny
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for i in range(len(vyvoje_cen)):
        plt.plot(vyvoje_cen[i], color=barvy[i])
    plt.title('Vývoj počtu chyb během trénování')
    plt.xlabel('číslo epochy')
    plt.ylabel('Počet špatně zařazených prvků')

    # Zobrazení rozdělení prostoru diskriminačními funkcemi
    plt.subplot(1, 2, 2)
    pocet_trid = len(np.unique(labels))
    posl = np.array(Y)  # kvuli zkracenemu indexovani
    labels = np.array(labels)  # kvuli zkracenemu indexovani
    grid = np.array(grid)
    grid_labels = np.array(grid_labels)


    # zobrazeni gridu
    for i in range(0, pocet_trid):
        plt.scatter(grid[grid_labels == i, 0], grid[grid_labels == i, 1], s=5, marker='o', color=barvy[i])
    # zobrazeni nezaraditelnych bodu v gridu
    plt.scatter(grid[grid_labels == -1, 0], grid[grid_labels == -1, 1], s=5, marker='o', color='lightgrey')
    plt.scatter(grid[grid_labels == np.inf, 0], grid[grid_labels == np.inf, 1], s=5, marker='o', color='lightgrey')

    # zobrazení rozdělených bodů
    for i in range(pocet_trid):
        plt.scatter(posl[labels == i, 0], posl[labels == i, 1], s=50, marker='o', color=barvy[i])

    # zobrazeni středů
    for i in range(0, pocet_trid):
        plt.scatter(stredy[i][0], stredy[i][1], s=100, marker='o', color='orange')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rossenblattův algoritmus')
    plt.show()
    return


def classify(Y, diskr_fce):
    # pocet bodu
    m = len(Y)
    pocet_trid = len(diskr_fce)
    labels = np.ones(m) * -1
    for i in range(m):
        bod = Y[i, :]
        bod = np.insert(bod, 0, 1)
        label = -1
        for trida in range(pocet_trid):
            q = diskr_fce[trida]
            vysledek = q.T.dot(bod)
            # pokud diskr fce urcila ze patri do tridy a jeste nebyl nikam zarazen
            if vysledek > 0 and label == -1:
                label = trida
            # pokud si ho privlastnilo vice trid
            elif vysledek > 0 and label != -1:
                label = np.inf
            # # urceni posledni tridy z predposledniho klasifikatoru
            # if trida == pocet_trid - 1:
            #     if vysledek < 0 and label == -1:
            #         label = trida + 1
        labels[i] = label
        # print('bod', bod, 'label', label)
    return labels


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    Y = np.loadtxt('Y.txt', dtype=float)
    labels = np.loadtxt('labels.txt', dtype=int)
    stredy = np.loadtxt('stredy.txt', dtype=float)
    pocet_trid = len(stredy)

    Y, labels = prehazet(Y, labels, pocet_prvku=len(Y))

    diskr_fce, vyvoje_cen = train_multiple_fcns(Y, labels, num_epochs=20)

    # vytvorit rast pro zobrazeni rozdeleni prostoru
    A, B = np.mgrid[-10:10.5:0.3, -5:10.5:0.3]
    grid = np.vstack((A.flatten(), B.flatten())).T

    # klasifikace bodů
    grid_labels = classify(grid, diskr_fce)

    zobraz_rozdeleni(stredy, grid, grid_labels, Y, labels)
