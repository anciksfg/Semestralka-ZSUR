import time

import matplotlib.pyplot as plt
import numpy as np
import kmeans as km


def load(nazev):
    """
    vstup:
        * nazev - nazev souboru
    vystup:
        * seznam dat (mx2)
    """
    c1 = []
    c2 = []
    with open(nazev, 'r') as f:
        for line in f:
            line = line.split()
            (a, b) = float(line[0]), float(line[1])
            c1.append(a)
            c2.append(b)
    return np.column_stack((c1, c2))


def rovnomerne_binarni_deleni(X, pocet_trid):
    # Zavedeni promennych
    Tridy = []  # data o kazde nove tride: mnozina bodu, stred, cena

    # dokud nevytvorim pozadovany pocet trid
    while len(Tridy) != pocet_trid:
        # pokud volam poprve - data pro volani kmeans je cela mnozina
        if len(Tridy) == 0:
            data = X
        # data pro volani kmeans - prvni trida
        else:
            data = Tridy.pop(0)[:][0]

        # rozdeleni vybrane tridy metodou kmeans do dvou trid
        stredy, labels, J, ceny_trid = km.kmeans(data, 2)

        labels = np.array(labels)
        # pridani prvni vytvorene tridy na konec trid
        data1 = data[labels == 0]
        Tridy.append((data1, stredy[0], ceny_trid[0]))
        # pridani druhe na konec trid
        data2 = data[labels == 1]
        Tridy.append((data2, stredy[1], ceny_trid[1]))

    # rozdeleni dat o tridach ke vraceni
    labels = []
    Y = []
    for i in range(pocet_trid):
        mnozina = Tridy[i][0]
        for j in range(len(mnozina)):
            Y.append(mnozina[j].tolist())
            labels.append(i)
    stredy = [x[1] for x in Tridy]
    ceny_trid = [x[2] for x in Tridy]
    J = np.sum(ceny_trid)
    return Y, labels, stredy, J, ceny_trid


def nerovnomerne_binarni_deleni(X, pocet_trid):
    # Zavedeni promennych
    Tridy = []  # data o kazde nove tride: mnozina bodu, stred, cena

    # dokud nevytvorim pozadovany pocet trid
    while len(Tridy) != pocet_trid:
        if len(Tridy) == 0:
            data = X  # data pouzita pro volani kmeans
        else:
            Tridy.sort(key=lambda x: x[2])  # seradit tridy podle jejich ceny (mensi slozitost nez hledat minimum)
            data = Tridy.pop(-1)[:][0]  # vybrat tridu s nejvetsi cenou

        stredy, labels, J, ceny_trid = km.kmeans(data, 2)

        labels = np.array(labels)
        # pridani prvni vytvorene tridy na konec trid
        data1 = data[labels == 0]
        Tridy.append((data1, stredy[0], ceny_trid[0]))
        # pridani druhe na konec trid
        data2 = data[labels == 1]
        Tridy.append((data2, stredy[1], ceny_trid[1]))

    # rozdeleni dat o tridach ke vraceni
    labels = []
    Y = []
    for i in range(pocet_trid):
        mnozina = Tridy[i][0]
        for j in range(len(mnozina)):
            Y.append(mnozina[j].tolist())
            labels.append(i)
    stredy = [x[1] for x in Tridy]
    ceny_trid = [x[2] for x in Tridy]
    J = np.sum(ceny_trid)
    return np.array(Y), np.array(labels), np.array(stredy), J, np.array(ceny_trid)


def zobraz_rozdelene_body(posl, labels, stredy, cislo_grafu):
    """
    Zobrazení označených bodů podle příslušných tříd
    """
    if cislo_grafu == 1:
        plt.figure(figsize=(15, 5))

    pocet_trid = len(stredy)
    plt.subplot(1, 2, cislo_grafu)
    posl = np.array(posl)  # kvuli zkracenemu indexovani
    labels = np.array(labels)  # kvuli zkracenemu indexovani
    barvy = ['red', 'blue', 'green', 'purple', 'pink', 'orange', 'firebrick', 'magenta']

    # zobrazení rozdělených bodů
    for i in range(pocet_trid):
        plt.scatter(posl[labels == i, 0], posl[labels == i, 1], s=50, marker='o', color=barvy[i])

    # zobrazeni středů
    for i in range(0, len(stredy)):
        plt.scatter(stredy[i][0], stredy[i][1], s=100, marker='o', color='orange')

    plt.xlabel('x')
    plt.ylabel('y')
    if cislo_grafu == 1:
        plt.title('Rovnoměrné binární dělení')
    if cislo_grafu == 2:
        plt.title('Nerovnoměrné binární dělení')
        plt.show()
    return


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    X = load('data.txt')

    indexes = np.random.choice(X.shape[0], replace=False, size=2000)
    Vyber = X[indexes, :].copy()

    # Skolni priklad
    Test = np.array([[0, 1], [2, 1], [1, 3], [1, -1], [1, 5], [1, 9], [-1, 7], [3, 7]])

    # Rozdělení metodou rovnomerneho binarniho deleni
    start = time.time()
    Y, labels, stredy, J, ceny_trid = rovnomerne_binarni_deleni(X, pocet_trid=4)
    print('Rovnoměrné binární dělení')
    print("Čas běhu:", time.time() - start, 's')
    print("Cena: ", J)
    zobraz_rozdelene_body(Y, labels, stredy, cislo_grafu=1)

    # Rozdělení metodou primeho binarniho deleni
    start = time.time()
    Y, labels, stredy, J, ceny_trid = nerovnomerne_binarni_deleni(X, pocet_trid=4)
    print("\nPřímé binární dělení")
    print("Čas běhu:", time.time() - start, 's')
    print("Cena: ", J)
    zobraz_rozdelene_body(Y, labels, stredy, cislo_grafu=2)

    # ulozeni dat a labelů pro iterativní optimalizaci
    Y = np.array(Y)
    np.savetxt('Y.txt', Y, fmt='%f')

    ceny_trid = np.array(ceny_trid)
    np.savetxt('ceny_trid.txt', ceny_trid, fmt='%f')

    stredy = np.array(stredy)
    np.savetxt('stredy.txt', stredy, fmt='%f')

    labels = np.array(labels)
    np.savetxt('labels.txt', labels, fmt='%d')
    labelsb = np.loadtxt('labels.txt', dtype=int)
