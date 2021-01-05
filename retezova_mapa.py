import matplotlib.pyplot as plt
import numpy as np
import time


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


def retezova_mapa(Data):
    # Zavedeni promennych
    m = Data.shape[0]  # pocet bodu
    d = np.zeros(m - 1)  # distance bodů
    # di je vzdálenost i-tého bodu od i+1 bodu
    nepouzite = Data.copy()  # zatím nepoužité body
    posl = list()  # výsledná posloupnost bodů

    # Určení startovacího bodu
    start_index = np.random.choice(Data.shape[0], replace=False, size=1)
    bod = nepouzite[start_index][0]
    nepouzite = np.delete(nepouzite, start_index, 0)  # odstraneni bodu
    posl.append(bod.tolist())

    # Zařazení dalších bodů za první, počítání vzdáleností
    for i in range(m - 1):
        dist, min_index = min_dist(bod, nepouzite)
        d[i] = dist  # vzdalenost od predchoziho bodu k novemu
        bod = nepouzite[min_index]
        nepouzite = np.delete(nepouzite, min_index, 0)
        posl.append(bod.tolist())
    return posl, d


def min_dist(a, b):
    """
    Nejmenší vzdálenost A od bodu z pole B a jeho index
    """
    diff = a - b
    # pomalejsi multidim implementace
    # distances = np.dot(diff, diff.T)
    # distances = np.diag(distances)
    # min_index = np.argmin(distances)
    # mdist = distances[min_index]

    # rychlejsi implementace
    mdist = np.inf
    min_index = np.inf
    for i in range(len(diff)):
        dist = np.dot(diff[i], diff[i].T)
        if dist < mdist and dist != 0:
            mdist = dist
            min_index = i
    return mdist, min_index


def zobraz_mapu(posl):
    # plt.figure(0)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    for i in range(len(posl) - 1):
        bod1 = posl[i]
        bod2 = posl[i + 1]
        plt.plot([bod1[0], bod2[0]], [bod1[1], bod2[1]], '-o', color='firebrick')
    plt.plot(posl[0][0], posl[0][1], 'o', color='orange')
    plt.title('Řetězová mapa')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()


def zobraz_vzdalenosti(d, threshold):
    # plt.figure(1)
    plt.subplot(1, 3, 2)
    plt.plot(d, 'o')
    plt.plot([0, len(d)], [threshold, threshold], 'red')
    plt.title('Vzdálenosti bodů')
    plt.ylabel("Míra podobnosti")
    plt.xlabel("Index bodu")
    # plt.show()


def urci_tridy(d, posl, threshold):
    labels = [0 for i in range(len(posl))]
    label = 0
    for i in range(len(posl) - 1):
        vzd = d[i]
        b1 = posl[i]
        b2 = posl[i + 1]
        if d[i] > threshold:
            label += 1  # nova trida pri prekroceni hranice vzdalenosti
        labels[i + 1] = label
        l1 = labels[i]
        l2 = labels[i + 1]
    pocet_trid = label + 1
    return pocet_trid, labels


def zobraz_rozdelene_body(posl, labels, pocet_trid):
    # plt.figure(2)
    plt.subplot(1, 3, 3)
    posl = np.array(posl)  # kvuli zkracenemu indexovani
    labels = np.array(labels)  # kvuli zkracenemu indexovani
    barvy = ['red', 'blue', 'green', 'purple', 'pink', 'orange', 'firebrick', 'magenta']
    for i in range(pocet_trid):
        plt.scatter(posl[labels == i, 0], posl[labels == i, 1], s=50, marker='o', color=barvy[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rozřazení bodů do ' + str(pocet_trid) + ' tříd')
    plt.show()


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    X = load('data.txt')
    indexes = np.random.choice(X.shape[0], replace=False, size=1000)
    Vyber = X[indexes, :].copy()

    start = time.time()
    posl, d = retezova_mapa(Vyber)
    print("Cas behu:", time.time() - start, 's')

    threshold = np.median(d) * len(d) * 2.5  # pri prekroceni nova trida

    pocet_trid, labels = urci_tridy(d, posl, threshold)
    print('počet tříd', pocet_trid)
    zobraz_mapu(posl)
    zobraz_vzdalenosti(d, threshold)
    zobraz_rozdelene_body(posl, labels, pocet_trid)
