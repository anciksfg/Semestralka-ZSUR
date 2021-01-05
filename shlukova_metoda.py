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



def shlukova_hladina(X, pocet_shluku = 1):
    """
    metoda shlukové hladiny - aglomerativní metoda (viz. ZSURd s150)
    vstup:
        * pocet_shluku - pocet trid do kterych data rozdelime
        * X - vstupni 2D data
    """
    # zavedeni promennych
    m = X.shape[0]  # pocet bodu
    labels = [i for i in range(m)]
    labels = np.array(labels)

    dist_table = np.zeros((m, m))  # vzdalenosti mezi jednotlivymi shluky
    Ti = [([i], 0) for i in range(m)]  # mnozina shluku pro konkretni krok, obsahuje indexy bodu a hladinu na ktere byla mnozina vytvoren
    T = [[] for i in range(m)]  # moziny shluku pro dane kroky
    h = [0 for i in range(m)]  # shluková hladina pro dane kroky

    # vypocet vychozi matice vzdalenosti
    for i in range(m):
        for j in range(i):
            distance = dist(X[i, :], X[j, :])
            dist_table[i, j] = distance
            dist_table[j, i] = distance

    # vytvoreni grafu pro dendrogram
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)

    # Jednotlive kroky shlukovani
    for k in range(1, m):
        # zastaveni shlukovani pri pozadovanem poctu trid
        if len(Ti) == pocet_shluku:
            break
        i1, i2, min_dist = find_min_indexes(dist_table)
        novy_shluk = Ti[i1][0] + Ti[i2][0]

        # pridani shluknuti do dendrogramu
        dendrogram(Ti[i1], Ti[i2], min_dist)

        Ti[i1] = (novy_shluk, min_dist)  # nahrazeni shluku na indexu i1
        Ti.pop(i2)  # odstraneni shluku na indexu i2

        dist_table = update_table(dist_table, i1, i2)
        T[k] = Ti
        h[k] = min_dist

    # Vygenerování označení tříd bodů
    if pocet_shluku != 1:
        for i in range(len(Ti)):
            labels[Ti[i][0]] = i

    # Zobrazeni dendogramu
    plt.xlabel('Hladina podobnosti h')
    plt.ylabel('Indexy jednotlivých bodů')
    plt.title('Dendrogram')
    plt.xlim([0, max(h) * 1.1])

    return h, labels


def dist(a, b):
    """
    Vzdálenost bodů a, b: kvadrat Eukleidovske vzdalenosti
    """
    return np.dot((a - b).T, (a - b))


def update_table(dist_table, i1, i2):
    """
    smaze radky a prepocita
    """
    min_row = np.minimum(dist_table[i1, :], dist_table[i2, :])
    dist_table[i1, :] = min_row  # nahrazeni minimem z obou odebranych radek
    dist_table[:, i1] = min_row.T
    dist_table = np.delete(dist_table, i2, 0)  # smazani raky i2
    dist_table = np.delete(dist_table, i2, 1)  # smazani sloupce i2
    return dist_table


def find_min_indexes(dist_table):
    """
    vrati indexy minima v tabulce vzdalenosti a minimalni vzdalenost dvou shluku
    """
    m = dist_table.shape[0]
    min_dist = np.inf
    i1 = np.inf
    i2 = np.inf
    for i in range(m):
        for j in range(i):
            if dist_table[i, j] < min_dist:
                i1 = i
                i2 = j
                min_dist = dist_table[i, j]
    return i1, i2, min_dist


def dendrogram(shluk1, shluk2, h_new):
    y1 = np.mean(shluk1[0])
    y2 = np.mean(shluk2[0])
    h1 = shluk1[1]
    h2 = shluk2[1]
    plt.plot([h1, h_new], [y1, y1], color='rosybrown', linestyle=':')  # vodorovna cara
    plt.plot([h2, h_new], [y2, y2], color='rosybrown', linestyle=':')  # vodorovna cara
    plt.plot([h_new, h_new], [y1, y2], color='firebrick')  # horizontalni cara
    return


def najdi_h_diff(h):
    """
    Nalezne ideální počet tříd, který určí pro krok, ve kterém byl největší skok
    v podobnostech dvou shluků při slučování nejmenší.
    """
    max_diff = 0
    pocet_trid = 0
    for i in range(1, len(h)):
        diff = h[i] - h[i-1]
        if diff > max_diff:
            max_diff = diff
            pocet_trid = len(h) - i + 1
    return pocet_trid, max_diff


def zobraz_rozdelene_body(posl, labels, pocet_trid):
    """
    Zobrazení označených bodů podle příslušných tříd
    """
    plt.subplot(1, 2, 2)
    posl = np.array(posl)  # kvuli zkracenemu indexovani
    labels = np.array(labels)  # kvuli zkracenemu indexovani
    barvy = ['red', 'blue', 'green', 'purple', 'pink', 'orange', 'firebrick', 'magenta']
    for i in range(pocet_trid):
        plt.scatter(posl[labels == i, 0], posl[labels == i, 1], s=50, marker='o', color=barvy[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rozřazení bodů do ' + str(pocet_trid) + ' tříd')
    plt.show()
    return


def zobraz_hladinu_podobnosti(h, h_diff, pocet_trid):
    plt.subplot(1, 2, 2)
    plt.plot([0, len(h)], [h_diff, h_diff], 'red')
    plt.plot(h[1:])
    plt.title('Hladina h')
    plt.xlabel('Počet kroků')
    plt.ylabel('Podobnost')
    plt.show()


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    X = load('data.txt')
    # print(X)
    # print(X.shape)
    #
    # # Zobrazení dat
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.title("Zobrazení datové sady")
    # plt.xlabel("c1")
    # plt.ylabel("c2")
    # plt.show()

    # Testování shlukovací metody na školním příkladu
    Test = np.array(([-3, 1], [1, 1], [-2, 0], [3, -3], [1, 2], [-2, -1]))

    # Náhodný výběr části dat
    indexes = np.random.choice(X.shape[0], replace=False, size=600)
    Vyber = X[indexes, :].copy()

    # Použití metody shlukové hladiny na zmenšeném výběru
    # metrika podobnosti: Kvadrát eukleidovské vzdálenosti
    start = time.time()
    h, _ = shlukova_hladina(Vyber)
    print('cas behu: ', time.time() - start, 's')

    # Automatické nalezení idealního počtu tříd
    pocet_trid, h_diff = najdi_h_diff(h)
    print('Počet tříd: ', pocet_trid)


    zobraz_hladinu_podobnosti(h, h_diff, pocet_trid)

    # Rozřazení bodů do určeného počtu tříd
    h, labels = shlukova_hladina(Vyber, pocet_shluku=pocet_trid)
    zobraz_rozdelene_body(Vyber, labels, pocet_trid)

