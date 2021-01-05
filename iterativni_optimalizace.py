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


def iter_opt(Y, labels, ceny_trid, stredy):
    pocet_trid = len(ceny_trid)
    m = Y.shape[0]
    cetnosti = [len(Y[labels == i]) for i in range(pocet_trid)]
    prehozeno = True

    # ukonci az pokud se pri projeti vsech obrazu nezmenilo rozlozeni
    while prehozeno:
        Y, labels = promichat(Y, labels)
        prehozeno = False
        # projed vsechny obrazy
        for i in range(m):
            bod = Y[i, :]
            puv_label = labels[i]
            puv_stredi = stredy[puv_label]
            cenai = ceny_trid[puv_label]
            si = cetnosti[puv_label]
            for j in range(pocet_trid):
                # pouze pokud vychozi trida nema jeden prvek a cilova neni stejna jako puvodni
                if puv_label != j and si > 1:
                    # zmeny v puvodni tride
                    novy_stredi = puv_stredi - (bod - puv_stredi)/(si - 1)
                    zmena_cenyi = si/(si-1) * dist(bod, puv_stredi)
                    nova_cenai = cenai - zmena_cenyi

                    # zmeny v nove tride
                    cenaj = ceny_trid[j]
                    sj = cetnosti[j]
                    puv_stredj = stredy[j]
                    novy_stredj = puv_stredj + (bod - puv_stredj)/(sj + 1)
                    zmena_cenyj = sj/(sj + 1) * dist(bod, puv_stredj)
                    nova_cenaj = cenaj + zmena_cenyj

                    # provedeme zmenu pokud pokles ceny v i-te tride prevazi narudst v j-te
                    if zmena_cenyi > zmena_cenyj:
                        prehozeno = True
                        labels[i] = j
                        stredy[puv_label] = novy_stredi
                        stredy[j] = novy_stredj
                        ceny_trid[puv_label] = nova_cenai
                        ceny_trid[j] = nova_cenaj
                        cetnosti[puv_label] -= 1
                        cetnosti[j] += 1

    return Y, labels, ceny_trid, cetnosti, stredy


def promichat(Y, labels):
    m = len(Y)
    indexes = np.random.choice(m, replace=False, size=m)
    Y = Y[indexes, :]
    labels = labels[indexes]
    return Y, labels


def dist(a, b):
    """
    a: bod (1x2)
    b: bod (1x2)
    returns:  vzdalenost bodu a od bodu b
    """
    a = np.array(a)
    b = np.array(b)
    diff = a - b
    return np.dot(diff, diff.T)


def zobraz_rozdelene_body(posl, labels, stredy, cislo_grafu):
    """
    Zobrazení označených bodů podle příslušných tříd
    """
    if cislo_grafu == 1:
        plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, cislo_grafu)

    pocet_trid = len(stredy)
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
        plt.title('Rozdělení před iterativní optimalizací')
    if cislo_grafu == 2:
        plt.title('Rozdělení po iterativní optimalizaci')
        plt.show()
    return


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    Y = np.loadtxt('Y.txt', dtype=float)
    labels = np.loadtxt('labels.txt', dtype=int)
    ceny_trid = np.loadtxt('ceny_trid.txt', dtype=float)
    stredy = np.loadtxt('stredy.txt', dtype=float)

    print("Cena před optimalizací:", np.sum(ceny_trid))
    zobraz_rozdelene_body(Y, labels, stredy, cislo_grafu=1)

    Y, labels, ceny_trid, cetnosti, stredy = iter_opt(Y, labels, ceny_trid, stredy)

    print("Cena po optimalizaci:", np.sum(ceny_trid))
    zobraz_rozdelene_body(Y, labels, stredy, cislo_grafu=2)
