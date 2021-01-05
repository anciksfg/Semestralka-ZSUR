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


def kmeans(X, pocet_trid):
    # zavedeni promennych
    labels = [0 for i in range(len(X))]
    m = len(X)  # pocet prvku trenovaci mnoziny
    vzd_bodu_ke_stredu = np.zeros((m, 1))  # vzdalenosti bodů od vlastního středu
    stredy_zustaly_stejne = False
    # průběžný vývoj celkové ceny
    J = []
    ceny_trid = []

    # urceni pocatecnich stredu - nejvzdalenejsi body od nahodne vybraneho
    stredy = urcit_pocatecni_stredy(X, pocet_trid)

    # zobrazeni pocatecniho rozmisteni středů
    # zobraz_rozdelene_body(X, labels, pocet_trid)
    # zobraz_stredy(stredy)

    # v k-tem iterativnim kroku
    while not stredy_zustaly_stejne:
        # rozdeleni obrazu do hluku
        labels, vzd_bodu_ke_stredu = rozrad_body(X, stredy, labels, vzd_bodu_ke_stredu)
        stare_stredy = stredy.copy()
        stredy = urci_nove_stredy(X, stredy, labels)
        # zjistit, jestli se stredy zmenily nebo ne
        stredy_zustaly_stejne = np.sum([dist(stare_stredy[i], stredy[i]) for i in range(pocet_trid)]) == 0

        # zjistit ceny pro jednotlive tridy (pro metody binarniho deleni)
        ceny_trid = urci_ceny_trid(labels, vzd_bodu_ke_stredu, pocet_trid)

        # celkova cena v i-tém kroku (kriterium k minimalizaci - suma vzdalenosti)
        J.append(np.sum(ceny_trid))
        # print("průběžná cena: ", J[-1])

        # prubezne zobrazeni rozdeleni bodů
        # zobraz_rozdelene_body(X, labels, pocet_trid)
        # zobraz_stredy(stredy)

    return stredy, labels, J, ceny_trid


def urci_ceny_trid(labels, vzd_bodu_ke_stredu, pocet_trid):
    ceny_trid = [0 for i in range(pocet_trid)]
    labels = np.array(labels)
    for trida in range(pocet_trid):
        ceny_trid[trida] = np.sum(vzd_bodu_ke_stredu[labels == trida])
    return ceny_trid


def urcit_pocatecni_stredy(X, pocet_trid):
    # urceni prvniho bodu nahodne z trenovaci mnoziny
    index1 = np.random.choice(len(X), replace=False, size=1)
    stredy = (X[index1].tolist())

    # urceni dalsich stredu jako nejvzdalenejsich
    for i in range(1, pocet_trid):
        # vzdalenosti bodů od všech středů
        vzdalenosti = np.zeros((len(X), len(stredy)))
        for j in range(len(X)):
            for k in range(len(stredy)):
                vzdalenosti[j][k] = dist(X[j, :], stredy[k])
        min_vzdalenosti = np.min(vzdalenosti, axis=1)
        nej_vzd_bod = np.argmax(min_vzdalenosti)
        stredy.append(X[nej_vzd_bod].tolist())
    return stredy


def rozrad_body(X, stredy, labels, vzd_bodu_ke_stredu):
    """
    return:
        labels: do jake tridy bod patri
        vzd_bodu_ke_stredu: vektor vzdáleností bodů od svých středů
    """
    m = len(labels)
    # rozrazeni i-teho bodu
    for i in range(m):
        bod = X[i]
        # print('bod', i, bod)
        # vzdalenosti i-teho bodu trenovaci mnoziny ke vsem stredum
        vzdalenosti = [0 for i in range(len(stredy))]
        for j in range(len(stredy)):
            vzdalenosti[j] = dist(X[i], stredy[j])
        # print('vzdalenosti', vzdalenosti)
        # kolikaty stred je nejbliz:
        trida = np.argmin(vzdalenosti)
        # vzdalenost bodu od stredu
        vzd_bodu_ke_stredu[i] = vzdalenosti[trida]
        labels[i] = trida

    return labels, vzd_bodu_ke_stredu


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


def urci_nove_stredy(X, stredy, labels):
    labels = np.array(labels)
    pocet_trid = len(stredy)
    for trida in range(pocet_trid):
        body_ve_tride = X[labels == trida, :]
        if len(body_ve_tride) != 0:
            stredy[trida] = np.mean(body_ve_tride, axis=0).tolist()
        # else:
        #     stredy[trida] = X[np.random.choice(pocet_trid, replace=False, size=1), :].tolist()
    return stredy


def zobraz_vyvoj_ceny(J):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(J)
    plt.xticks([i for i in range(len(J))])
    plt.title("Vývoj ceny v i-té iteraci")
    plt.xlabel("Číslo iterace")
    plt.ylabel("Celková cena")
    plt.legend(["cena (minimum = " + str(J[-1]) + ")"])


def zobraz_rozdelene_body(posl, labels, stredy):
    """
    Zobrazení označených bodů podle příslušných tříd
    """
    pocet_trid = len(stredy)
    plt.subplot(1, 2, 2)
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
    plt.title('Rozřazení bodů do ' + str(pocet_trid) + ' tříd')
    plt.show()
    return


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    X = load('data.txt')

    indexes = np.random.choice(X.shape[0], replace=False, size=1000)
    Vyber = X[indexes, :].copy()

    # Skolni priklad
    Test = np.array([[0, 1], [2, 1], [1, 3], [1, -1], [1, 5], [1, 9], [-1, 7], [3, 7]])

    # Určení středů shluků metodou maximin
    start = time.time()
    stredy, labels, J, _ = kmeans(X, pocet_trid=4)
    print("Cas behu:", time.time() - start, 's')
    print("Počet tříd:", len(stredy))
    print("Cena: ", J[-1])

    zobraz_vyvoj_ceny(J)
    zobraz_rozdelene_body(X, labels, stredy)
