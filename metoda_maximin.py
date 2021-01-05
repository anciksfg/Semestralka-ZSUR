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


def maximin(Data, q=0.5):
    """
    najde stredy shluku a pocet trid metodou maximin
    """
    # zavedeni promennych
    nepouzite = Data.copy()
    m = Data.shape[0]  # pocet bodů
    stredy = []  # stredy shluků mu1, mu2

    # Určení startovacího bodu
    start_index = np.random.choice(Data.shape[0], replace=False, size=1)

    # Přidání prvního středu
    bod = nepouzite[start_index][0].tolist()
    stredy.append(bod)
    prumer = 0  # průměr vzdáleností středů mezi sebou
    nepouzite = np.delete(nepouzite, start_index, 0)

    # tabulka vzdalenosti vsech nepouzitych bodů od všech středů
    tab_vzdalenosti = np.zeros((len(nepouzite), 1))
    for i in range(len(tab_vzdalenosti)):
        for j in range(len(stredy)):
            a = nepouzite[i]
            b = stredy[j]
            tab_vzdalenosti[i, j] = np.dot((a - b), (a - b).T)

    # zavedeni tabulky vzajemnych vzdalenosti stredu
    tab_vzdalenosti_stredu = np.zeros((1, 1))

    # zavedeni vektoru minimalnich vzdálenostích od nejbližšího středu pro každý bod
    tab_min_vzdalenosti = np.zeros((len(nepouzite), 1))
    tab_min_vzdalenosti = np.min(tab_vzdalenosti, axis=1)  # m x 1, m pocet nepouzitych

    # pro kazdy dalsi bod
    for i in range(m - 1):
        max_index = np.argmax(tab_min_vzdalenosti)  # index bodu nejvzdalenejsiho od středů
        max_vzdalenost = tab_min_vzdalenosti[max_index]

        # ukončení, pokud už žádný bod není dostatečně daleko od nejbližšího středu
        if max_vzdalenost <= q * prumer:
            break
        else:
            bod = nepouzite[max_index].tolist()
            stredy.append(bod)
            nepouzite = np.delete(nepouzite, max_index, 0)

            # prepocitani vzajemnych vzdalenosti stredu a nalezeni noveho prumeru
            tab_vzdalenosti_stredu, prumer = update_tab_vzdalenosti_stredu(tab_vzdalenosti_stredu, stredy, bod, prumer)

            # pridat sloupec k tabulce vzdalenosti a odebrani radku po pouzitem bodu
            tab_vzdalenosti = update_tab_vzdalenosti(tab_vzdalenosti, max_index, nepouzite, bod)
            tab_min_vzdalenosti = np.min(tab_vzdalenosti, axis=1)  # m x 1, m pocet nepouzitych
    return stredy


def update_tab_vzdalenosti(tab_vzdalenosti, max_index, nepouzite, bod):
    # smazat radek po vyrazenem bodu
    tab_vzdalenosti = np.delete(tab_vzdalenosti, max_index, 0)

    # pridat novy sloupec vzdalenosti k novemu stredu
    novy_sloupec = np.zeros((len(tab_vzdalenosti), 1))
    for i in range(len(novy_sloupec)):
        a = nepouzite[i]
        b = bod
        novy_sloupec[i] = np.dot((a - b), (a - b).T)

    tab_vzdalenosti = np.hstack((tab_vzdalenosti, novy_sloupec))

    return tab_vzdalenosti


def update_tab_vzdalenosti_stredu(tab_vzdalenosti_stredu, stredy, bod, prumer):
    # pocet prvku ze kterych se puvodne pocital prumer
    pocet_prvku = len(tab_vzdalenosti_stredu)**2

    # zvetseni matice vzájemných vzdáleností
    novy_sloupec = np.zeros((len(tab_vzdalenosti_stredu), 1))
    tab_vzdalenosti_stredu = np.hstack((tab_vzdalenosti_stredu, novy_sloupec))
    novy_radek = np.append(novy_sloupec.T, 0)
    tab_vzdalenosti_stredu = np.vstack((tab_vzdalenosti_stredu, novy_radek))

    for i in range(len(novy_sloupec)):
        a = np.array(stredy[i])
        b = np.array(bod)
        vzd = np.dot((a - b), (a - b).T)
        tab_vzdalenosti_stredu[-1][i] = vzd

        # prepocitani prumeru vzdalenosti
        prumer = (prumer*pocet_prvku + vzd)/(pocet_prvku + 1)
        pocet_prvku += 1
        tab_vzdalenosti_stredu[i][-1] = vzd
        prumer = (prumer*pocet_prvku + vzd)/(pocet_prvku + 1)
        pocet_prvku += 1

    # pripocitani posledni nuly k prumeru
    prumer = (prumer*pocet_prvku + 0)/(pocet_prvku + 1)
    pocet_prvku += 1

    return tab_vzdalenosti_stredu, prumer


def rozrad_body(Data, stredy):
    labels = [0 for i in range(len(Data))]
    vzd = [0 for i in range(len(stredy))]
    for i in range(len(Data)):
        a = Data[i]
        for j in range(len(stredy)):
            b = stredy[j]
            vzd[j] = np.dot((a-b), (a-b).T)
        labels[i] = np.argmin(vzd)
    return labels

def zobraz_stredy(X, stredy):
    """
    Zobrazení středů mezi body
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', color='turquoise')
    plt.scatter(stredy[0][0], stredy[0][1], s=50, marker='o', color ='orange')
    for i in range(1, len(stredy)):
        plt.scatter(stredy[i][0], stredy[i][1], s=50, marker='o', color='red')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Zobrazení středů shluků')
    return


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


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    X = load('data.txt')

    indexes = np.random.choice(X.shape[0], replace=False, size=1000)
    Vyber = X[indexes, :].copy()

    # Skolni priklad
    Test = np.array([[2, -3], [3, 3], [2, 2], [-3, 1], [-1, 0], [-3, -2], [1, -2], [3, 2]])

    # Určení středů shluků metodou maximin
    start = time.time()
    stredy = maximin(X, q=0.5)
    print("Cas behu:", time.time() - start, 's')
    print("Počet tříd:", len(stredy))

    labels = rozrad_body(X, stredy)
    zobraz_stredy(X, stredy)
    zobraz_rozdelene_body(X, labels, len(stredy))
