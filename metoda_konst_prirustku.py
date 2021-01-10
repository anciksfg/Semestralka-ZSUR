import matplotlib.pyplot as plt
import numpy as np


def train_multiple_fcns(Y, labels, num_epochs=20, beta=0.5):
    """
    Vraci n x n - 1 diskriminacnich fci, kde n je pocet trid
    """
    print("Training multiple diskr functions...")
    pocet_trid = len(np.unique(labels))
    # slovnik funkci pro kazdou tridu (vs ostatnim), pro prvni tridu {1:q1, 2:q2, 3:q3}
    diskr_fce = [{} for i in range(pocet_trid)]
    vyvoje_cen = []
    mnoziny = [Y[labels == i] for i in range(pocet_trid)]
    # pocty bodů v jednotlivých třídách
    pocty = [len(mnozina) for mnozina in mnoziny]

    # pro i-tou tridu
    for i in range(pocet_trid):
        # diskriminacni fce i-ta vs ostatni
        for j in range(i, pocet_trid):
            if i == j:
                continue
            # v i-tem kroku rozlisovat jen dve tridy, aktualni i-tou a j-tou
            mnozina_i = mnoziny[i]
            mnozina_j = mnoziny[j]
            # sjednoceni trenovacich mnozin
            body = np.array(
                [mnozina_i[k, :] if k < pocty[i] else mnozina_j[k - pocty[i], :] for k in range(pocty[i] + pocty[j])])
            oznaceni = np.array([1 if k < pocty[i] else -1 for k in range(pocty[i] + pocty[j])])
            body, oznaceni = prehazet(body, oznaceni)

            print("Training weight vector", i, j, "...")
            q, vyvoj_ceny = train(body, oznaceni, num_epochs=num_epochs, beta=beta)

            # # zkontrolovat spravne rozdeleni dvou trid
            # testovaci_body = np.array([[1, 7.5, 4], [1, 0, -3], [1, -7.5, 3], [1, 0, 5]])
            # for bod in testovaci_body:
            #     print('bod:', bod, 'hodnota fce:', q.T.dot(bod))
            # vytvor_grid_klasifikuj_zobraz(q)

            diskr_fce[i][j] = q
            diskr_fce[j][i] = -q
            vyvoje_cen.append(vyvoj_ceny)

    # počet chyb pro každou epochu (přes všechny binární klasifikátory)
    vyvoje_cen = np.sum(vyvoje_cen, axis=0)
    return diskr_fce, vyvoje_cen


def train(X, labels, num_epochs, beta):
    """
    Natrénuje vektor parametrů q pro rozhodnutí mezi dvěma třídami
    """
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
            label = int(labels[i])
            # urcena trida klasifikatorem
            predicted = int(predict(bod, q))

            rozdil = (label - predicted) / 2
            # rozdil = (predicted - label)/2
            # pokud tridy souhlasi, gradient nulovy, jinak ve smeru -x
            grad = -bod * rozdil
            # updatovat vahovou matici
            c = beta / (np.dot(bod.T, bod))
            q = q - c * grad

            if rozdil != 0:
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
        label = -1
    return label


def prehazet(X, labels, pocet_prvku=0):
    m = len(X)
    if pocet_prvku == 0:
        pocet_prvku = m
    indexes = np.random.choice(m, replace=False, size=pocet_prvku)
    return X[indexes, :], labels[indexes]


def zobraz_rozdeleni(vyvoje_cen, stredy, grid, grid_labels, Y, labels, beta):
    """
    Zobrazení označených bodů podle příslušných tříd
    """
    barvy = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'firebrick', 'magenta']

    # Zobrazení vývoje ceny
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(vyvoje_cen, color='firebrick')
    plt.title('Vývoj počtu chyb během trénování')
    plt.xlabel('číslo epochy')
    plt.ylabel('Počet špatně zařazených prvků')

    # Zobrazení rozdělení prostoru diskriminačními funkcemi
    plt.subplot(1, 2, 2)
    pocet_trid = len(stredy)
    posl = np.array(Y)  # kvuli zkracenemu indexovani
    labels = np.array(labels)  # kvuli zkracenemu indexovani
    grid = np.array(grid)
    grid_labels = np.array(grid_labels)

    # zobrazeni gridu
    for i in range(0, pocet_trid):
        plt.scatter(grid[grid_labels == i, 0], grid[grid_labels == i, 1], s=5, marker='o', color=barvy[i])
    # zobrazeni nezaraditelnych bodu v gridu
    plt.scatter(grid[grid_labels == -1, 0], grid[grid_labels == -1, 1], s=5, marker='o', color='lightgrey')
    b = plt.scatter(grid[grid_labels == -2, 0], grid[grid_labels == -2, 1], s=5, marker='o', color='lightgrey')

    # zobrazení rozdělených bodů
    c = []
    for i in range(pocet_trid):
        c.append(plt.scatter(posl[labels == i, 0], posl[labels == i, 1], s=50, marker='o', color=barvy[i]))

    # zobrazeni středů
    for i in range(pocet_trid):
        plt.scatter(stredy[i][0], stredy[i][1], s=100, marker='o', color='orange')

    plt.legend(c, range(pocet_trid))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Metoda konstantních přírůstků, beta=' + str(beta))
    plt.show()
    return


def classify(Data, diskr_fce, norm_mean):
    Data -= norm_mean
    pocet_trid = len(diskr_fce)
    # pocet bodů
    m = len(Data)
    labels = np.ones(m) * -1
    for i in range(m):
        bod = Data[i, :]
        if len(bod) != len(diskr_fce[0][1]):
            bod = np.insert(bod, 0, 1)
        # label jeste nebyl urcen
        label = -1

        for j in range(pocet_trid):
            # rozhodnuti zda podle vsech diskr funkci dane tridy oproti ostatnim nalezi do tridy
            hodnoty_diskr_fci = []
            # zkontrolovani dane tridy vs vsem ostatnim
            for k in range(pocet_trid):
                if j == k:
                    continue
                q = diskr_fce[j][k]
                hodnota_diskr_fce = q.T.dot(bod)
                hodnoty_diskr_fci.append(hodnota_diskr_fce)
            # vsechny diskr fce zaradily bod do tridy j
            if min(hodnoty_diskr_fci) > 0 and label == -1:
                label = j
            # bod si privlastnilo vice trid
            elif min(hodnoty_diskr_fci) > 0 and label != -1:
                label = -2
        labels[i] = label
    Data += norm_mean
    return labels


def vytvor_grid_klasifikuj_zobraz(Y, labels, norm_mean, stredy, diskr_fce, vyvoje_cen, beta):
    # vytvorit rast pro zobrazeni rozdeleni prostoru
    x_min = np.min(Y[:, 0]) - 1
    x_max = np.max(Y[:, 0]) + 1
    x_step = (x_max - x_min)/60
    y_min = np.min(Y[:, 1]) - 1
    y_max = np.max(Y[:, 1]) + 1
    y_step = (y_max - y_min)/60
    A, B = np.mgrid[x_min:x_max:x_step, y_min:y_max:y_step]
    grid = np.vstack((A.flatten(), B.flatten())).T

    # klasifikace bodů
    grid_labels = classify(grid, diskr_fce, norm_mean)

    zobraz_rozdeleni(vyvoje_cen, stredy, grid, grid_labels, Y, labels, beta)


def normalize(Y):
    mean = np.mean(Y, axis=0)
    Y_normalized = Y.copy() - mean
    return Y_normalized, mean


# ___________________Testing_________________


def classify_podle_jedne_fce(Y, q):
    # pocet bodu
    m = len(Y)
    pocet_trid = len(q)
    labels = np.ones(m) * -1
    for i in range(m):
        bod = Y[i, :]
        bod = np.insert(bod, 0, 1)
        label = -1

        vysledek = q.T.dot(bod)
        # pokud diskr fce urcila ze patri do tridy a jeste nebyl nikam zarazen
        if vysledek > 0 and label == -1:
            label = 0
        if vysledek <= 0 and label == -1:
            label = 1
        # # pro info i s oznacenim bodu privlastnenych vice tridami
        # if vysledek > max_vysledek:
        #     label = trida
        #     max_vysledek = vysledek

        labels[i] = label
        # print('bod', bod, 'label', label)
    return labels


def vytvor_grid_klasifikuj_zobraz_podle_jedne(q):
    """
    zobrazi klasifikaci podle dane fce
    """
    # vytvorit rast pro zobrazeni rozdeleni prostoru
    A, B = np.mgrid[-10:10.5:0.3, -5:10.5:0.3]
    grid = np.vstack((A.flatten(), B.flatten())).T

    # klasifikace bodů
    grid_labels = classify_podle_jedne_fce(grid, q)
    zobraz_rozdeleni([], stredy, grid, grid_labels, Y, labels)


# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    Y = np.loadtxt('Y.txt', dtype=float)
    labels = np.loadtxt('labels.txt', dtype=int)
    stredy = np.loadtxt('stredy.txt', dtype=float)
    pocet_trid = len(stredy)

    Y, labels = prehazet(Y, labels, pocet_prvku=len(Y))

    Y_normalized, norm_mean = normalize(Y)
    diskr_fce, vyvoje_cen = train_multiple_fcns(Y_normalized, labels, num_epochs=20, beta=0.01)

    vytvor_grid_klasifikuj_zobraz(Y, labels, norm_mean, stredy, diskr_fce, vyvoje_cen, beta=0.01)

