import random
from collections import Counter


def wczytaj_dane(nazwa_pliku):
    dane = []
    etykiety = []
    with open(nazwa_pliku, 'r') as plik:
        for linia in plik:
            linia = linia.strip()
            if not linia:
                continue
            elementy = linia.split(',')
            cechy = [float(x) for x in elementy[:-1]]
            klasa = elementy[-1]

            dane.append(cechy)
            etykiety.append(klasa)
    return dane, etykiety


def kwadrat_odleglosci(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


def kmeans(dane, etykiety, k, max_iter=100):
    random.seed(0) #
    indeksy = random.sample(range(len(dane)), k) # k indeksow probek
    centroidy = [dane[i] for i in indeksy]

    poprzednie_przypisania = None

    for iteracja in range(1, max_iter + 1):
        grupy = [[] for _ in range(k)]
        przypisania = [[] for _ in range(k)]
        suma_odleglosci = 0.0

        # Przypisywanie każdego przykładu do najbliższego centroidu
        for i, punkt in enumerate(dane):
            odleglosci = [kwadrat_odleglosci(punkt, c) for c in centroidy]
            min_odleglosc = min(odleglosci)
            indeks_najblizszego = odleglosci.index(min_odleglosc)

            grupy[indeks_najblizszego].append(punkt)
            przypisania[indeks_najblizszego].append(i)
            suma_odleglosci += min_odleglosc

        print(f"Iteracja {iteracja}: {suma_odleglosci:.2f}")

        # algorytm kończy działanie, gdy punkty przestaną zmieniać grupy
        if poprzednie_przypisania is not None and przypisania == poprzednie_przypisania:
            print("Algorytm zbiegł się - przypisania przestały się zmieniać.\n")
            break

        poprzednie_przypisania = przypisania

        # Aktualizacja pozycji centroidów (średnia z przypisanych punktów)
        for i in range(k):
            if grupy[i]:  # Sprawdzamy czy grupa nie jest pusta
                centroidy[i] = [sum(wymiar) / len(grupy[i]) for wymiar in zip(*grupy[i])]

    # EEwaluacja grup
    print("Składy grup i ich czystość:")
    for i in range(k):
        etykiety_w_grupie = [etykiety[idx] for idx in poprzednie_przypisania[i]]
        liczba_elementow = len(etykiety_w_grupie)

        if liczba_elementow == 0:
            print(f"\nGrupa {i + 1}: Pusta")
            continue


        zliczenia = Counter(etykiety_w_grupie)
        print(f"\nGrupa {i + 1} (Liczba elementów: {liczba_elementow}):")

        # Wypisywanie procentowej zawartości każdej z klas
        for etykieta, ilosc in zliczenia.most_common():
            procent = (ilosc / liczba_elementow) * 100
            print(f"  - {etykieta}: {ilosc} szt. ({procent:.2f}%)")


if __name__ == "__main__":
    plik_danych = 'iris.data'

    dane, etykiety = wczytaj_dane(plik_danych)

    # k = int(input("Wybierz liczbę grup (k): "))
    k = 3

    print(f"Uruchamianie algorytmu k-means dla k = {k}\n")
    kmeans(dane, etykiety, k)
