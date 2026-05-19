import random
from collections import Counter


def wczytaj_dane(nazwa_pliku):
    dane = []
    etykiety = []
    with open(nazwa_pliku, 'r') as plik:
        for linia in plik:
            linia = linia.strip()
            if not linia:  # Pomiń puste linie
                continue
            elementy = linia.split(',')
            # Pierwsze cztery kolumny to atrybuty liczbowe (cechy)
            cechy = [float(x) for x in elementy[:-1]]
            # Ostatnia kolumna to ukryta klasa decyzyjna
            klasa = elementy[-1]

            dane.append(cechy)
            etykiety.append(klasa)
    return dane, etykiety


def kwadrat_odleglosci(p1, p2):
    # Suma kwadratów różnic między współrzędnymi.
    # To ta wartość ściśle maleje w klasycznym k-means po każdej iteracji.
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


def kmeans(dane, etykiety, k, max_iter=100):
    # Inicjalizacja: losowy wybór k punktów początkowych jako centroidy
    random.seed(0)  # Ustawienie ziarna losowości dla powtarzalności wyników
    indeksy = random.sample(range(len(dane)), k)
    centroidy = [dane[i] for i in indeksy]

    poprzednie_przypisania = None

    for iteracja in range(1, max_iter + 1):
        grupy = [[] for _ in range(k)]
        przypisania = [[] for _ in range(k)]
        suma_odleglosci = 0.0

        # Etap 1: Przypisywanie każdego przykładu do najbliższego centroidu
        for i, punkt in enumerate(dane):
            odleglosci = [kwadrat_odleglosci(punkt, c) for c in centroidy]
            min_odleglosc = min(odleglosci)
            indeks_najblizszego = odleglosci.index(min_odleglosc)

            grupy[indeks_najblizszego].append(punkt)
            przypisania[indeks_najblizszego].append(i)
            suma_odleglosci += min_odleglosc

        print(f"Iteracja {iteracja}: {suma_odleglosci:.2f}")

        # Warunek stopu: algorytm kończy działanie, gdy punkty przestaną zmieniać grupy
        if poprzednie_przypisania is not None and przypisania == poprzednie_przypisania:
            print("Algorytm zbiegł się - przypisania przestały się zmieniać.\n")
            break

        poprzednie_przypisania = przypisania

        # Etap 2: Aktualizacja pozycji centroidów (średnia z przypisanych punktów)
        for i in range(k):
            if grupy[i]:  # Sprawdzamy czy grupa nie jest pusta
                centroidy[i] = [sum(wymiar) / len(grupy[i]) for wymiar in zip(*grupy[i])]

    # Etap 3: Ewaluacja grup (czystość, składy grup na podstawie odciętych na początku etykiet)
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
    # Zakładamy, że plik iris.data znajduje się w tym samym folderze co skrypt
    plik_danych = 'iris.data'

    try:
        dane, etykiety = wczytaj_dane(plik_danych)

        # Odkomentuj poniższą linię, jeśli program ma prosić użytkownika o wpisanie liczby 'k'
        # k = int(input("Wybierz liczbę grup (k): "))
        k = 3  # Ustawione domyślnie na 3, czyli rzeczywistą liczbę gatunków irysów

        print(f"Uruchamianie algorytmu k-means dla k = {k}\n")
        kmeans(dane, etykiety, k)

    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{plik_danych}'. Upewnij się, że jest w tym samym folderze.")