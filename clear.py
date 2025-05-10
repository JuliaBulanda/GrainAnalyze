import os


def clear(folder):
    """
    Usuwa wszystkie pliki we wskazanym folderze.

    :param folder: Ścieżka do folderu, z którego chcesz usunąć pliki.
    """
    try:
        # Sprawdzenie, czy folder istnieje
        if not os.path.exists(folder):
            print(f"Folder {folder} nie istnieje.")
            return

        # Iteracja przez wszystkie pliki w folderze
        for nazwa_pliku in os.listdir(folder):
            sciezka_pliku = os.path.join(folder, nazwa_pliku)
            # Sprawdzenie, czy to jest plik
            if os.path.isfile(sciezka_pliku):
                os.remove(sciezka_pliku)
                print(f"Usunięto plik: {sciezka_pliku}")
            else:
                print(f"Pominięto: {sciezka_pliku} (nie jest plikiem)")

        print("Proces usuwania plików zakończony.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")


# Przykład użycia
folder_docelowy = r"C:\ścieżka\do\twojego\folderu"
usun_pliki(folder_docelowy)
