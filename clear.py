import os
import shutil


def pliki(folder):
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

def folder(folder):
    shutil.rmtree(folder)


def clear(keras=None, output=None, trainingmasks=None, trainingpictures=None, all=None):
    if all:
        if keras is None:
            keras=True
        if output is None:
            output=True
        if trainingmasks is None:
            trainingmasks=True
        if trainingpictures is None:
            trainingpictures=True

    if keras:
        tu wstaw kod usuwający pliki z rozszerzeniem keras
    if output:
        shutil.rmtree('output_contours')
    if trainingmasks:
        pliki('training/mask')
    if trainingpictures:
        pliki('training/original')


if __name__=="__main__":
    clear(keras=True, output=True)