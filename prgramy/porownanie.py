import os
import cv2


def get_image_files(folder, extensions={'.png', '.jpg', '.jpeg', '.bmp', '.gif'}):
    """
    Funkcja przeszukująca podany folder i zwracająca listę nazw plików,
    których rozszerzenia znajdują się w podanym zbiorze.
    """
    return [file for file in os.listdir(folder)
            if os.path.splitext(file)[1].lower() in extensions]


def compare_images_cv(image1_path, image2_path, diff_save_path=None):
    """
    Funkcja wczytuje dwa obrazy przy użyciu OpenCV i porównuje je.

    Parametry:
      image1_path - ścieżka do pierwszego obrazu,
      image2_path - ścieżka do drugiego obrazu,
      diff_save_path - opcjonalna ścieżka, gdzie zapisany zostanie obraz różnic.

    Zwraca normę różnicy (norma L1) między obrazami.
    """
    # Wczytanie obrazów
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print(f"Błąd wczytywania obrazów: {image1_path} lub {image2_path}")
        return None

    # Sprawdzanie, czy obrazy mają ten sam rozmiar
    if img1.shape != img2.shape:
        print(f"Obrazy {image1_path} i {image2_path} mają różne rozmiary i nie mogą być porównane.")
        return None

    # Obliczenie różnicy pomiędzy obrazami
    diff_img = cv2.absdiff(img1, img2)

    # Obliczenie normy L1 różnicy
    norm_value = cv2.norm(img1, img2, cv2.NORM_L1)

    # Jeśli podana została ścieżka zapisu, dokonujemy normalizacji
    # obrazu różnic i próbujemy zapisać plik z kontrolą wyniku zapisu.
    if diff_save_path:
        # Normalizacja różnic, aby były widoczne (skalowanie do zakresu 0-255)
        diff_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
        # Próba zapisu obrazu różnic
        if not cv2.imwrite(diff_save_path, diff_img):
            print(f"Nie udało się zapisać obrazu różnic do pliku: {diff_save_path}")
        else:
            print(f"Obraz różnic zapisano w: {os.path.abspath(diff_save_path)}")

    return norm_value


def main(folder1, folder2, diff_output_folder="output_diff", threshold=0):
    """
    Funkcja główna programu, która:
      - tworzy folder wyjściowy, jeśli nie istnieje,
      - pobiera listę plików graficznych w obu folderach,
      - porównuje obrazy o tych samych nazwach,
      - wyświetla wynik porównania i zapisuje obraz różnic (jeśli obrazy się różnią).

    Parametr threshold (domyślnie 0) określa próg różnicy, poniżej którego obrazy uznawane są za identyczne.
    """
    if not os.path.exists(diff_output_folder):
        os.makedirs(diff_output_folder)

    files_folder1 = set(get_image_files(folder1))
    files_folder2 = set(get_image_files(folder2))
    common_files = files_folder1.intersection(files_folder2)

    if not common_files:
        print("Nie znaleziono wspólnych plików graficznych do porównania.")
        return

    for file in common_files:
        path1 = os.path.join(folder1, file)
        path2 = os.path.join(folder2, file)
        diff_path = os.path.join(diff_output_folder, f"diff_{file}")

        diff_value = compare_images_cv(path1, path2, diff_save_path=diff_path)
        if diff_value is None:
            continue

        if diff_value <= threshold:
            print(f"Obrazy '{file}' są identyczne (norma różnicy: {diff_value}).")
        else:
            print(f"Obrazy '{file}' różnią się (norma różnicy: {diff_value}). Różnice zapisano w: {diff_path}")


if __name__ == "__main__":
    # Zamień poniższe ścieżki na rzeczywiste lokalizacje folderów z obrazami
    folder1 = "output_contours"
    folder2 = "../output_contours"

    # Uruchomienie programu z progiem różnicy 0 - obrazy muszą być identyczne,
    # by zostały uznane za takie.
    main(folder1, folder2, diff_output_folder="roznice", threshold=0)
