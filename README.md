# Rozpoznawanie Potraw (CNN Food Recognizer)

Aplikacja webowa oparta na głębokiej sieci konwolucyjnej (CNN) w PyTorch oraz interfejsie Streamlit, potrafiąca rozpoznać 101 różnych rodzajów potraw z dokładnością testową na poziomie ~78%.

## WAŻNE: POBIERANIE WAG MODELU (GIT LFS)

Ten projekt wykorzystuje **Git Large File Storage (LFS)** do przechowywania wag wyuczonego modelu (`best_model.pth` ~32 MB). 

**NIE POBIERAJ tego repozytorium używając zielonego przycisku "Download ZIP" na GitHubie.** 
GitHub w plikach ZIP udostępnia jedynie małe (130-bajtowe) wskaźniki tekstowe zamiast fizycznych, dużych plików. Próba uruchomienia aplikacji z pliku ZIP zakończy się krytycznym błędem PyTorch (`UnpicklingError`).

## Instrukcja instalacji i uruchomienia (Krok po Kroku)

### 1. Wymagania wstępne
Upewnij się, że masz zainstalowane na swoim komputerze:
* [Python 3.8+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)
* **[Git LFS](https://git-lfs.github.com/)** (Kluczowe!)

### 2. Aktywacja Git LFS i klonowanie repozytorium
Otwórz terminal (lub wiersz poleceń) i wklej poniższe komendy:

# 1. Sklonuj repozytorium (LFS automatycznie dociągnie plik .pth)

        git clone [https://github.com/TWOJA_NAZWA/TWOJE_REPO.git]

# 2. Wejdź do folderu z projektem

        cd TWOJE_REPO

## Uruchomienie w Dockerze (wymagne docker desktop lub docker)

Najprostsza opcja to uruchomienie całego projektu w kontenerze (terminal bash):

        docker compose up --build

Po starcie aplikacja będzie dostępna pod wskazanym w terminalu adresem.

## Uruchomienie lokalne

Jeżeli chcesz odpalić projekt bez Dockera:

        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        streamlit run app.py
