from model_architecture import CNN
from transformedDataset import transformedDataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset, DataLoader
import torch.optim as optim
import time
from datetime import datetime


#spradzenie czy jest dostępne GPU i wymuiszenie pracy na GPU
if not torch.cuda.is_available():
    print("BRAK DOSTĘPNEGO GPU!")
    raise Exception("Exception: Brak dostępnego GPU")

model = CNN().to("cuda")
print(model)

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✓ Model załadowany na GPU!")

"""
VRAM uzyty ~= (wagi sieci * 4 bajty) + (pamięć optrymalizatora) + (aktywacje i gradienty * batch_size) + cuda
    siec CNN ma okolo 8.4mln wag, pamięć optymalizatora (ADAM) to jakies 56MB, aktywacje/zdjecie to ok 7.7mln * 4bajty
    batch_size = (VRAM - wagi - optymalizator - cuda) / aktywacje = (8GB - 8.4mln * 4B - 60MB - 2GB) / (7,7mln * 4B) = 190
    do tego backpropagation, która robi, że aktywacje i gradienty zajmują 2x więcej pamięci, więc max batch_size = 95
"""
seria_zdjec = 64    #najbliższe i najniższe przybliżona potega 2 do 95

#dzielenie datasetu programowo na zbiór treningowy, walidacyjny i testowy
dataset = datasets.ImageFolder(root = "./images")

print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Dataset załadowany!")
print(f"  • Liczba klas: {len(dataset.classes)}")
print(f"  • Łącznie próbek: {len(dataset)}")

train_size = int(0.8 * len(dataset))    #80% do trenowania
val_size = int(0.1 * len(dataset))      #10% do walidacji
test_size = len(dataset) - train_size - val_size    #10% do testu na koniec

print(f"  • Train: {train_size} | Val: {val_size} | Test: {test_size}\n")

#dzielenie losowo na powyższe proporcje
train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])


#transformacja do trenowania - kolejność nie przypadkowa
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),    #wyciecie losowego fragmentu 224x224
    transforms.RandomHorizontalFlip(p=0.5),  #odbija obraz w poziomie z prawdopodobienstwem 50%
    transforms.RandomRotation(degrees=10),   #rotacja o 10stopnii (czasem ludzie robia zdjecia pod katem lub jedzenie tak jest)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),  #symulacja róznego oswietlenia
    transforms.ToTensor(),   #do tensora PyTorch [batch, kanały, wysokość, szerokość]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #normalizacja wg statystyk
])

#transformacja do walidacji i testu
valAndTest_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # zawsze ma zostawić środek
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_subset = transformedDataset(train_subset, transform=train_transform)
val_subset = transformedDataset(val_subset, transform=valAndTest_transform)
test_subset = transformedDataset(test_subset, transform=valAndTest_transform)

#tworzenie DataLoaderów do iterowania po danych w partiach
#shuffle=True, by dane były losowo mieszane w każdej epoce, co pomaga w generalizacji modelu
train_loader = DataLoader(train_subset, batch_size=seria_zdjec, shuffle=True) 
val_loader = DataLoader(val_subset, batch_size=seria_zdjec, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=seria_zdjec, shuffle=False)

#funkcje straty
criterion = torch.nn.CrossEntropyLoss()   #dla klasyfikacji wieloklasowej, standardowo CrossEntropyLoss

"""
optymalziator
    ADAM jest popularny dla CNN, jest optymalniejszy niż SGD, 
            bo adaptacyjnie dostosowuje współczynnik uczenia dla każdego parametru
    lr to learning rate, czyli wielkość kroku, jaki model robi w kierunku minimalizacji straty
             im lr mniejszy tym dokładniejsze, ale spowalnia uczenie
    model.parameters - zeby optymalizował po wszystkich trenowalnych parametrach modelu
    """
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

#ZACZYNAMY OD 20 EPOK NA TEST (DUŻY MODEL), POTEM ZWIEKSYC JAK ZADZIALA
liczba_epok = 20   #liczba epok, czyli ile razy model przejdzie przez cały zbiór treningowy
best_model_accuracy = 0.0   #zmienna do monitorowania najlepszej dokładności walidacyjnej

# === HISTORIA METRYK DO ANALIZY ===
historia_train_loss = []
historia_train_acc = []
historia_val_loss = []
historia_val_acc = []
epochs_bez_poprawy = 0  #licznik epok bez polepszenia walidacyjnej dokładności (do early stopping)

print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Startowanie treningu ({liczba_epok} epok)...\n")
czas_start_treningu = time.time()

for epoka in range(liczba_epok):
    czas_start_epoki = time.time()
    
    #zerowanie statystyk
    train_liczba_probek = 0.0
    train_liczba_poprawnych = 0.0
    train_suma_strat = 0.0

    # === TRENOWANIE ===
    model.train()   #ustawienie modelu w tryb trenowania
    # iteracja po batchach danych z train_loadera
    for img, label in train_loader:
        img, label = img.to("cuda"), label.to("cuda")   #przeniesienie danych na GPU

        optimizer.zero_grad()   #zerowanie gradientów z poprzedniej iteracji
        output = model(img)     #przepuszczenie danych przez model, otrzymanie predykcji
        loss = criterion(output, label)   #obliczenie straty między predykcjami a prawdziwymi etykietami
        loss.backward()         #obliczenie gradientów poprzez backpropagation
        optimizer.step()        #aktualizacja wag modelu na podstawie obliczonych gradientów

        #zbieranie statystyk
        train_liczba_probek += label.size(0)   #zliczanie liczby próbek w batchu
        train_suma_strat += loss.item() * label.size(0)   #sumowanie strat ważonych liczbą próbek (bo zwraca stednia straty na batch)
        predykcje_modelu = output.argmax(dim=1)   #pobranie indeksu klasy z najwyższym prawdopodobieństwem jako predykcję modelu
        train_liczba_poprawnych += (predykcje_modelu == label).sum().item()   #zliczanie poprawnych predykcji

    train_dokladnosc = train_liczba_poprawnych / train_liczba_probek
    train_sr_loss = train_suma_strat / train_liczba_probek

    # === WALIDACJA NA KONIEC EPOKI===
    val_suma_strat = 0.0
    val_liczba_poprawnych = 0.0
    val_samples = 0

    model.eval()    #ustawienie modelu w tryb ewaluacji
    with torch.no_grad():   #wyłączenie obliczania gradientów, bo nie potrzebujemy ich do walidacji
        #iteracja po batchach danych z val_loadera
        for img, label in val_loader:
            img, label = img.to("cuda"), label.to("cuda")

            output = model(img)    #przepuszczenie danych przez model, otrzymanie predykcji
            val_loss = criterion(output, label)     #obliczenie straty walidacyjnej

            #zbieranie statystyk walidacyjnych
            val_suma_strat += val_loss.item() * label.size(0)
            predykcje_modelu = output.argmax(dim=1)
            val_liczba_poprawnych += (predykcje_modelu == label).sum().item()
            val_samples += label.size(0)
    
    val_dokladnosc = val_liczba_poprawnych / val_samples
    val_sr_loss = val_suma_strat / val_samples
    val_bledy = 1 - val_dokladnosc

    # === ZAPISYWANIE NAJLEPSZEGO MODELU ===
    if val_dokladnosc > best_model_accuracy:
        best_model_accuracy = val_dokladnosc
        torch.save(model.state_dict(), "best_model.pth") #zapisanie wag najlepszego modelu do pliku
        print(f"  ★ Nowy najlepszy model! Val Acc: {val_dokladnosc:.4f} ↑")
        epochs_bez_poprawy = 0  #resetowanie licznika, bo była poprawa
    else:
        epochs_bez_poprawy += 1  #inkrementacja licznika bez polepszenia

    # === DODANIE DO HISTORII ===
    historia_train_loss.append(train_sr_loss)
    historia_train_acc.append(train_dokladnosc)
    historia_val_loss.append(val_sr_loss)
    historia_val_acc.append(val_dokladnosc)

    czas_epoki = time.time() - czas_start_epoki
    
    print(
        f"[Epoka {epoka+1}/{liczba_epok}] ({czas_epoki:.1f}s) "
        f"| Train: Loss={train_sr_loss:.4f} Acc={train_dokladnosc:.4f} "
        f"| Val: Loss={val_sr_loss:.4f} Acc={val_dokladnosc:.4f}"
    )

    # === EARLY STOPPING ===
    patience = 5  #liczba epok bez polepszenia, po której zatrzymujemy trening
    if epochs_bez_poprawy >= patience:
        print(f"\n⚠️  Early stopping! Brak polepszenia przez {patience} epok.")
        break


# === PODSUMOWANIE TRENINGU ===
czas_calosci = time.time() - czas_start_treningu
print(f"\n{'='*70}")
print(f"✓ Trening zakończony!")
print(f"  • Czas całkowity: {czas_calosci/60:.1f} minut ({czas_calosci:.0f}s)")
print(f"  • Najlepsza Val Accuracy: {best_model_accuracy:.4f}")
print(f"  • Liczba epok: {len(historia_train_loss)}")
print(f"{'='*70}\n")

# === ZAŁADOWANIE NAJLEPSZEGO MODELU ===
print("Ładowanie najlepszego modelu...")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# === EWALUACJA NA ZBIORZE TESTOWYM ===
print("Ewaluacja na zbiorze testowym...\n")
czas_start_testu = time.time()
test_liczba_poprawnych = 0.0
test_suma_strat = 0.0
test_samples = 0

with torch.no_grad():
    for img, label in test_loader:
        img, label = img.to("cuda"), label.to("cuda")
        
        output = model(img)
        test_loss = criterion(output, label)
        
        test_suma_strat += test_loss.item() * label.size(0)
        predykcje_modelu = output.argmax(dim=1)
        test_liczba_poprawnych += (predykcje_modelu == label).sum().item()
        test_samples += label.size(0)

test_dokladnosc = test_liczba_poprawnych / test_samples
test_sr_loss = test_suma_strat / test_samples

print(f"\n{'='*70}")
print(f"TEST RESULTS (Czas: {time.time() - czas_start_testu:.1f}s)")
print(f"{'='*70}")
print(f"  • Test Loss: {test_sr_loss:.4f}")
print(f"  • Test Accuracy: {test_dokladnosc:.4f} ({int(test_liczba_poprawnych)}/{int(test_samples)})")
print(f"  • Test Error: {1-test_dokladnosc:.4f}")
print(f"{'='*70}")


