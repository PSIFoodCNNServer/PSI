import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.datasets as datasets
import os
from model_architecture import CNN 

@st.cache_data
def load_classes():
    with open("classes.txt", "r") as f:
        # line.strip() usuwa białe znaki i entery; warunek 'if line.strip()' odrzuca puste linie
        classes = [line.strip().replace("_", " ").title() for line in f.readlines() if line.strip()]
    return classes

food_classes = load_classes()
#Interfejs Streamlit
st.set_page_config(page_title="Food Detector")
#test test test
@st.cache_resource
def load_model():
    # Inicjalizujemy klasę CNN 
    model = CNN()
    # Ładujemy wagi wygenerowane przez skrypt treningowy
    # Wybierz urządzenie (jeżeli jest CUDA użyj jej, w przeciwnym razie CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # Tryb ewaluacji



    return model

# Przygotowanie danych
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Strona www
st.title("Rozpoznawanie Potraw")
st.markdown("Wgraj zdjęcie.")

file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "jpeg", "png"])

if file is not None:
    # Wyświetlamy obrazek użytkownika
    img = Image.open(file).convert('RGB')
    st.image(img, caption="Przesłane zdjęcie", use_container_width=True)

    if st.button("Analizuj potrawę"):
        with st.spinner("Sieć przetwarza dane..."):
            # 1. Ładowanie modelu
            net = load_model()
            
            # 2. Przygotowanie zdjęcia
            img_tensor = preprocess(img).unsqueeze(0) 
            # Przenieś tensor na to samo urządzenie, na którym jest model
            device = next(net.parameters()).device
            img_tensor = img_tensor.to(device)
            
            # 3. Predykcja
            with torch.no_grad():
                outputs = net(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                prob, index = torch.max(probabilities, 1)

            # 4. Wyświetlanie wyników
            class_id = index.item()
            confidence = prob.item() * 100

            try:
                potrawa_name = food_classes[class_id]
            except IndexError:
                potrawa_name = f"Nieznane danie, ID: {class_id}"

            st.success(f"### Wynik: {potrawa_name}")
            st.metric("Pewność modelu", f"{confidence:.2f}%")
            
            st.info(f"Dla klasy {class_id} przygotowaliśmy specjalny przepis, który wyświetli się tutaj.")

st.sidebar.markdown("### Statystyki Modelu")
st.sidebar.text("Architektura: Custom CNN")
st.sidebar.text("Liczba wag: ~8.4 mln")
st.sidebar.text("Docelowe RF: 158")


