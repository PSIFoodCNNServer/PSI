import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.datasets as datasets
import os
from model_architecture import CNN 
from getFood import load_food_base

@st.cache_data
def load_classes():
    with open("classes.txt", "r") as f:
        # line.strip() usuwa białe znaki i entery; warunek 'if line.strip()' odrzuca puste linie
        classes = [line.strip().replace("_", " ").title() for line in f.readlines() if line.strip()]
    return classes

food_classes = load_classes()
food_database = load_food_base()
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
            food_data = None
            display_name = f"Nieznane danie ID: {class_id}"

            try:
                potrawa_name = food_classes[class_id]
            except IndexError:
                potrawa_name = f"Nieznane danie, ID: {class_id}"

            if class_id < len(food_classes):
                food_key = food_classes[class_id] # np. "pizza"
                display_name = food_key.replace("_", " ").title() # np. "Pizza"
                
                # Pobieramy obiekt z bazy (kluczem jest mała nazwa folderu, np. "pizza")
                food_data = food_database.get(food_key.lower())
                if food_data is None:
                    food_data = food_database.get(food_key) 

            st.success(f"### Wynik: {potrawa_name}")
            st.metric("Pewność modelu", f"{confidence:.2f}%")
            
            st.info(f"Dla potrawy {potrawa_name} przygotowaliśmy specjalny przepis, który wyświetli się tutaj.")

            if food_data is not None:
                st.markdown("---")
                st.subheader("Wartości odżywcze (w porcji)")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Kalorie", f"{food_data.getKcal()} kcal")
                col2.metric("Białko", f"{food_data.getProtein()}g")
                col3.metric("Węglowodany", f"{food_data.getCarbs()}g")
                col4.metric("Tłuszcze", f"{food_data.getFat()}g")
                
                st.markdown("---")
                st.subheader(f"📖 Przepis na {food_data.getName()}")
                
                # Używamy metody z literówką kolegi: getReceipe()
                recipe_text = food_data.getReceipe()
                if recipe_text:
                    st.write(recipe_text)
                else:
                    st.info("Brak opisu przepisu w bazie dla tej potrawy.")
            else:
                st.warning(f"Rozpoznano klucz '{food_classes[class_id]}', ale brakuje go w pliku foodBase.txt.")

st.sidebar.markdown("### Statystyki Modelu")
st.sidebar.text("Architektura: Custom CNN")
st.sidebar.text("Liczba wag: ~8.4 mln")
st.sidebar.text("Docelowe RF: 158")


