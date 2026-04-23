"""
import streamlit as st
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model_architecture import CNN 

#Interfejs Streamlit
st.set_page_config(page_title="Food AI Detector", page_icon="🥘")

@st.cache_resource
def load_model():
    # Inicjalizujemy klasę CNN 
    model = CNN()
    # Ładujemy wagi wygenerowane przez skrypt treningowy
    state_dict = torch.load("best_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
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
            
            # 3. Predykcja
            with torch.no_grad():
                outputs = net(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                prob, index = torch.max(probabilities, 1)

            # 4. Wyświetlanie wyników
            class_id = index.item()
            confidence = prob.item() * 100

            st.success(f"### Wynik: Klasa nr {class_id}")
            st.metric("Pewność modelu", f"{confidence:.2f}%")
            
            st.info(f"Dla klasy {class_id} przygotowaliśmy specjalny przepis, który wyświetli się tutaj.")

st.sidebar.markdown("### Statystyki Modelu")
st.sidebar.text("Architektura: Custom CNN")
st.sidebar.text("Liczba wag: ~8.4 mln")
st.sidebar.text("Docelowe RF: 158")
"""
import streamlit as st
import time
import random
from PIL import Image

# --- KONFIGURACJA ---
st.set_page_config(page_title="TEST - Food AI", page_icon="🧪")

# Udawana lista klas (pobierz ją od kolegi, jak tylko będzie ją miał)
MOCK_CLASSES = ["Pizza", "Burger", "Sushi", "Pad Thai", "Spaghetti Carbonara", "Steak"]

# --- INTERFEJS ---
st.title("🧪 TEST Interfejsu (Tryb bez AI)")
st.info("Obecnie aplikacja działa w trybie testowym. Nie wymaga modelu 'best_model.pth'.")

file = st.file_uploader("Wgraj dowolne zdjęcie, by sprawdzić układ strony...", type=["jpg", "png"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Podgląd zdjęcia", use_container_width=True)

    if st.button("🚀 Uruchom testową analizę"):
        # Symulacja paska postępu
        with st.spinner("Symulacja pracy sieci neuronowej..."):
            time.sleep(2) # Udajemy, że model liczy (2 sekundy)
            
            # Losujemy wynik, żeby zobaczyć jak wyglądają komunikaty
            random_food = random.choice(MOCK_CLASSES)
            random_conf = random.uniform(85, 99.9)

        # Wyświetlanie wyników testowych
        st.success(f"### Rozpoznano: {random_food}")
        st.metric("Pewność (symulacja)", f"{random_conf:.2f}%")
        
        # --- TEST SEKCJI PRZEPISU ---
        st.divider()
        st.subheader(f"📖 Przykładowy przepis na {random_food}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Składniki:**")
            st.write("- Składnik testowy A\n- Składnik testowy B\n- Tajny przyprawa X")
        with col2:
            st.markdown("**Sposób przygotowania:**")
            st.write("1. Podgrzej atmosferę na prezentacji.\n2. Pokaż działający interfejs.\n3. Zbierz gratulacje od prowadzącego.")

# --- SIDEBAR DO TESTÓW ---
st.sidebar.header("Panel deweloperski")
st.sidebar.write("Tu możesz dodać suwaki do testowania różnych ustawień wyglądu.")
theme = st.sidebar.selectbox("Zmień motyw (wizualnie)", ["Jasny", "Ciemny", "Systemowy"])