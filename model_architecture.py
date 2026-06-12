import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()     #odwołenie do konstruktora klasy bazowej nn.Module
        """
        Ilość warstw jest oblcizana ze wzoru na stride
        celujemy w RF = 158, czyli 70% pokrycia obrazu 224x224

        2 osatnie 2 warstwy konwolucyjne bez poolingu
            chodzi o to by zwiększyć RF do 158, jednocześnie nie zwiekszając nadmiarowo ilości kanałów, 
            by nie przepłacać obliczeniowo oraz uniknać przeuczenia modelu.
        """

        #1 ===== warstwa ======
        """operacja konwulcji: 
            in_channels - liczba kanałów wejściowych
            out_channels - liczba kanałów wyjściowych (liczba filtrów)
            kernel_size - rozmiar filtra nxn
            stride - krok przesuwania filtra (domyślnie 1)
            padding - ilość pikseli dodawanych dookoła obrazu by nie tracić informacje na brzegach (domyślnie 0)
        """
        #1 warstwa konwolucyjna + pooling: in_channels = 3 przez RGB
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.pool1  = nn.MaxPool2d(kernel_size = 2, stride = 2)   #operacja max poolingu

        #2 warstwa konwolucyjna + pooling
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.pool2  = nn.MaxPool2d(kernel_size = 2, stride = 2)   #operacja max poolingu

        #3 warstwa konwolucyjna + pooling
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.pool3  = nn.MaxPool2d(kernel_size = 2, stride = 2)   #operacja max poolingu

        #4 warstwa konwolucyjna + pooling
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.pool4  = nn.MaxPool2d(kernel_size = 2, stride = 2)   #operacja max poolingu

        #5 warstwa konwolucyjna + pooling
        self.conv5 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.pool5  = nn.MaxPool2d(kernel_size = 2, stride = 2)   #operacja max poolingu
    
        self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)

        #ograniczenie cech wrzucanych do linear - aby uniknac przeuczenia i bardzo agresywnogo spłaszczenia
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))       

        self.fc1 = nn.Linear(in_features = 512 * 4 * 4, out_features = 256) 
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features = 256, out_features = 101)     #mamy 101 klas

   
    def forward(self, x):
        x = F.relu(self.conv1(x))   #funkcja aktywacji ReLU - 1
        x = self.pool1(x)           #operacja poolingu

        x = F.relu(self.conv2(x))   #funkcja aktywacji ReLU - 2
        x = self.pool2(x)           #operacja poolingu

        x = F.relu(self.conv3(x))   #funkcja aktywacji ReLU - 3
        x = self.pool3(x)           #operacja poolingu

        x = F.relu(self.conv4(x))   #funkcja aktywacji ReLU - 4
        x = self.pool4(x)           #operacja poolingu

        x = F.relu(self.conv5(x))   #funkcja aktywacji ReLU - 5
        x = self.pool5(x)           #operacja poolingu

        x = F.relu(self.conv6(x))   #funkcja aktywacji ReLU - 6 (bez poolingu, by zwiększyć RF do 158)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)   #flatten - spłaszcza tensor tutaj z wymiaru 3d do 1d

        x = F.relu(self.fc1(x))              #łączenie cech w odpowiedzi dla klas

        #dropout losowo zeruje cześć (tutaj połowe) aktywacji w wywołanej warstwie w trakcie treningu
        #   dzieki czemu tak jakby trenujemy wiele mniejszych sieci o takiej samej pojemnosci jak jedna wielka
        #   potem te wszyskie sieci razem podejmują decyzję
        x = self.dropout(x)          
        x = self.fc2(x) 
        
        return x