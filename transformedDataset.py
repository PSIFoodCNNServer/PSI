class transformedDataset:
    #konstruktor przyjmuje subset danych i transformacje, która ma być zastosowana do obrazów
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    #zwraca długość subsetu
    def __len__(self):
        return len(self.subset)

    #pobiera obraz i etykietę z subsetu, stosuje transformację (jeśli jest) i zwraca je
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label