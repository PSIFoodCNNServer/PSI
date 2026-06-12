import os
import shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from model_architecture import CNN
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


class ImageFolderWithPaths(ImageFolder):
	#Klasa zawierająca obraz oraz jego ściżke i etykiete, 
    #będzie użyta do mapowania indexu na faktyczny obraz w celu przedstawienia realnych wyników
	
	def __getitem__(self, index):
		#wywołuje getidem z klasy rodzica, pobiera obraz (samblple) i etykiete (target)
		sample, target = super(ImageFolderWithPaths, self).__getitem__(index) 
		#pobiera sciezke do konkretnego
		path, _ = self.samples[index]
		return sample, target, path


#sprawdza czy katalog istnieje, jeżeli nie to go tworzy
def ensure_dir(d):
	if not os.path.exists(d):
		#tworzenie katalogu gdy nie istnieje
		os.makedirs(d, exist_ok=True)


def evaluate(test_data_loader, CNN_model, device, criterion_func, class_names_list, max_most_missclassifiedd_imgs=20):
	CNN_model.eval()  #przełączenie modelu w tryb ewaluadcji
	all_preds = []  #wszyskie predykcje dla kazdego obrazu
	all_labels = [] #etykiety obrazu
	all_predicted_confidences = []  #pewnosc predykcji wszyskich
	from_all_test_loss_sum = 0.0 #suma losow z wszyskich batchy testowych
	n_samples = 0   #ilosc wszyskich przetestowanych obrazow
	misclassified = []  # lista blednych klasywikacji w tuplach (pewnosc predykcji, prawdziwa klasa, klasa z predytkcji, path)

	with torch.no_grad():
		#pętla batchami po danych testowych
		for imgs, labels, paths in test_data_loader:
			#przenoszenie obrazow i etykiet na GPU
			imgs = imgs.to(device)
			labels = labels.to(device) 
			#przepuszcza batcha przez model i zapisuje wyniki 
			outputs = CNN_model(imgs)  
			#liczy bład dla aktualnego batcha
			loss = criterion_func(outputs, labels)
			#zapisuje loss z batcha do statystyk
			from_all_test_loss_sum += loss.item() * labels.size(0)
			#zmienia wyniki modelu (logits) na rozklad prawdopodobienstwa po klasach jedzenia
			probs = F.softmax(outputs, dim=1)
			#biuerze z rozkladu prawdopodobienstwa index najwiekszej wartosci i ta wartosc (przewidziana klasa i pewnosc)
			confs, preds = probs.max(dim=1)
			
            #dopisuje przewidziane klasy z aktualnego batcha do listy wszystkich predykcji
			all_preds.extend(preds.cpu().numpy().tolist())
			#dopisuje prawdziwe etykiety z aktualnego batcha do listy wszystkich etykiet
			all_labels.extend(labels.cpu().numpy().tolist())
			#dopisuje pewnosci (najwyzsze prawdopodobienstwa) dla kazdej predykcji
			all_predicted_confidences.extend(confs.cpu().numpy().tolist())
            #zwieksza licznik o liczbe probek w aktualnym batchu
			n_samples += labels.size(0)

			# zbieranie zle przewidzianych, iteracja po sciezkach obrazow z batcha
			for i in range(len(paths)):
				#jesli przewidział zle sciezka z klasy nie pokrywa sie z predykcja
				if preds[i].item() != labels[i].item():
					#dodaj do listy zlej predykcji: pewnosc, faktyczna klasa, przewidziana klasa, sciezka
					misclassified.append((confs[i].item(), labels[i].item(), preds[i].item(), paths[i]))

	# koncowe metryki-podsumowanie i rzutowanie do tabel
	all_preds = np.array(all_preds) #
	all_labels = np.array(all_labels)
	all_predicted_confidences = np.array(all_predicted_confidences)
	test_loss = from_all_test_loss_sum / n_samples
	accuracy = (all_preds == all_labels).mean() #srednia z odsetku poprawnych predykcji

	# macierz pomylek (jak sie myli dla kazdej klasy - co rozpoznaje zamiast niej)
	num_classes = len(class_names_list)
	confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
	for t, p in zip(all_labels, all_preds):
		confusion_matrix[t, p] += 1

	# dla kazdej klasy: precision/recall(czulosc)/f1
	#precyzja - kiedy model mówi 'tak', to jak bardzo mogę mu ufać
	#recall - ile faktycznych, prawdziwych przypadków z całego zbioru model zdołał wyłapać
	#F1 - ogolny bilans miedzy powyzszymi
	
	tp = np.diag(confusion_matrix).astype(float)    #przekatna na float - celne predykcje
	fp = confusion_matrix.sum(axis=0) - tp  #sumowanie kolumnami (ile razy model przewidzial konkretna klase)
	fn = confusion_matrix.sum(axis=1) - tp  #sumowanie wierszami (ile faktycnzie razy mial ja przewidziec)
	with np.errstate(divide='ignore', invalid='ignore'):    #wylaczneiu warninogw przy dzieleniu przez 0 itp
		precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)  #oblicza precyzje dla kazdej klasy
		recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0) #oblicze recall
		f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)   #oblciza f1

	# zapisanie macierzy pomylek do PNG
	ensure_dir('evaluation')
	plt.figure(figsize=(10, 10))
	plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(class_names_list))
	
	if len(class_names_list) <= 30:
		plt.xticks(tick_marks, class_names_list, rotation=90)
		plt.yticks(tick_marks, class_names_list)
	else:
		plt.xticks([])
		plt.yticks([])
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.savefig('evaluation/confusion_matrix.png', dpi=200)
	plt.close()

	# zapisanie raportu skutecznosci klasyfikacji (jakosc modelu)
	with open('evaluation/classification_report.txt', 'w') as f:
		f.write(f"Evaluation time: {datetime.now().isoformat()}\n")
		f.write(f"Samples: {n_samples}\n")
		f.write(f"Test Loss: {test_loss:.4f}\n")
		f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
		f.write("Class, Precision, Recall, F1, Support\n")
		for i, name in enumerate(class_names_list):
			support = int(confusion_matrix[i].sum())
			f.write(f"{name}, {precision[i]:.4f}, {recall[i]:.4f}, {f1[i]:.4f}, {support}\n")

	# zapisanie najbardziej pewnych blednych odpowiedzi
	misclassified.sort(key=lambda x: x[0], reverse=True)
	top_mis = misclassified[:max_most_missclassifiedd_imgs]
	mis_dir = 'evaluation/misclassified'
	ensure_dir(mis_dir)
	for idx, (conf, true, pred, path) in enumerate(top_mis):
		# kopia orginalnej sciezki z prefixem true/pred/conf
		base = os.path.basename(path)
		new_name = f"{idx+1:02d}_true-{class_names_list[true]}_pred-{class_names_list[pred]}_conf-{conf:.3f}_{base}"
		shutil.copy(path, os.path.join(mis_dir, new_name))

	#zapisanie macierzy pomylek do csv
	np.savetxt('evaluation/confusion_matrix.csv', confusion_matrix, fmt='%d', delimiter=',')

	return {
		'n_samples': n_samples,
		'test_loss': test_loss,
		'accuracy': accuracy,
		'precision': precision.tolist(),
		'recall': recall.tolist(),
		'f1': f1.tolist(),
		'confusion_matrix': confusion_matrix.tolist(),
	}


def main():
	if not torch.cuda.is_available():
		raise RuntimeError("BŁĄD: Karta graficzna nie jest dostępna!.")

	device = 'cuda' 
	print(f"Using device: {device}")


	#Transormacje dla danych testowych, takie same jak dla treningowych
	val_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	#ładowanie danych testowych
	dataset = ImageFolderWithPaths(root='images', transform=val_transform)
	class_names = dataset.classes
	print(f"Loaded dataset with {len(dataset)} samples and {len(class_names)} classes")

	#tworzymy podziały na zbiór treningowy, walidacyjny i testowy 
	total = len(dataset)
	test_size = int(0.1 * total)
	val_size = int(0.1 * total)
	train_size = total - val_size - test_size
	generator = torch.Generator().manual_seed(42)
	train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
	
	#dataloader dla zbioru testowego, tworzy batch po 64 obrazy
	test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

	#wczytanie modelu
	model = CNN()
	checkpoint = 'best_model.pth'
	if os.path.exists(checkpoint):
		model.load_state_dict(torch.load(checkpoint, map_location=device))
		print(f"Loaded weights from {checkpoint}")
	else:
		print(f"Checkpoint {checkpoint} not found. Exiting.")
		return

	model.to(device)
	criterion = torch.nn.CrossEntropyLoss()

	#zestawienie wyników
	results = evaluate(test_loader, model, device, criterion, class_names, max_most_missclassifiedd_imgs=20)
	print("Evaluation finished:")
	print(f"  Samples: {results['n_samples']}")
	print(f"  Test Loss: {results['test_loss']:.4f}")
	print(f"  Test Accuracy: {results['accuracy']:.4f}")
	print("Results saved to ./evaluation/")


if __name__ == '__main__':
	main()
