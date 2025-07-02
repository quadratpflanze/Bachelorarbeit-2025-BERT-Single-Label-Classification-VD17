# Bachelorthesis-2025-BERT-Single-Label-Classification-VD17
1.	Trainingsdaten1600-1650-gekuerzt1306.csv 
2.	Testdaten1651-1700-gekuerzt1306.csv
3.	Gattungsbegriffe0-272.txt
4.	FullScriptVD17.py
5.	BAbertTrainingAndTest.py
6.	testingProbabilitiesBertAllLabels.py
7.	1BAbert2e-5.txt
8.	1BAbert3e-5.txt
9.	1BAbert4e-5.txt
10.	1BAbert5e-5.txt
11.	2BAbert2e-5-3ep-Probabilities.txt
12.	2BAbert5e-5-2ep-Probabilities.txt
13.	4BAbert2e-5-3ep-weightd-1.txt
14.	4BAbert2e-5-3ep-weightd-2.txt
15.	4BAbert2e-5-3ep-weightd-3.txt
16.	4BAbert2e-5-3ep-weightd-4.txt
17.	4BAbert2e-5-3ep-weightd-5.txt
18.	4BAbert5e-5-2ep-weightd-1.txt
19.	4BAbert5e-5-2ep-weightd-2.txt
20.	4BAbert5e-5-2ep-weightd-3.txt
21.	4BAbert5e-5-2ep-weightd-4.txt
22.	4BAbert5e-5-2ep-weightd-5.txt
<br>
1. und 2.: Die in dieser Arbeit verwendeten Trainings- und Testdaten als CSV-Datei.<br>
3.: Die zum Label-Mapping verwendete Textdatei mit allen Gattungsbegriffen der AAD.<br>
4.: Das Skript zum Download der Dateien über die OAI-Schnittstelle der Staatsbibliothek zu Berlin und zur Konvertierung zu den CSV-Dateien. Läuft mit Python 3.11.<br>
5.: Der verwendete Code, für das BERT-Finetuning. Absolute Pfade zu den CSV-Dateien müssen vom Anwender angepasst werden, ebenso alle Trainingsparameter. Läuft mit Python 3.10. Verwendete Versionen der kritischen Pakete:<br>
	huggingface-hub: 0.33.0<br>
	datasets: 3.6.0<br>
	evaluate: 0.4.3<br>
	tokenizers: 0.21.1<br>
	transformers: 4.51.3<br>
	torch: 2.7.1+cu126<br>
	scikit-learn: 1.7.0<br>
	numpy: 2.2.6<br>

6. Das Skript, welches zur Ermittlung der Wahrscheinlichkeiten pro Label für ein geladenes Modell, welches zuvor den Prozess des Finetunings unterlaufen ist, genutzt wurde. Der Modellpfad muss an das zu testende Modell angepasst werden. Läuft mit Python 3.10. Pakete siehe 5. <br>
7.-10.: Die Konsolenausgabe beim Finetuning mit den im Filename genannten Parametern (Testreihe 1 dieser Arbeit) als Textdatei.<br>
11. und 12.: Die Konsolenausgabe für die Testung der Labelwahrscheinlichkeiten der Modelle mit den im Filename genannten Parametern als Textdatei.<br>
13.-22.: Testreihe zu den weight decay Anpassungen mit jeweils 5 Testungen pro Parameter als Textdatei<br>
