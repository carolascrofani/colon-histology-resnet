#AI-Driven Histopathology: Colon Adenocarcinoma Classification
A Deep Learning approach using Transfer Learning and Grad-CAM Explainability.

##Project Overview
Questo progetto esplora l'efficacia delle Convolutional Neural Networks (CNN) nella distinzione automatizzata tra tessuto sano e adenocarcinoma del colon in immagini istopatologiche colorate con Ematossilina ed Eosina (H&E). L'obiettivo è validare la capacità di astrazione di modelli pre-trainati su task di computer vision generica applicati alla patologia digitale.

##Methodology
Model Architecture: ResNet50 (Transfer Learning con pesi ImageNet).

Dataset: LC25000 (selezione di 25.000 immagini totali). [LC25000 su Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

Optimization: Adam Optimizer (lr=0.001), Cross-Entropy Loss.

Explainability: Implementazione di Grad-CAM (Gradient-weighted Class Activation Mapping) per mappare le aree di attivazione neuronale e verificare la coerenza con i criteri diagnostici istologici.

##Performance & Critical Discussion
Accuracy: 99.87%

AUC-ROC: 1.000

Nota Critica: Sebbene le metriche indichino una separazione lineare perfetta delle classi nel dataset LC25000, tale performance è indicativa della standardizzazione del dataset stesso. In contesti clinici real-world, la varianza inter-laboratorio e la presenza di artefatti tecnici rappresentano la vera sfida per la generalizzazione del modello.

##Limitations & Future Directions
1. Dataset Homogeneity: Le immagini provengono da un'unica fonte, limitando la robustezza del modello verso la variabilità cromatica (staining).

2. External Validation: È necessaria una validazione su dataset indipendenti (es. TCGA) per confermare l'utilità clinica.

3. Future Work: Implementazione di tecniche di Color Normalization e test su modelli basati su Vision Transformers (ViT).

##Author
Carola Scrofani
Studentessa di Medicina | Università Cattolica del Sacro Cuore, Roma.
