# Colon Histology Classifier

Questo progetto analizza sezioni di colon colorate con ematossilina 
ed eosina (H&E) al fine di determinare la presenza o meno di un 
adenocarcinoma del colon.

## Modello

Transfer learning con ResNet50 pre-trained su ImageNet, 
fine-tuned sul dataset LC25000 (25.000 immagini istologiche).
L'interpretabilità è garantita tramite Grad-CAM, che visualizza 
le regioni dell'immagine su cui il modello basa la classificazione.

## Risultati

- Accuracy: 99.87%
- AUC: 1.000
- ROC curve: quasi perfetta

## Limiti

Il dataset LC25000 è costituito da preparati ben conservati e 
con colorazioni omogenee. I risultati potrebbero variare su immagini 
provenienti da contesti clinici reali, con variabilità 
inter-laboratorio nelle colorazioni e nella qualità dei preparati.

## Setup

```bash
pip install torch torchvision scikit-learn grad-cam matplotlib pillow
```

Dataset: [LC25000 su Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

## Autore

Studentessa di Medicina, Università Cattolica del Sacro Cuore, Roma
