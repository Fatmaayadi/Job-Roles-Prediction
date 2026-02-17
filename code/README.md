# 🎯 Job Classification Pipeline

Un pipeline complet de Machine Learning pour la classification automatique de postes/métiers basé sur les compétences, descriptions et certifications.

## 📋 Table des matières

- [Description](#-description)
- [Structure du projet](#-structure-du-projet)
- [Technologies utilisées](#-technologies-utilisées)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline de traitement](#-pipeline-de-traitement)
- [Modèles et performances](#-modèles-et-performances)
- [Résultats](#-résultats)

## 🎯 Description

Ce projet implémente un système de classification multi-classes capable de prédire le titre d'un poste à partir de :
- **Compétences techniques** (Skills)
- **Description du poste** (Job Description)
- **Certifications** (Certifications)

Le système utilise plusieurs algorithmes de Machine Learning optimisés avec GridSearchCV et suit les expériences avec MLflow.

## 📁 Structure du projet

```
job-classification/
│
├── jobs.csv                              # Dataset d'origine
│
├── 1_Preprocessing.ipynb                 # Nettoyage et préparation des données
├── 2_Feature_Engineering.ipynb           # Création des features
├── 3_Modeling_GridSearch.ipynb           # Entraînement et optimisation des modèles
├── 4_MLflow.ipynb                        # Tracking et gestion des expériences
│
├── preprocessed_data.pkl                 # Données après preprocessing
├── feature_sets.pkl                      # Différentes représentations de features
├── modeling_results_gridsearch.pkl       # Résultats de tous les modèles
├── label_encoder.pkl                     # Encoder pour les labels
│
├── mlflow.db                             # Base de données MLflow
├── mlruns/                               # Répertoire des expériences MLflow
│
├── requirements.txt                      # Dépendances Python
└── README.md                             # Documentation (ce fichier)
```

## 🛠️ Technologies utilisées

### Librairies principales
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques
- **scikit-learn** : Algorithmes ML et outils
- **scipy** : Matrices creuses et opérations scientifiques
- **MLflow** : Tracking des expériences et gestion des modèles

### Algorithmes de classification testés
1. **Logistic Regression** - Baseline linéaire
2. **Multinomial Naive Bayes** - Adapté au texte
3. **Linear SVC** - Support Vector Classifier
4. **Random Forest** - Ensemble de Decision Trees
5. **K-Nearest Neighbors (KNN)** - Classification par proximité
6. **Decision Tree** - Arbre de décision simple

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étapes d'installation

1. **Cloner le repository** (ou télécharger les fichiers)

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### Ordre d'exécution des notebooks

Exécutez les notebooks dans l'ordre suivant :

#### 1️⃣ Preprocessing (1_Preprocessing.ipynb)
```bash
jupyter notebook 1_Preprocessing.ipynb
```
**Objectif** : Nettoyer les données, gérer les valeurs manquantes, encoder les labels
**Sortie** : `preprocessed_data.pkl`, `label_encoder.pkl`

#### 2️⃣ Feature Engineering (2_Feature_Engineering.ipynb)
```bash
jupyter notebook 2_Feature_Engineering.ipynb
```
**Objectif** : Créer différentes représentations des features textuelles
**Sortie** : `feature_sets.pkl` avec 5 types de features :
- **TF-IDF** (5000 features)
- **Count Vectorizer** (3000 features)
- **TF-IDF + SVD** (300 composantes)
- **Features statistiques** (5 features)
- **TF-IDF + Stats combinées** (5005 features)

#### 3️⃣ Modeling avec GridSearchCV (3_Modeling_GridSearch.ipynb)
```bash
jupyter notebook 3_Modeling_GridSearch.ipynb
```
**Objectif** : Entraîner et optimiser plusieurs modèles
**Sortie** : `modeling_results_gridsearch.pkl`

**Modèles testés** :
- 6 algorithmes différents
- 4 configurations de features
- ~20 combinaisons au total
- Optimisation des hyperparamètres avec GridSearchCV

#### 4️⃣ MLflow Tracking (4_MLflow.ipynb)
```bash
jupyter notebook 4_MLflow.ipynb
```
**Objectif** : Enregistrer tous les modèles et métriques dans MLflow
**Sortie** : Expériences MLflow accessibles via l'interface web

### Visualiser les expériences MLflow

Après avoir exécuté le notebook MLflow :

```bash
mlflow ui
```

Puis ouvrez votre navigateur à : `http://localhost:5000`

## 🔄 Pipeline de traitement

### 1. Preprocessing
- Chargement du dataset (`jobs.csv`)
- Nettoyage du texte (minuscules, ponctuation, caractères spéciaux)
- Combinaison des features textuelles
- Encodage des labels (Job Title)
- Split train/test (80/20)

### 2. Feature Engineering
- **TF-IDF Vectorization** : Transformation en vecteurs TF-IDF
- **Count Vectorization** : Comptage de fréquences des mots
- **Dimensionality Reduction** : Réduction avec TruncatedSVD
- **Statistical Features** : Features statistiques (longueur, nombre de mots, etc.)
- **Combined Features** : Combinaison TF-IDF + Stats

### 3. Modeling
- Entraînement de 6 modèles différents
- GridSearchCV pour optimisation des hyperparamètres
- Cross-validation 3-fold
- Évaluation sur l'ensemble de test
- Sélection automatique du meilleur modèle

### 4. Tracking
- Enregistrement de tous les modèles dans MLflow
- Logging des métriques (accuracy, precision, recall, F1)
- Sauvegarde des hyperparamètres optimaux
- Enregistrement du meilleur modèle pour déploiement

## 📊 Modèles et performances

### Métriques évaluées
- **Accuracy** : Précision globale
- **Precision** (weighted & macro)
- **Recall** (weighted & macro)
- **F1-Score** (weighted & macro)
- **Training Time** : Temps d'entraînement
- **Prediction Time** : Temps de prédiction

### Configuration optimale (exemple)
Basé sur les résultats des notebooks :
- **Meilleur modèle** : Random Forest avec features combinées
- **F1-Score** : ~0.75+ (variable selon les données)
- **Nombre de classes** : 119 job titles différents

### Grilles d'hyperparamètres

**Logistic Regression** :
- C: [0.1, 1, 10, 100]
- solver: ['saga', 'liblinear']
- max_iter: [1000, 2000]

**Random Forest** :
- n_estimators: [50, 100, 200]
- max_depth: [10, 20, 30]
- min_samples_split: [2, 5]

**KNN** :
- n_neighbors: [3, 5, 7, 9]
- weights: ['uniform', 'distance']

*(Voir notebook 3 pour la liste complète)*

## 📈 Résultats

Les résultats détaillés sont disponibles dans :
1. **Le notebook 3** : Tableaux de comparaison des modèles
2. **MLflow UI** : Visualisation interactive des expériences
3. **`modeling_results_gridsearch.pkl`** : Résultats sauvegardés

### Exemple de résultats typiques :

| Modèle | Features | F1-Score | Accuracy | Training Time |
|--------|----------|----------|----------|---------------|
| Random Forest | Combined | 0.75+ | 0.75+ | ~40s |
| Logistic Regression | TF-IDF | 0.73+ | 0.73+ | ~130s |
| Linear SVC | TF-IDF | 0.71+ | 0.72+ | ~6s |

*(Résultats indicatifs basés sur les notebooks fournis)*

## 🔍 Analyse des données

### Dataset
- **Nombre total d'exemples** : 2,458 jobs
- **Nombre de classes** : 119 job titles différents
- **Split** : 80% train (1,966) / 20% test (492)
- **Features** : Skills, Job Description, Certifications

### Distribution
- La classe la plus fréquente : "Backend Developer" (45 occurrences)
- Dataset relativement équilibré entre les classes principales

## 🔧 Maintenance et amélioration

### Prochaines étapes possibles
1. **Deep Learning** : Tester des modèles BERT ou transformers
2. **Feature Engineering** : Ajouter des embeddings (Word2Vec, GloVe)
3. **Ensemble Methods** : Stacking ou voting de plusieurs modèles
4. **Déséquilibre de classes** : SMOTE ou class weights
5. **Déploiement** : API REST avec Flask/FastAPI
6. **Interface utilisateur** : Application web pour les prédictions

### Réentraînement
Pour réentraîner avec de nouvelles données :
1. Remplacer `jobs.csv` avec les nouvelles données
2. Exécuter les notebooks 1-4 dans l'ordre
3. Les modèles seront automatiquement sauvegardés

## 📝 Notes importantes

- Le preprocessing nettoie et normalise le texte en français
- GridSearchCV utilise 3-fold cross-validation
- Les modèles sont sauvegardés au format pickle
- MLflow utilise une base SQLite locale (`mlflow.db`)
- Les features combinées ne fonctionnent pas avec tous les modèles (certains nécessitent une normalisation)

## 🤝 Contribution

Pour contribuer :
1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -m 'Ajout d'une fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## 👤 Auteur

Développé dans le cadre d'un projet de Machine Learning pour la classification automatique de postes.

## 🙏 Remerciements

- scikit-learn pour les algorithmes de ML
- MLflow pour le tracking des expériences
- Pandas pour la manipulation de données
- La communauté open-source

---

**Note** : Ce README décrit un projet complet de classification multi-classes avec optimisation d'hyperparamètres et tracking MLflow. Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue.
