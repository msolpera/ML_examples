import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import nltk
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
warnings.filterwarnings("ignore")

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fakenews_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FakeNewsModel")

class FakeNews:
    """
    Clase para la predicción de Fake News.
    
    Implementa un flujo de trabajo completo:
    - Carga y exploración de datos
    - Preprocesamiento y feature engineering
    - Entrenamiento y evaluación de modelos
    - Almacenamiento del modelo final
    """
    
    def __init__(self, data_path=None, random_state=42):
        """
        Inicializa el modelo con la ruta a los datos y un estado aleatorio para reproducibilidad.
        
        Parameters:
        -----------
        data_path : str
            Ruta al directorio que contiene los datos de vivienda
        random_state : int
            Semilla para reproducibilidad
        """
        self.random_state = random_state
        
        # Crear directorios para los resultados si no existen
        self.output_dir = Path("output_fakenews")
        self.models_dir = self.output_dir / "models"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"
        
        # Crear directorios si no existen
        for directory in [self.output_dir, self.models_dir, self.figures_dir, self.reports_dir]:
            if not directory.exists():
                directory.mkdir(parents=True)
                
        # Configurar ruta de datos
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("datasets/fakenews")
    

            
        # Atributos a inicializar más tarde
        self.fakenews = None
        self.X_train = None
        self.X_test = None
        self.X_valid = None
        self.y_train = None
        self.y_test = None
        self.y_valid = None
        self.X_train_tf = None
        self.X_valid_tf = None
        self.X_test_tf = None
        self.models = {}
        self.nlp = None
        self.model_results = {}
        self.best_model = None
        self.stop_words = None
        
        logger.info("Inicialización completada.")
        
    def load_data(self):
        """Carga los datos desde un archivo CSV."""
        try:
            csv_path1 = self.data_path / "Fake.csv"
            csv_path2 = self.data_path / "True.csv"
            logger.info(f"Cargando datos desde {csv_path1} y {csv_path1}")
            
            data_fake = pd.read_csv(csv_path1)
            data_real = pd.read_csv(csv_path2)

            # Agregar etiquetas (0 para fake, 1 para real)
            data_fake['label'] = 0
            data_real['label'] = 1

            # Concatenar los datasets
            self.fakenews = pd.concat([data_fake, data_real], axis=0).reset_index(drop=True)
            
            logger.info(f"Datos cargados exitosamente. Shape: {self.fakenews.shape}")
            return self.fakenews
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
            
    def explore_data(self):
        """Realiza un análisis exploratorio básico de los datos."""
        if self.fakenews is None:
            self.load_data()
            
        logger.info("Iniciando exploración de datos...")
        
        
        # Información básica
        info = {
            "shape": self.fakenews.shape,
            "columns": list(self.fakenews.columns),
            "dtypes": self.fakenews.dtypes.to_dict(),
            "missing_values": self.fakenews.isnull().sum().to_dict(),
            "class_distribution": self.fakenews['label'].value_counts().to_dict(),
            "description": self.fakenews.describe().to_dict()
        }
        
        # Información de longitud de textos
        try:
            self.fakenews['title_length'] = self.fakenews['title'].apply(len)
            self.fakenews['text_length'] = self.fakenews['text'].apply(len)
        except TypeError:
            self.fakenews.dropna(subset=['title'], inplace=True)
            self.fakenews.dropna(subset=['text'], inplace=True)
            self.fakenews['title_length'] = self.fakenews['title'].apply(len)
            self.fakenews['text_length'] = self.fakenews['text'].apply(len)

        info["text_stats"] = {
            "title_length_mean": self.fakenews['title_length'].mean(),
            "title_length_median": self.fakenews['title_length'].median(),
            "text_length_mean": self.fakenews['text_length'].mean(),
            "text_length_median": self.fakenews['text_length'].median(),
        }
        
        # Guardar resultados
        with open(self.reports_dir / "data_exploration.txt", "w") as f:
            f.write("EXPLORACIÓN DE DATOS\n")
            f.write(f"Forma del dataset: {info['shape']}\n\n")
            f.write("Columnas:\n")
            for col in info['columns']:
                f.write(f"- {col} ({info['dtypes'][col]})\n")
            f.write("\nValores faltantes:\n")
            for col, missing in info['missing_values'].items():
                if missing > 0:
                    f.write(f"- {col}: {missing} ({missing/self.fakenews.shape[0]:.2%})\n")
            f.write("\nDistribución de clases:\n")
            f.write(f"- Fake news (0): {info['class_distribution'].get(0, 0)}\n")
            f.write(f"- Real news (1): {info['class_distribution'].get(1, 0)}\n")
            f.write("\nEstadísticas de texto:\n")
            f.write(f"- Longitud media de títulos: {info['text_stats']['title_length_mean']:.2f}\n")
            f.write(f"- Longitud media de textos: {info['text_stats']['text_length_mean']:.2f}\n")
                    
        logger.info("Exploración básica completada.")
        return info
            
    def visualize_data(self):
        """Genera visualizaciones para entender mejor los datos."""
        if self.fakenews is None:
            self.load_data()
            
        logger.info("Generando visualizaciones...")
        
        # Configurar estilo de las visualizaciones
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Distribución de noticias Fakes y Reals
        plt.figure(figsize=(10, 6))
        count_classes=self.fakenews['label'].value_counts()
        sns.barplot(x=count_classes.index, y=count_classes.values)
        plt.xlabel("Class (0=Fake, 1=Real)")
        plt.ylabel(r"$N_{News}$")
        plt.xticks([0, 1], ['Fake', 'Real'])
        plt.title("Real/Fake News Distribution")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "real_fake_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribución de longitud de textos por clase
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.fakenews, x='title_length', hue='label', bins=30, kde=True, element="step")
        plt.title("Distribución de longitud de títulos")
        plt.xlabel("Longitud del título")
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=self.fakenews, x='text_length', hue='label', bins=30, kde=True, element="step")
        plt.title("Distribución de longitud de textos")
        plt.xlabel("Longitud del texto")
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "text_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Visualización de palabras más comunes (implementar después de clean_data)

        logger.info("Visualizaciones generadas y guardadas en el directorio de figuras.")

    def clean_data(self, save_intermediate=True):
        """Limpia y preprocesa los textos de las noticias."""

        cache_path = self.output_dir / "noticias_limpias.csv"
        if save_intermediate and cache_path.exists():
            logger.info("Cargando datos preprocesados desde caché...")
            self.fakenews = pd.read_csv(cache_path)
            self.fakenews.dropna(subset=['title_proc'], inplace=True)
            self.fakenews.dropna(subset=['text_proc'], inplace=True)
            return self.fakenews
        else:
            logger.info("Limpiando data ...")

            self.nlp = spacy.load("en_core_web_sm")
            self.stop_words = stopwords.words('english')
        
            def clean_text(text):
                # Convertir a minúsculas
                text = text.lower()

                # Eliminar caracteres especiales y URLs
                text = re.sub(r"http\S+|www\S+|@\w+|#\w+", '', text)  
                text = re.sub(r'[^\w\s]', '', text)  

                # Tokenizar, Lematizar y convertir a minúsculas
                doc = self.nlp(text)
                cleaned = [token.lemma_.lower() for token in doc if token.is_alpha  
                and token.text.lower() not in self.stop_words]
                return " ".join(cleaned)

            # Aplicar limpieza a títulos y textos
            self.fakenews['title_proc'] = self.fakenews['title'].apply(clean_text)
            self.fakenews['text_proc'] = self.fakenews['text'].apply(clean_text)

            # Eliminar filas que puedan haber quedado nulas
            self.fakenews.dropna(subset=['title_proc'], inplace=True)
            self.fakenews.dropna(subset=['text_proc'], inplace=True)
        
            # Guardar datos procesados
            self.fakenews.to_csv(self.output_dir / "noticias_limpias.csv", index=False)
            logger.info("Limpieza de datos completada.")
            return self.fakenews
    
    def _visualize_word_frequencies(self):
        """Visualiza las palabras más frecuentes en cada clase."""
        from collections import Counter
        import itertools
        
        # Separar textos por clase
        fake_texts = ' '.join(self.fakenews[self.fakenews['label'] == 0]['text_proc'])
        real_texts = ' '.join(self.fakenews[self.fakenews['label'] == 1]['text_proc'])
        
        # Obtener palabras más frecuentes
        fake_words = Counter(fake_texts.split()).most_common(20)
        real_words = Counter(real_texts.split()).most_common(20)
        
        # Crear visualización
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        words, counts = zip(*fake_words) if fake_words else ([], [])
        sns.barplot(x=list(counts), y=list(words))
        plt.title("Palabras más comunes en noticias falsas")
        plt.xlabel("Frecuencia")
        
        plt.subplot(1, 2, 2)
        words, counts = zip(*real_words) if real_words else ([], [])
        sns.barplot(x=list(counts), y=list(words))
        plt.title("Palabras más comunes en noticias reales")
        plt.xlabel("Frecuencia")
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "word_frequencies.png", dpi=300, bbox_inches='tight')
        plt.close()
            
    def split_data(self):
        """Divide los datos en conjuntos de entrenamiento, validación y prueba"""
        if self.fakenews is None:
            self.load_data()

        if 'title_proc' not in self.fakenews.columns:
            self.clean_data()
            
        logger.info("Dividiendo datos en conjuntos de entrenamiento y prueba...")
        
        X = self.fakenews['title_proc'] + ' ' + self.fakenews['text_proc']  
        y = self.fakenews['label']  

        X_train_full, self.X_test, y_train_full, self.y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y) 
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=self.random_state, stratify=y_train_full
        )
        

        logger.info(f"Datos divididos. Conjunto de entrenamiento: {self.X_train.shape} \n" +
                    f"Conjunto de validación: {self.X_valid.shape}\n"
                    f"Conjunto de prueba: {self.X_test.shape}")
        
                
    def preprocess_data(self):
        """Preprocesa los datos para modelado."""
        if self.X_train is None:
            self.split_data()
            
        logger.info("Preprocesando datos...")

        # Crear y ajustar el vectorizador TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=0.95,  # Ignorar términos que aparecen en >95% de los documentos
            min_df=2,     # Ignorar términos que aparecen en <2 documentos
            stop_words=self.stop_words,
            max_features=5000  # Limitar a 5000 características
        )

        # Transformar datos
        self.X_train_tf = self.tfidf_vectorizer.fit_transform(self.X_train) 
        self.X_valid_tf = self.tfidf_vectorizer.transform(self.X_valid) 
        self.X_test_tf = self.tfidf_vectorizer.transform(self.X_test) 
        
        # Guardar características más importantes
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        logger.info(f"Datos preprocesados. Características: {self.X_train_tf.shape[1]}")

        # Guardar información sobre las características
        with open(self.reports_dir / "feature_info.txt", "w") as f:
            f.write(f"Número total de características: {len(feature_names)}\n\n")
            f.write("Primeras 50 características (ordenadas alfabéticamente):\n")
            for feature in sorted(feature_names)[:50]:
                f.write(f"- {feature}\n")
        
        return self.X_train_tf, self.X_valid_tf, self.X_test_tf
        
    def train_and_evaluate_models(self):
        """Entrena y evalúa múltiples modelos."""
        if self.X_train_tf is None:
            self.preprocess_data()
            
        logger.info("Entrenando y evaluando modelos...")
        
        # Definir modelos a entrenar
        self.models = {
            "passive_aggressive": PassiveAggressiveClassifier(max_iter = 50, random_state=self.random_state),
            "linear_scv": LinearSVC(random_state=self.random_state),
            "logistic": LogisticRegression(random_state=self.random_state),
            'multinomial_nb': MultinomialNB(),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.random_state)
        }
        
        # Parámetros para búsqueda de hiperparámetros
        param_grids = {
            "passive_aggressive": {
                "C": [0.01, 0.1, 1.0]
            },
            "linear_scv": {
                "C": [0.1, 1.0, 10.0]
            },
            "logistic": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["liblinear", "saga"]
            },
            "multinomial_nb": {
                "alpha": [0.01, 0.1, 0.5, 1.0]
            },
            "gradient_boosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        }
        
        # Entrenar y evaluar modelos
        results = []
        
        # Entrenar y evaluar cada modelo con búsqueda de hiperparámetros
        for name, base_model in self.models.items():
            logger.info(f"Entrenando modelo con GridSearch: {name}")
        
            # Corregir nombre de clave si es necesario (verificar que coincida con param_grids)
            grid_key = name.lower()
        
            # Realizar búsqueda de hiperparámetros
            grid_search = GridSearchCV(
                base_model, 
                param_grids[grid_key], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1  # Usar todos los núcleos disponibles
                )
        
            # Ajustar la búsqueda de hiperparámetros
            grid_search.fit(self.X_train_tf, self.y_train)
        
            # Guardar el mejor modelo encontrado
            self.models[name] = grid_search.best_estimator_
        
            # Predicciones en conjunto de validación con el mejor modelo
            y_pred = grid_search.predict(self.X_valid_tf)
        
            # Métricas de evaluación
            accuracy = accuracy_score(self.y_valid, y_pred)
            precision = precision_score(self.y_valid, y_pred)
            recall = recall_score(self.y_valid, y_pred)
            f1 = f1_score(self.y_valid, y_pred)

            results.append({
                "Modelo": name,
                "Best Params": str(grid_search.best_params_),
                "CV Accuracy": f"{grid_search.best_score_:.4f}",
                "Validation Accuracy": f"{accuracy:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "F1 Score": f"{f1:.4f}"
            })

            logger.info(f"Modelo {name}: Best params={grid_search.best_params_}, Accuracy={accuracy:.4f}, F1={f1:.4f}")

        self.model_results = results

        # Crear DataFrame con resultados
        results_df = pd.DataFrame(results)

        # Guardar resultados
        results_df.to_csv(self.reports_dir / "model_evaluation.csv", index=False)

        # Visualizar resultados
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Modelo", y="Validation Accuracy", data=results_df)
        plt.title("Comparación de Precisión de Modelos")
        plt.ylim(0.7, 1.0)  # Ajustar según los resultados
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Evaluación de modelos completada.")
        return self.model_results
        
    def tune_best_model(self):
        """Ajusta hiperparámetros del mejor modelo."""
        if not self.model_results:
            self.train_and_evaluate_models()
            
        logger.info("Ajustando hiperparámetros del mejor modelo...")
        
        # Encontrar el mejor modelo basado en F1 Score
        best_model = max(self.model_results, key=lambda x: float(x["F1 Score"].split()[0]))
        best_model_name = best_model['Modelo']
        
        logger.info(f"Mejor modelo seleccionado: {best_model_name}")
        
        # Definir parámetros para búsqueda según el modelo
        param_grids = {
            "passive_aggressive": {
                "C": [0.001, 0.01, 0.1, 0.5, 5.],
                "max_iter": [50, 100, 200]
            },
            "linear_svc": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "loss": ["hinge", "squared_hinge"],
                "dual": [True, False]
            },
            "logistic": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["liblinear", "saga"],
                "penalty": ["l1", "l2"]
            },
            "multinomial_nb": {
                "alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
            },

            #"gradient_boosting": {
            #    "n_estimators": [50, 100, 200, 300],
            #    "learning_rate": [0.01, 0.05, 0.1, 0.2],
            #    "max_depth": [3, 4, 5, 6],
            #    "min_samples_split": [2, 5, 10]
            #}
        }
        
        # Realizar búsqueda de hiperparámetros
        if best_model_name == "gradient_boosting":
            self.best_model = self.models[best_model_name]
            best_params = best_model['Best Params']
            return self.best_model, best_params
        
        else:
            grid_search = GridSearchCV(
                self.models[best_model['Modelo']],
                param_grids[best_model_name],
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(self.X_train_tf, self.y_train)

            # Obtener mejores hiperparámetros
            best_params = grid_search.best_params_
            logger.info(f"Mejores hiperparámetros encontrados: {best_params}")

            # Entrenar modelo final con mejores hiperparámetros
            self.best_model = grid_search.best_estimator_

            # Evaluar en conjunto de validación
            y_pred = self.best_model.predict(self.X_valid_tf)
            accuracy = accuracy_score(self.y_valid, y_pred)
            f1 = f1_score(self.y_valid, y_pred)

            logger.info(f"Modelo optimizado: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            # Guardar información de hiperparámetros
            with open(self.reports_dir / "best_model_params.txt", "w") as f:
                f.write(f"Mejor modelo: {best_model_name}\n")
                f.write("Mejores hiperparámetros:\n")
                for param, value in best_params.items():
                    f.write(f"- {param}: {value}\n")
                f.write(f"\nMétricas en validación:\n")
                f.write(f"- Accuracy: {accuracy:.4f}\n")
                f.write(f"- F1 Score: {f1:.4f}\n")

            return self.best_model, best_params
    
    def evaluate_final_model(self):
        """Evalúa el modelo final en el conjunto de prueba."""
        if self.best_model is None:
            self.tune_best_model()
            
        logger.info("Evaluando modelo final en conjunto de prueba...")
        
        # Predecir en conjunto de prueba
        y_pred = self.best_model.predict(self.X_test_tf)
        
        # Calcular métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        
        logger.info(f"Rendimiento final: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        # Guardar resultados
        with open(self.reports_dir / "final_evaluation.txt", "w") as f:
            f.write("EVALUACIÓN DEL MODELO FINAL\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Matriz de Confusión:\n")
            f.write(f"{cm}\n\n")
            f.write("Reporte de Clasificación:\n")
            f.write(f"{report}\n")
        
        # Visualizar matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión')
        plt.tight_layout()
        plt.savefig(self.figures_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Retornar métricas
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }
    
    def save_model(self):
        """Guarda el modelo final y el vectorizador TF-IDF."""
        if self.best_model is None:
            self.tune_best_model()
            
        logger.info("Guardando modelo final...")
        
        # Crear pipeline completo
        pipeline = {
            "vectorizer": self.tfidf_vectorizer,
            "model": self.best_model
        }
        
        # Guardar modelo y vectorizador
        
        joblib.dump(pipeline, self.models_dir / "fakenews_model_pipeline.pkl")
        
        logger.info(f"Modelo guardado en {self.models_dir / 'fakenews_model_pipeline.pkl'}")

    def predict(self, title, text):
        """Realiza predicciones con el modelo entrenado."""
        if self.best_model is None:
            # Intentar cargar modelo guardado
            try:
                pipeline = joblib.load(self.models_dir / "fakenews_model_pipeline.pkl")
                self.tfidf_vectorizer = pipeline["vectorizer"]
                self.best_model = pipeline["model"]
            except:
                logger.error("No se encontró un modelo entrenado.")
                return None
        
        # Limpiar texto
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
            self.stop_words = stopwords.words('english')
            
        def clean_text(text):
            if not isinstance(text, str):
                return ""
                
            # Convertir a minúsculas
            text = text.lower()
            
            # Eliminar caracteres especiales y URLs
            text = re.sub(r"http\S+|www\S+|@\w+|#\w+", '', text)  
            text = re.sub(r'[^\w\s]', '', text)  

            # Tokenizar, Lematizar y eliminar stopwords
            doc = self.nlp(text)
            cleaned = [token.lemma_.lower() for token in doc 
                      if token.is_alpha and token.text.lower() not in self.stop_words]
            return " ".join(cleaned)
        
        # Aplicar limpieza
        clean_title = clean_text(title)
        clean_text_content = clean_text(text)
        
        # Combinar título y texto, similar al entrenamiento
        combined_text = clean_title + " " + clean_text_content
        
        # Transformar texto a características TF-IDF
        X = self.tfidf_vectorizer.transform([combined_text])
        
        # Realizar predicción
        prediction = self.best_model.predict(X)[0]
        
        # Obtener probabilidades si el modelo lo admite
        probas = None
        if hasattr(self.best_model, "predict_proba"):
            probas = self.best_model.predict_proba(X)[0]
        
        result = {
            "prediction": int(prediction),
            "label": "Real" if prediction == 1 else "Fake",
            "probabilities": probas
        }
        
        logger.info(f"Predicción realizada: {result['label']}")
        return result

    def run_full_pipeline(self):
        
        """Ejecuta el pipeline de trabajo completo."""
        logger.info("Iniciando pipeline completo...")
        
        # Cargar y explorar datos
        self.load_data()
        self.explore_data()
        self.visualize_data()
        
        # Limpiar y preprocesar
        self.clean_data()
        self.split_data()
        self.preprocess_data()
        
        # Entrenar y optimizar modelos
        self.train_and_evaluate_models()
        self.tune_best_model()
        
        # Evaluar y guardar modelo final
        metrics = self.evaluate_final_model()
        self.save_model()
        
        logger.info("Pipeline completado.")
        return metrics

    
if __name__ == "__main__":
    # Crear modelo
    fakenews_model = FakeNews()
    
    # Ejecutar el pipeline completo
    metrics = fakenews_model.run_full_pipeline()
    
    # Realizar una predicción de ejemplo
    test_title = "Scientists discover revolutionary cancer treatment"
    test_text = "A team of researchers has announced a breakthrough in cancer treatment that could save millions of lives..."
    
    prediction = fakenews_model.predict(test_title, test_text)
    print(f"Predicción para noticia de prueba: {prediction['label']}")
    
    # Mostrar métricas finales
    print("\nMétricas finales del modelo:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")