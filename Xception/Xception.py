# IMPORTS 
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from glob import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# DATA MANAGEMENT CLASS 
class DatasetManager:
    
    def __init__(self, dimensions=(299, 299), batch_sz=32):
        self.dimensions = dimensions
        self.batch_sz = batch_sz
        self.training_data = None
        self.testing_data = None
        self.validation_data = None
        
    def create_dataframe(self, directory_path):
        file_paths = []
        categories = []
        
        for category in os.listdir(directory_path):
            category_dir = os.path.join(directory_path, category)
            if os.path.isdir(category_dir):
                for img_file in os.listdir(category_dir):
                    file_paths.append(os.path.join(category_dir, img_file))
                    categories.append(category)
        
        return pd.DataFrame({
            'Class Path': file_paths,
            'Class': categories
        })
    
    def load_datasets(self, train_dir, test_dir):
        """Load training and testing datasets"""
        self.training_data = self.create_dataframe(train_dir)
        self.testing_data = self.create_dataframe(test_dir)
        return self.training_data, self.testing_data
    
    def split_validation(self, split_ratio=0.5):
        """Create validation set from test data"""
        self.validation_data, self.testing_data = train_test_split(
            self.testing_data,
            train_size=split_ratio,
            random_state=20,
            stratify=self.testing_data['Class']
        )
        return self.validation_data


# VISUALIZATION MODULE 
class DataVisualizer:
    
    @staticmethod
    def plot_class_distribution(dataframe, title_text):
        plt.figure(figsize=(15, 7))
        chart = sns.countplot(data=dataframe, y=dataframe['Class'])
        
        plt.xlabel('')
        plt.ylabel('')
        plt.title(title_text, fontsize=20)
        chart.bar_label(chart.containers[0])
        plt.show()
    
    @staticmethod
    def display_sample_images(generator, category_map):
        """Show sample images from generator"""
        category_list = list(category_map.keys())
        sample_imgs, sample_labels = next(generator)
        
        plt.figure(figsize=(20, 20))
        
        for idx, (img, lbl) in enumerate(zip(sample_imgs, sample_labels)):
            plt.subplot(4, 4, idx + 1)
            plt.imshow(img)
            category = category_list[np.argmax(lbl)]
            plt.title(category, color='k', fontsize=15)
        
        plt.show()
    
    @staticmethod
    def plot_training_history(history_data):
        metrics = {
            'accuracy': history_data.history['accuracy'],
            'loss': history_data.history['loss'],
            'precision': history_data.history['precision'],
            'recall': history_data.history['recall'],
            'val_accuracy': history_data.history['val_accuracy'],
            'val_loss': history_data.history['val_loss'],
            'val_precision': history_data.history['val_precision'],
            'val_recall': history_data.history['val_recall']
        }
        
        optimal_epochs = {
            'loss': np.argmin(metrics['val_loss']),
            'accuracy': np.argmax(metrics['val_accuracy']),
            'precision': np.argmax(metrics['val_precision']),
            'recall': np.argmax(metrics['val_recall'])
        }
        
        best_values = {
            'loss': metrics['val_loss'][optimal_epochs['loss']],
            'accuracy': metrics['val_accuracy'][optimal_epochs['accuracy']],
            'precision': metrics['val_precision'][optimal_epochs['precision']],
            'recall': metrics['val_recall'][optimal_epochs['recall']]
        }
        
        epoch_range = [i + 1 for i in range(len(metrics['accuracy']))]
        
        plt.figure(figsize=(20, 12))
        plt.style.use('fivethirtyeight')
        
        plt.subplot(2, 2, 1)
        plt.plot(epoch_range, metrics['loss'], 'r', label='Training loss')
        plt.plot(epoch_range, metrics['val_loss'], 'g', label='Validation loss')
        plt.scatter(optimal_epochs['loss'] + 1, best_values['loss'], 
                   s=150, c='blue', label=f'Best epoch = {optimal_epochs["loss"] + 1}')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(epoch_range, metrics['accuracy'], 'r', label='Training Accuracy')
        plt.plot(epoch_range, metrics['val_accuracy'], 'g', label='Validation Accuracy')
        plt.scatter(optimal_epochs['accuracy'] + 1, best_values['accuracy'],
                   s=150, c='blue', label=f'Best epoch = {optimal_epochs["accuracy"] + 1}')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(epoch_range, metrics['precision'], 'r', label='Precision')
        plt.plot(epoch_range, metrics['val_precision'], 'g', label='Validation Precision')
        plt.scatter(optimal_epochs['precision'] + 1, best_values['precision'],
                   s=150, c='blue', label=f'Best epoch = {optimal_epochs["precision"] + 1}')
        plt.title('Precision and Validation Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(epoch_range, metrics['recall'], 'r', label='Recall')
        plt.plot(epoch_range, metrics['val_recall'], 'g', label='Validation Recall')
        plt.scatter(optimal_epochs['recall'] + 1, best_values['recall'],
                   s=150, c='blue', label=f'Best epoch = {optimal_epochs["recall"] + 1}')
        plt.title('Recall and Validation Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
        plt.show()


# MODEL ARCHITECTURE 
class ImageClassificationModel:
    """Neural network model for image classification"""
    
    def __init__(self, input_dims=(299, 299, 3), num_classes=4):
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.model = None
        self.category_mapping = None
        
    def build_architecture(self):
        """Construct the model architecture"""
        # Load pre-trained backbone
        backbone = tf.keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_dims,
            pooling='max'
        )
        
        self.model = Sequential([
            backbone,
            Flatten(),
            Dropout(rate=0.3),
            Dense(128, activation='relu'),
            Dropout(rate=0.25),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            Adamax(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        
        return self.model
    
    def display_architecture(self):
        self.model.summary()
        return tf.keras.utils.plot_model(self.model, show_shapes=True)
    
    def train_model(self, train_gen, valid_gen, num_epochs=10):
        training_history = self.model.fit(
            train_gen,
            epochs=num_epochs,
            validation_data=valid_gen,
            shuffle=False
        )
        return training_history
    
    def evaluate_performance(self, train_gen, valid_gen, test_gen):
        scores = {}
        
        train_metrics = self.model.evaluate(train_gen, verbose=1)
        scores['train'] = {
            'loss': train_metrics[0],
            'accuracy': train_metrics[1]
        }
        
        valid_metrics = self.model.evaluate(valid_gen, verbose=1)
        scores['validation'] = {
            'loss': valid_metrics[0],
            'accuracy': valid_metrics[1]
        }
        
        test_metrics = self.model.evaluate(test_gen, verbose=1)
        scores['test'] = {
            'loss': test_metrics[0],
            'accuracy': test_metrics[1]
        }
        
        print(f"Train Loss: {scores['train']['loss']:.4f}")
        print(f"Train Accuracy: {scores['train']['accuracy']*100:.2f}%")
        print('-' * 20)
        print(f"Validation Loss: {scores['validation']['loss']:.4f}")
        print(f"Validation Accuracy: {scores['validation']['accuracy']*100:.2f}%")
        print('-' * 20)
        print(f"Test Loss: {scores['test']['loss']:.4f}")
        print(f"Test Accuracy: {scores['test']['accuracy']*100:.2f}%")
        
        return scores
    
    def generate_confusion_matrix(self, test_gen, category_dict):
        predictions = self.model.predict(test_gen)
        predicted_classes = np.argmax(predictions, axis=1)
        
        conf_matrix = confusion_matrix(test_gen.classes, predicted_classes)
        category_names = list(category_dict.keys())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=category_names, yticklabels=category_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('Truth Label')
        plt.show()
        
        report = classification_report(test_gen.classes, predicted_classes)
        print(report)
        
        return conf_matrix
    
    def single_image_prediction(self, image_path, category_dict):
        category_names = list(category_dict.keys())
        
        plt.figure(figsize=(12, 12))
        
        original_img = Image.open(image_path)
        preprocessed = original_img.resize((299, 299))
        img_array = np.asarray(preprocessed)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255
        
        prediction_probs = self.model.predict(img_array)
        probabilities = list(prediction_probs[0])
        
        plt.subplot(2, 1, 1)
        plt.imshow(preprocessed)
        
        plt.subplot(2, 1, 2)
        bar_chart = plt.barh(category_names, probabilities)
        plt.xlabel('Probability', fontsize=15)
        ax = plt.gca()
        ax.bar_label(bar_chart, fmt='%.2f')
        plt.show()


#  DATA GENERATOR FACTORY 
class GeneratorFactory:
    
    @staticmethod
    def create_generators(train_df, valid_df, test_df, img_size=(299, 299), batch_size=32):
        """Initialize all data generators"""
        
        augmentation_gen = ImageDataGenerator(
            rescale=1/255,
            brightness_range=(0.8, 1.2)
        )
        
        test_preprocessor = ImageDataGenerator(rescale=1/255)
        
        train_generator = augmentation_gen.flow_from_dataframe(
            train_df,
            x_col='Class Path',
            y_col='Class',
            batch_size=batch_size,
            target_size=img_size
        )
        
        valid_generator = augmentation_gen.flow_from_dataframe(
            valid_df,
            x_col='Class Path',
            y_col='Class',
            batch_size=batch_size,
            target_size=img_size
        )
        
        test_generator = test_preprocessor.flow_from_dataframe(
            test_df,
            x_col='Class Path',
            y_col='Class',
            batch_size=16,
            target_size=img_size,
            shuffle=False
        )
        
        return train_generator, valid_generator, test_generator


# MAIN EXECUTION PIPELINE 
def main_pipeline():
    
    data_manager = DatasetManager(dimensions=(299, 299), batch_sz=32)
    visualizer = DataVisualizer()
    
    train_df = data_manager.create_dataframe('')  # Add train path
    test_df = data_manager.create_dataframe('')   # Add test path
    
    print(train_df)
    print(test_df)
    
    visualizer.plot_class_distribution(train_df, 'Count of images in each class')
    
    plt.figure(figsize=(15, 7))
    ax = sns.countplot(y=test_df['Class'], palette='viridis')
    ax.set(xlabel='', ylabel='', title='Count of images in each class')
    ax.bar_label(ax.containers[0])
    plt.show()
    
    valid_df, test_df = train_test_split(
        test_df,
        train_size=0.5,
        random_state=20,
        stratify=test_df['Class']
    )
    print(valid_df)
    
    train_generator, valid_generator, test_generator = GeneratorFactory.create_generators(
        train_df, valid_df, test_df
    )
    
    class_dict = train_generator.class_indices
    visualizer.display_sample_images(test_generator, class_dict)
    
    classifier = ImageClassificationModel(input_dims=(299, 299, 3), num_classes=4)
    classifier.build_architecture()
    classifier.display_architecture()
    
    history = classifier.train_model(train_generator, valid_generator, num_epochs=10)
    print(history.history.keys())
    
    visualizer.plot_training_history(history)
    
    classifier.evaluate_performance(train_generator, valid_generator, test_generator)
    
    classifier.generate_confusion_matrix(test_generator, class_dict)
    
    return classifier, class_dict


def predict(img_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    label = list(class_dict.keys())
    plt.figure(figsize=(12, 12))
    img = Image.open(img_path)
    resized_img = img.resize((299, 299))
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    predictions = model.predict(img)
    probs = list(predictions[0])
    labels = label
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    plt.subplot(2, 1, 2)
    bars = plt.barh(labels, probs)
    plt.xlabel('Probability', fontsize=15)
    ax = plt.gca()
    ax.bar_label(bars, fmt='%.2f')
    plt.show()


if __name__ == "__main__":
    # Run the main pipeline
    model_instance, class_dict = main_pipeline()
    model = model_instance.model  # For compatibility with predict function
