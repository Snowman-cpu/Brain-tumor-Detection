import pickle
import cv2
import numpy as np
import os
from PIL import Image

def predict_brain_image(image_path, model_path='svm_brain_model.pkl'):
    """
    Analyzes a brain image using the trained SVM model and returns classification results.
    
    Parameters:
        image_path (str): Path to the input brain image file
        model_path (str): Path to the trained model file (default: 'svm_brain_model.pkl')
    
    Returns:
        dict: Classification results including prediction, confidence, and probability distribution
              or None if processing fails
    """
    print("Initializing model...")
    try:
        with open(model_path, 'rb') as model_file:
            brain_data = pickle.load(model_file)
        
        # Extract model components
        model = brain_data['model']
        scaler = brain_data['scaler']
        pca = brain_data['pca']
        class_names = brain_data['class_names']
        img_size = brain_data['img_size']
        
        print(f"Model loaded successfully. Detectable classes: {', '.join(class_names)}")
        
        # Process the input image
        print(f"Processing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Unable to read the specified image file.")
            return None
        
        # Preprocess image according to training parameters
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_flat = img.flatten().reshape(1, -1)
        
        # Apply feature transformation
        img_scaled = scaler.transform(img_flat)
        img_pca = pca.transform(img_scaled)
        
        # Generate prediction and probabilities
        prediction = model.predict(img_pca)[0]
        probabilities = model.predict_proba(img_pca)[0]
        confidence = probabilities[list(class_names).index(prediction)] * 100
        
        # Format results
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "all_probabilities": dict(zip(class_names, probabilities * 100))
        }
        
        return result
        
    except Exception as e:
        print(f"Error during image processing: {e}")
        return None

# Main execution
if __name__ == "__main__":
    model_path = input("Please specify the model path (default: 'svm_brain_model.pkl'): ") or "svm_brain_model.pkl"
    
    while True:
        # Request image input
        image_path = input("\nPlease enter the path to the image file (or 'exit' to terminate): ")
        if image_path.lower() == 'exit':
            print("Program terminated.")
            break
            
        if not os.path.exists(image_path):
            print("Error: The specified file path does not exist.")
            continue
            
        # Process the image
        result = predict_brain_image(image_path, model_path)
        
        if result:
            print("\nRESULTS:")
            print(f"Classification: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            
            print("\nProbability Distribution:")
            # Sort probabilities in descending order
            sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                print(f"  {class_name}: {prob:.2f}%")
            
            print("\nWould you like to analyze another image?")
