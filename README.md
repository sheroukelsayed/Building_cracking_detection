# Building_cracking_detection
 Crack detection has vital importance for structural health monitoring and inspection. We would like to train a network to detect Cracks, we will denote the images that contain cracks as positive and images with no cracks as negative.

The provided code  for building and training Convolutional Neural Network (CNN) models using the Keras library. Below is a step-by-step explanation of the code:

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Mounts Google Drive to the Colab environment, allowing access to files stored on Google Drive.

2. Install skillsnetwork Package:
   ```python
   !pip install skillsnetwork
   ```
   Installs the `skillsnetwork` package, which may contain custom functionalities or utilities.

3. Import Libraries:
   ```python
   import numpy as np
   from keras.preprocessing.image import ImageDataGenerator
   from keras.models import Sequential
   from keras.layers import Dense
   from keras.applications import ResNet50, VGG16
   from keras.applications.resnet50 import preprocess_input
   ```
   Imports necessary libraries for working with images and building the CNN models.

4. Prepare Data:
   ```python
   await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip", overwrite=True)
   ```
   Downloads and extracts the concrete data required for training and testing the models.

5. **Data Generators:**
   ```python
   data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

   train_generator = data_generator.flow_from_directory(
       'concrete_data_week4/train',
       target_size=(224, 224),
       batch_size=100,
       class_mode='categorical'
   )
   valid_generator = data_generator.flow_from_directory(
       'concrete_data_week4/valid',
       target_size=(224, 224),
       batch_size=100,
       class_mode='categorical'
   )
   test_generator = data_generator.flow_from_directory(
       'concrete_data_week4/test',
       target_size=(224, 224),
       batch_size=100,
       class_mode='categorical'
   )
   ```
   Sets up image data generators for training, validation, and testing, including data preprocessing and augmentation.

6. Build and Train ResNet50 Model:
   ```python
   model = Sequential()
   model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
   model.add(Dense(2, activation='softmax'))
   ```
   Builds a Sequential model with a ResNet50 base and a Dense layer for classification. Trains the model for 2 epochs.

7. Build and Train VGG16 Model:
   ```python
   model_VGG = Sequential()
   model_VGG.add(VGG16(include_top=False, pooling='avg', weights='imagenet'))
   model_VGG.add(Dense(2, activation='softmax'))
   ```
   Builds a Sequential model with a VGG16 base and a Dense layer for classification. Trains the model for 3 epochs.

8. Evaluate Model Performance:
   ```python
   vgg16_performance = model_VGG.evaluate_generator(test_generator)
   resnet50_performance = model.evaluate_generator(test_generator)
   ```
   Evaluates the performance of both models using the `evaluate_generator` method.

9. Print Model Performance:
   ```python
   print("Performance of the VGG16 model:")
   print("Loss:", vgg16_performance[0])
   print("Accuracy:", vgg16_performance[1])

   print("\nPerformance of the ResNet50 model:")
   print("Loss:", resnet50_performance[0])
   print("Accuracy:", resnet50_performance[1])
   ```
   Prints the performance metrics for both models.

10. Generate Predictions:
   ```python
   vgg16_predictions = model_VGG.predict_generator(test_generator)
   resnet50_predictions = model.predict_generator(test_generator)
   ```
   Generates predictions for the test set using both VGG16 and ResNet50 models.

11. Get Class Predictions:
   ```python
   vgg16_class_predictions = np.argmax(vgg16_predictions[:5], axis=1)
   resnet50_class_predictions = np.argmax(resnet50_predictions[:5], axis=1)
   ```
   Gets the class predictions for the first five images using `argmax`.

12. Print Class Predictions:
   ```python
   class_labels = {0: 'Negative', 1: 'Positive'}

   print("VGG16 Class Predictions:")
   for prediction in vgg16_class_predictions:
       print(class_labels[prediction])

   print("\nResNet50 Class Predictions:")
   for prediction in resnet50_class_predictions:
       print(class_labels[prediction])
   ```
   Prints the class predictions for the first five images using both VGG16 and ResNet50 models.

This code demonstrates the process of preparing data, building, training, and evaluating CNN models for a concrete-cracking classification task using both ResNet50 and VGG16 architectures. The results are then printed, including class predictions for a subset of images.
