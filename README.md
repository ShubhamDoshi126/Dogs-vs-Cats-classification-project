# Dogs-vs-Cats-classification-project
This project demonstrates how to classify images of dogs and cats using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. It includes functionalities for organizing the dataset, building and training the model, making predictions, and visualizing results. Testing when binary classification is done on multiple types of image of a single class i.e dogs and cats breeds in this case how will it work out.

Files:
dogs_cats.py: Python script defining the DogsCats class for dataset preparation, model building, training, saving, loading, and prediction.
module10.ipynb: Jupyter Notebook for training the model, visualizing training progress, and making predictions.

Key Learnings:

Data Preparation
Dataset Organization: Organize the dataset into training, validation, and test sets using the make_dataset_folders method.
Image Dataset Creation: Create TensorFlow image datasets from directories using image_dataset_from_directory.

Model Building
CNN Architecture: Build a Convolutional Neural Network (CNN) using TensorFlow and Keras, including layers for data augmentation, convolution, pooling, and dense layers.
Model Compilation: Compile the model with a loss function, optimizer, and evaluation metrics.

Training and Evaluation
Training Loop: Train the model using the fit method, including the use of callbacks for early stopping, model checkpointing, and TensorBoard logging.
Performance Visualization: Visualize training and validation accuracy and loss using Matplotlib.

Model Saving and Loading
Model Saving: Save the trained model to a file using the ModelCheckpoint callback.
Model Loading: Load a saved model for future use.

Prediction
Image Preprocessing for Prediction: Preprocess images for prediction, including resizing, normalization, and reshaping.
Making Predictions: Make predictions on new images using the trained model and visualize the results.

Usage
Dataset Preparation
Run the following commands in the Jupyter Notebook to organize the dataset into training, validation, and test sets:
dogs_cats = DogsCats()
dogs_cats.make_dataset_folders('validation', 0, 2400)
dogs_cats.make_dataset_folders('train', 2400, 12000)
dogs_cats.make_dataset_folders('test', 12000, 12500)
dogs_cats.make_dataset()


Model Building and Training
Run the following commands to build and train the CNN model:
dogs_cats.build_network()
dogs_cats.model.summary()
dogs_cats.train('model.dogs-cats')

Model Saving and Loading
Run the following commands to save and load the trained model:
dogs_cats.save_model('model.dogs-cats')
dogs_cats.load_model('model.dogs-cats')

Making Predictions
Run the following command to make predictions on new images:
dogs_cats.predict('path/to/image.jpg')

Conclusion
This project provides a comprehensive understanding of implementing, training, and evaluating a Convolutional Neural Network (CNN) for image classification tasks using TensorFlow and Keras. It demonstrates the integration of various tools and libraries to create a complete workflow for classifying images of dogs and cats, including dataset preparation, model building, training, saving, loading, and making predictions.
