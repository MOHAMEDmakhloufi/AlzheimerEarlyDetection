
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

def ad_model(all_data, target):
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(all_data, target, test_size=0.2, random_state=42)

    input_shape = (312, 1003, 3)
    # Define the input layer
    inputs = Input(shape=input_shape)

    # Preprocess the input data
    x = preprocess_input(inputs)

    # Load the DenseNet121 model with ImageNet weights, excluding the top classification layers
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=x)

    # Add a global average pooling layer to reduce dimensions
    x = GlobalAveragePooling2D()(base_model.output)

    # Add a fully connected layer for binary classification
    x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularization to avoid overfitting

    # Add the final output layer with sigmoid for binary classification
    outputs = Dense(1, activation='sigmoid')(x)

    # Create the model
    model = Model(inputs, outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),  # Use the specified learning rate
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )

    # Print the model summary
    model.summary()

    # Define parameters for training
    batch_size = 15
    epochs = 60
    learning_rate = 1e-3

    # Define callbacks for early stopping and model checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras', monitor='val_loss', save_best_only=True
        )
    ]

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Set up data augmentation with ImageDataGenerator
    datagen = ImageDataGenerator(
        brightness_range=[0.8, 1.2],  # Brightness adjustments
        width_shift_range=0.2,  # Horizontal shift
        horizontal_flip=True  # Horizontal flip
    )

    # Use datagen.flow() to generate batches of augmented data
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

    # Fit the model
    history = model.fit(
        X_train, y_train,  # No need for extra parentheses here
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_val, y_val),  # Validation data as a tuple
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1  # Set to 0 or 2 for silent or detailed logs
    )

    # Evaluate on the test set
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_val, y_val)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")