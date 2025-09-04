import tensorflow as tf

model = tf.keras.models.load_model('inception_dogcat_model.h5') # Load your Keras H5 model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

tflite_output_path = "inception_dogcat_optimized.tflite"
    
# Save model
with open(tflite_output_path, "wb") as f:
    f.write(quantized_tflite_model)