from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Load PyTorch model and convert to ONNX
onnx_model = ORTModelForSequenceClassification.from_pretrained("my_transformer_model", export=True)

# Create quantizer
quantizer = ORTQuantizer.from_pretrained(onnx_model)

# Define the quantization strategy by creating the appropriate configuration
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# Quantize the model
model_quantized_path = quantizer.quantize(
    save_dir="indobertweet_sentiment_optimized",
    quantization_config=dqconfig,
)