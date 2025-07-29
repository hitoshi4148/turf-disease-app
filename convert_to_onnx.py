import tensorflow as tf
import tf2onnx
import onnx

# モデルの読み込み
model = tf.keras.models.load_model("model.h5")

# ONNXに変換
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# 保存
onnx.save(onnx_model, "model.onnx")
print("✅ model.onnx に変換完了")
