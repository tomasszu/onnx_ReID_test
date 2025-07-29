import onnx

model = onnx.load("model/net_39_opt.onnx")

# Check input/output tensor data types
print("Model Inputs:")
for input_tensor in model.graph.input:
    print(f"{input_tensor.name}: {input_tensor.type}")

print("\nModel Outputs:")
for output_tensor in model.graph.output:
    print(f"{output_tensor.name}: {output_tensor.type}")

# Check precision of initializers (weights)
print("\nModel Weights (Initializers):")
for initializer in model.graph.initializer[:5]:  # print only a few for brevity
    print(f"{initializer.name}: {onnx.TensorProto.DataType.Name(initializer.data_type)}")
