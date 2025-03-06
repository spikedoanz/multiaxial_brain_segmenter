#!/usr/bin/env python3
import os
import argparse
import numpy as np
import onnx
from onnx import numpy_helper

def analyze_onnx_model(model_path):
    """
    Analyze the structure of an ONNX model to understand its architecture.
    
    Args:
        model_path: Path to the ONNX model file
    """
    print(f"Analyzing ONNX model: {model_path}")
    model = onnx.load(model_path)
    
    # Model metadata
    print("\n=== Model Metadata ===")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} - {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Model Version: {model.model_version}")
    print(f"Doc String: {model.doc_string}")
    
    # Model inputs and outputs
    print("\n=== Inputs ===")
    for i, input_info in enumerate(model.graph.input):
        print(f"Input #{i}: {input_info.name}")
        print(f"  Shape: {[dim.dim_value for dim in input_info.type.tensor_type.shape.dim]}")
        print(f"  Type: {input_info.type.tensor_type.elem_type}")
    
    print("\n=== Outputs ===")
    for i, output_info in enumerate(model.graph.output):
        print(f"Output #{i}: {output_info.name}")
        print(f"  Shape: {[dim.dim_value for dim in output_info.type.tensor_type.shape.dim]}")
        print(f"  Type: {output_info.type.tensor_type.elem_type}")
    
    # Node structure (operations)
    print("\n=== Node Structure ===")
    node_types = {}
    for i, node in enumerate(model.graph.node):
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
        print(f"Node #{i}: {node.op_type}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        if node.attribute:
            print(f"  Attributes: {[attr.name for attr in node.attribute]}")
    
    # Node type statistics
    print("\n=== Node Type Distribution ===")
    for op_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_type}: {count}")
    
    # Initializers (weights, biases, etc.)
    print("\n=== Initializers ===")
    for i, initializer in enumerate(model.graph.initializer):
        tensor = numpy_helper.to_array(initializer)
        print(f"Initializer #{i}: {initializer.name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Data Type: {tensor.dtype}")
        
        # For small tensors, show the values
        if np.prod(tensor.shape) <= 20:
            print(f"  Values: {tensor}")
        else:
            # Print statistics for larger tensors
            print(f"  Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")
            
            # Show a sample of the values
            if len(tensor.shape) == 2:  # For 2D matrices
                sample_size = min(3, tensor.shape[0])
                print(f"  Sample ({sample_size} rows):")
                for row in range(sample_size):
                    print(f"    Row {row}: {tensor[row][:5]}...")

    # Visualize model structure
    print("\n=== Model Structure Visualization ===")
    # Create a dictionary to track connections
    node_connections = {}
    
    # Map from tensor name to source node index
    tensor_to_producer = {}
    for i, node in enumerate(model.graph.node):
        for output in node.output:
            tensor_to_producer[output] = i
    
    # Build connections
    for i, node in enumerate(model.graph.node):
        node_connections[i] = []
        for input_tensor in node.input:
            if input_tensor in tensor_to_producer:
                node_connections[i].append(tensor_to_producer[input_tensor])
    
    # Print connections
    print("Node connections (node_index -> input_nodes):")
    for node_idx, input_nodes in sorted(node_connections.items()):
        if input_nodes:
            print(f"  Node {node_idx} ({model.graph.node[node_idx].op_type}) receives input from: {input_nodes}")
    
    # Identify potential consensus mechanism
    print("\n=== Potential Consensus Layer Analysis ===")
    
    # Look for patterns like concatenation followed by reduction operations
    concat_nodes = [i for i, node in enumerate(model.graph.node) if node.op_type == "Concat"]
    reduction_nodes = [i for i, node in enumerate(model.graph.node) if node.op_type in ["ArgMax", "ReduceMax", "ReduceMean", "Softmax"]]
    
    if concat_nodes:
        print(f"Found {len(concat_nodes)} concatenation nodes that might combine multiple model outputs.")
        for i in concat_nodes:
            node = model.graph.node[i]
            print(f"  Node {i}: {node.op_type}")
            print(f"    Inputs: {node.input}")
            print(f"    Outputs: {node.output}")
            # Check if this node's output goes to a reduction operation
            for j in reduction_nodes:
                if any(out in model.graph.node[j].input for out in node.output):
                    print(f"    -> Feeds into reduction node {j} ({model.graph.node[j].op_type})")
    
    if reduction_nodes:
        print(f"Found {len(reduction_nodes)} reduction nodes that might implement the consensus logic.")
        for i in reduction_nodes:
            node = model.graph.node[i]
            print(f"  Node {i}: {node.op_type}")
            print(f"    Inputs: {node.input}")
            print(f"    Outputs: {node.output}")
    
    # Look for fully connected layers that might be doing the final classification
    linear_nodes = [i for i, node in enumerate(model.graph.node) if node.op_type in ["MatMul", "Gemm"]]
    if linear_nodes:
        print(f"Found {len(linear_nodes)} linear operation nodes that might perform classification.")
        for i in linear_nodes:
            node = model.graph.node[i]
            print(f"  Node {i}: {node.op_type}")
            print(f"    Inputs: {node.input}")
            print(f"    Outputs: {node.output}")
            
            # Try to find associated weights
            for input_name in node.input[1:]:  # Skip the first input (data)
                weight_initializers = [init for init in model.graph.initializer if init.name == input_name]
                if weight_initializers:
                    weight = numpy_helper.to_array(weight_initializers[0])
                    print(f"    Weight shape: {weight.shape}")
                    if len(weight.shape) == 2:
                        print(f"    This appears to be a classification layer with {weight.shape[1]} output classes")

def main():
    parser = argparse.ArgumentParser(description='Analyze an ONNX model to understand its architecture')
    parser.add_argument('--model', type=str, required=True, help='Path to the ONNX model file')
    args = parser.parse_args()
    
    analyze_onnx_model(args.model)

if __name__ == '__main__':
    main()
