#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import nibabel as nib
from tinygrad.tensor import Tensor
import onnx
from tinyonnx import OnnxRunner

def benchmark_consensus_model(consensus_model_path, input_shape=(256, 256, 256), num_channels=22, 
                             chunk_sizes=[1, 10, 100, 1000, 10000], 
                             batch_sizes=[1], 
                             num_samples=1000, 
                             warmup=True):
    """
    Benchmark the consensus model with different chunk sizes and batch sizes.
    
    Args:
        consensus_model_path: Path to the consensus model ONNX file
        input_shape: Shape of the input volume (height, width, depth)
        num_channels: Number of channels in the combined data
        chunk_sizes: List of chunk sizes to benchmark
        batch_sizes: List of batch sizes to benchmark
        num_samples: Number of samples to process (to avoid running the entire volume)
        warmup: Whether to perform a warmup run
        
    Returns:
        Dictionary with benchmark results
    """
    # Load model
    print(f"Loading consensus model from {consensus_model_path}")
    consensus_model = onnx.load(consensus_model_path)
    consensus_runner = OnnxRunner(consensus_model)
    
    # Determine input name
    input_name = consensus_model.graph.input[0].name
    print(f"Model input name: {input_name}")
    
    # Print model details
    print(f"Model input shape: {[dim.dim_value for dim in consensus_model.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"Model output shape: {[dim.dim_value for dim in consensus_model.graph.output[0].type.tensor_type.shape.dim]}")
    
    # Generate random data for benchmark
    print(f"Generating random data with shape {input_shape} and {num_channels} channels")
    height, width, depth = input_shape
    total_voxels = height * width * depth
    
    # Only generate the number of samples we need
    if num_samples < total_voxels:
        print(f"Using {num_samples} random samples instead of all {total_voxels} voxels")
        combined_data_flat = np.random.rand(num_samples, num_channels).astype(np.float32)
    else:
        print(f"Using all {total_voxels} voxels")
        combined_data = np.random.rand(height, width, depth, num_channels).astype(np.float32)
        combined_data_flat = combined_data.reshape(-1, num_channels)
        num_samples = total_voxels
    
    # Pre-calculate indices for the entire dataset
    all_indices = np.arange(num_samples)
    i_indices = all_indices // (width * depth)
    j_indices = (all_indices % (width * depth)) // depth
    k_indices = all_indices % depth
    
    results = {}
    
    # Warmup run to eliminate JIT compilation time
    if warmup:
        print("Performing warmup run...")
        warmup_data = np.random.rand(1, 1, 1, 1, num_channels).astype(np.float32)
        warmup_tensor = Tensor(warmup_data, requires_grad=False)
        _ = consensus_runner({input_name: warmup_tensor})
    
    # Benchmark different configurations
    for batch_size in batch_sizes:
        batch_results = {}
        
        for chunk_size in chunk_sizes:
            print(f"\nBenchmarking with batch_size={batch_size}, chunk_size={chunk_size}")
            
            # Initialize output array
            output = np.zeros((height, width, depth), dtype=np.int64)
            
            # Start timing
            start_time = time.time()
            
            # Process in chunks
            for chunk_start in tqdm(range(0, num_samples, chunk_size)):
                chunk_end = min(chunk_start + chunk_size, num_samples)
                chunk_size_actual = chunk_end - chunk_start
                
                # Get indices for this chunk
                chunk_i = i_indices[chunk_start:chunk_end]
                chunk_j = j_indices[chunk_start:chunk_end]
                chunk_k = k_indices[chunk_start:chunk_end]
                
                # Get data for this chunk
                chunk_data = combined_data_flat[chunk_start:chunk_end]
                
                if batch_size == 1:
                    # Process one voxel at a time
                    for idx in range(chunk_size_actual):
                        # Extract voxel data
                        voxel_data = chunk_data[idx]
                        
                        # Reshape to expected input shape [1, 1, 1, 1, channels]
                        X = voxel_data.reshape(1, 1, 1, 1, -1)
                        
                        # Create TinyGrad tensor
                        X_tensor = Tensor(X, requires_grad=False)
                        
                        # Run inference
                        outputs = consensus_runner({input_name: X_tensor})
                        
                        # Get output tensor
                        output_tensor = list(outputs.values())[0]
                        output_data = output_tensor.numpy()
                        
                        # Get prediction
                        if len(output_data.shape) == 5:  # [1, 1, 1, 1, classes]
                            prediction = np.argmax(output_data[0, 0, 0, 0])
                        else:
                            prediction = output_data[0, 0, 0, 0]
                            
                        # Store prediction
                        if chunk_i[idx] < height and chunk_j[idx] < width and chunk_k[idx] < depth:
                            output[chunk_i[idx], chunk_j[idx], chunk_k[idx]] = prediction
                else:
                    # Process in batches (if the model supports it)
                    for batch_start in range(0, chunk_size_actual, batch_size):
                        batch_end = min(batch_start + batch_size, chunk_size_actual)
                        batch_data = chunk_data[batch_start:batch_end]
                        
                        # Reshape to expected input shape [batch, 1, 1, 1, channels]
                        X = batch_data.reshape(-1, 1, 1, 1, num_channels)
                        
                        # Create TinyGrad tensor
                        X_tensor = Tensor(X, requires_grad=False)
                        
                        try:
                            # Run inference
                            outputs = consensus_runner({input_name: X_tensor})
                            
                            # Get output tensor
                            output_tensor = list(outputs.values())[0]
                            output_data = output_tensor.numpy()
                            
                            # Get predictions
                            if len(output_data.shape) == 5:  # [batch, 1, 1, 1, classes]
                                batch_preds = np.argmax(output_data[:, 0, 0, 0], axis=1)
                            else:
                                batch_preds = output_data[:, 0, 0, 0]
                            
                            # Store predictions
                            for idx in range(batch_end - batch_start):
                                i = chunk_i[batch_start + idx]
                                j = chunk_j[batch_start + idx]
                                k = chunk_k[batch_start + idx]
                                if i < height and j < width and k < depth:
                                    output[i, j, k] = batch_preds[idx]
                        except Exception as e:
                            print(f"Error with batch size {batch_size}: {e}")
                            print("Your model likely doesn't support batching. Try with batch_size=1.")
                            break
            
            # End timing
            end_time = time.time()
            elapsed_time = end_time - start_time
            voxels_per_second = num_samples / elapsed_time
            
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"Voxels per second: {voxels_per_second:.2f}")
            
            batch_results[chunk_size] = {
                'elapsed_time': elapsed_time,
                'voxels_per_second': voxels_per_second
            }
        
        results[batch_size] = batch_results
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark consensus model for MRI segmentation')
    parser.add_argument('--model', type=str, default="models/consensus_layer.onnx", help='Path to consensus model ONNX file')
    parser.add_argument('--shape', type=int, nargs=3, default=[256, 256, 256], help='Input shape (height, width, depth)')
    parser.add_argument('--channels', type=int, default=22, help='Number of input channels')
    parser.add_argument('--samples', type=int, default=50000, help='Number of samples to process')
    parser.add_argument('--chunk-sizes', type=int, nargs='+', default=[1, 10, 100, 1000, 10000], help='Chunk sizes to benchmark')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1], help='Batch sizes to benchmark')
    parser.add_argument('--no-warmup', action='store_true', help='Skip warmup run')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\nConsensus Model Benchmark")
    print("=========================")
    print(f"Model: {args.model}")
    print(f"Input shape: {args.shape}")
    print(f"Input channels: {args.channels}")
    print(f"Samples: {args.samples}")
    print(f"Chunk sizes: {args.chunk_sizes}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warmup: {not args.no_warmup}")
    print("=========================\n")
    
    # Run benchmark
    results = benchmark_consensus_model(
        consensus_model_path=args.model,
        input_shape=tuple(args.shape),
        num_channels=args.channels,
        chunk_sizes=args.chunk_sizes,
        batch_sizes=args.batch_sizes,
        num_samples=args.samples,
        warmup=not args.no_warmup
    )
    
    # Print summary of results
    print("\nBenchmark Results Summary")
    print("=========================")
    for batch_size, batch_results in results.items():
        print(f"\nBatch size: {batch_size}")
        print("------------------")
        print("{:<10} {:<15} {:<15}".format("Chunk Size", "Time (s)", "Voxels/s"))
        print("-" * 40)
        for chunk_size, metrics in batch_results.items():
            print("{:<10} {:<15.2f} {:<15.2f}".format(
                chunk_size, 
                metrics['elapsed_time'], 
                metrics['voxels_per_second']
            ))
    
if __name__ == '__main__':
    main()
