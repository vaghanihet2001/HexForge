import torch
import time

def profile_model(model, input_shape):
    """
    Runs a dummy tensor through the model and measures execution time per layer using hooks.
    """
    if not isinstance(input_shape, list) or len(input_shape) == 0:
        input_shape = [1, 3, 224, 224]

    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    dummy_input = torch.randn(*input_shape).to(device)
    
    profiling_results = []
    
    # Store timing details here
    start_times = {}
    end_times = {}
    
    # Define hooks
    def pre_hook(module, module_input, module_name):
        start_times[module_name] = time.perf_counter()
        
    def post_hook(module, module_input, output, module_name):
        end_times[module_name] = time.perf_counter()
        
        duration_ms = (end_times[module_name] - start_times[module_name]) * 1000
        
        # Extract the original node ID from the module name e.g., "conv2d_4" -> "4"
        node_id = module_name.split('_')[-1] if '_' in module_name else module_name
        
        profiling_results.append({
            "name": module_name,
            "node_id": node_id,
            "duration_ms": round(duration_ms, 4)
        })

    # Register hooks on child modules
    hook_handles = []
    for name, layer in model.named_children():
        h1 = layer.register_forward_pre_hook(lambda m, i, n=name: pre_hook(m, i, n))
        h2 = layer.register_forward_hook(lambda m, i, o, n=name: post_hook(m, i, o, n))
        hook_handles.extend([h1, h2])
        
    # Run the model once for warmup to eliminate initialization overhead
    try:
        with torch.no_grad():
            _ = model(dummy_input)
            
        start_times.clear()
        end_times.clear()
        profiling_results.clear()
        
        # Run the model to profile
        with torch.no_grad():
            _ = model(dummy_input)
            
    except Exception as e:
        # Clean up hooks before raising
        for handle in hook_handles:
            handle.remove()
        raise RuntimeError(f"Model validation/profiling failed: {str(e)}")
        
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
        
    return profiling_results
