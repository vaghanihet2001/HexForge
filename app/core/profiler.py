import torch
import time

def profile_model(model, input_shape):
    """
    Runs a dummy tensor through the model layer-by-layer.
    Measures execution time, extracts output shapes, and pinpoints errors to specific nodes.
    """
    if not isinstance(input_shape, list) or len(input_shape) == 0:
        input_shape = [1, 3, 224, 224]

    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    x = torch.randn(*input_shape).to(device)
    
    profiling_results = []
    
    # Warm up / run once to check for errors and initialize memory
    temp_x = x.clone()
    for name, layer in model.named_children():
        node_id = name.split('_')[-1] if '_' in name else name
        try:
            with torch.no_grad():
                temp_x = layer(temp_x)
        except Exception as e:
            raise RuntimeError(f"Node execution failed at node_id: {node_id}. Error: {str(e)}") from e
            
    # Measure time of forward execution layer-by-layer
    for name, layer in model.named_children():
        node_id = name.split('_')[-1] if '_' in name else name
        
        # Extract input shape
        in_shape_val = None
        if isinstance(x, torch.Tensor):
            in_shape_val = list(x.shape)
        elif isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            in_shape_val = list(x[0].shape)

        # We know it won't fail here since the warmup succeeded
        t0 = time.perf_counter()
        with torch.no_grad():
            out = layer(x)
        t1 = time.perf_counter()
        duration_ms = (t1 - t0) * 1000
        
        # Extract output shape
        shape_val = None
        if isinstance(out, torch.Tensor):
            shape_val = list(out.shape)
        elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            shape_val = list(out[0].shape)
        
        x = out # output becomes input to next layer
        
        profiling_results.append({
            "name": name,
            "node_id": node_id,
            "duration_ms": round(duration_ms, 4),
            "shape": shape_val,
            "input_shape": in_shape_val
        })
        
    return profiling_results
