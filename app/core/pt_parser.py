import torch
import torch.nn as nn

def parse_pt_model(file_path):
    """
    Loads a PyTorch model (.pt) and parses it into the standard HEXForge JSON structure.
    It registers forward hooks to infer intermediate tensor shapes, and extracts
    parameters (weights, biases) as constants.
    """
    # Load model
    model = torch.load(file_path, map_location='cpu', weights_only=False)
    if not isinstance(model, nn.Module):
        if isinstance(model, dict):
            # Try to find an nn.Module inside the dictionary
            for k, v in model.items():
                if isinstance(v, nn.Module):
                    model = v
                    break
            else:
                raise ValueError("The uploaded file does not contain a PyTorch nn.Module instance.")
        else:
            raise ValueError("The uploaded file is not a PyTorch nn.Module instance.")

    model.eval()

    # Infer input shape by looking at first layer
    input_shape = [1, 3, 224, 224]
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            input_shape = [1, module.in_channels, 224, 224]
            break
        elif isinstance(module, nn.Linear):
            input_shape = [1, module.in_features]
            break

    # Extract shapes via forward hooks
    shapes = {}
    def hook_fn(module, input, output):
        module_id = id(module)
        in_shape = list(input[0].shape) if input and torch.is_tensor(input[0]) else None
        out_shape = list(output.shape) if torch.is_tensor(output) else None
        shapes[module_id] = (in_shape, out_shape)

    hooks = []
    # Register hooks on immediate named children
    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(hook_fn))

    # Run single forward pass to trigger hooks and resolve shapes
    try:
        dummy_input = torch.randn(*input_shape)
        with torch.no_grad():
            model(dummy_input)
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()

    nodes = []
    inputs = [{"name": "input", "shape": input_shape}]
    outputs = []
    current_tensor = "input"

    for i, (name, module) in enumerate(model.named_children()):
        module_id = id(module)
        in_shape, out_shape = shapes.get(module_id, (None, None))

        # Map module class name to ONNX op_type
        class_name = module.__class__.__name__
        op_type = class_name
        attrs = {}

        if isinstance(module, nn.Conv1d):
            op_type = "Conv1d"
            attrs["strides"] = list(module.stride)
            attrs["pads"] = list(module.padding)
            attrs["kernel_shape"] = list(module.kernel_size)
        elif isinstance(module, nn.Conv2d):
            op_type = "Conv"
            attrs["strides"] = list(module.stride)
            attrs["pads"] = list(module.padding)
            attrs["kernel_shape"] = list(module.kernel_size)
        elif isinstance(module, nn.ConvTranspose2d):
            op_type = "ConvTranspose"
            attrs["strides"] = list(module.stride)
            attrs["pads"] = list(module.padding)
            attrs["kernel_shape"] = list(module.kernel_size)
        elif isinstance(module, nn.Linear):
            op_type = "Gemm"
        elif isinstance(module, nn.ReLU):
            op_type = "Relu"
        elif isinstance(module, nn.LeakyReLU):
            op_type = "LeakyRelu"
            attrs["alpha"] = module.negative_slope
        elif isinstance(module, nn.Sigmoid):
            op_type = "Sigmoid"
        elif isinstance(module, nn.Tanh):
            op_type = "Tanh"
        elif isinstance(module, nn.GELU):
            op_type = "Gelu"
        elif isinstance(module, nn.SiLU):
            op_type = "SiLU"
        elif isinstance(module, nn.ELU):
            op_type = "Elu"
            attrs["alpha"] = module.alpha
        elif isinstance(module, nn.Softmax):
            op_type = "Softmax"
            attrs["axis"] = module.dim
        elif isinstance(module, nn.MaxPool2d):
            op_type = "MaxPool"
            attrs["strides"] = [module.stride] if isinstance(module.stride, int) else list(module.stride)
            attrs["kernel_shape"] = [module.kernel_size] if isinstance(module.kernel_size, int) else list(module.kernel_size)
        elif isinstance(module, nn.AvgPool2d):
            op_type = "AveragePool"
            attrs["strides"] = [module.stride] if isinstance(module.stride, int) else list(module.stride)
            attrs["kernel_shape"] = [module.kernel_size] if isinstance(module.kernel_size, int) else list(module.kernel_size)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            op_type = "GlobalAveragePool"
        elif isinstance(module, nn.BatchNorm2d):
            op_type = "BatchNormalization"
            attrs["epsilon"] = module.eps
            attrs["momentum"] = module.momentum
        elif isinstance(module, nn.Dropout):
            op_type = "Dropout"
            attrs["ratio"] = module.p
        elif isinstance(module, nn.Flatten):
            op_type = "Flatten"
            attrs["axis"] = module.start_dim
        elif isinstance(module, nn.PReLU):
            op_type = "PRelu"

        # Extract weight/bias parameters as constants
        constants = {}
        for p_name, param in module.named_parameters():
            # Convert tensors to list for JSON serialization
            constants[p_name] = param.detach().cpu().numpy().tolist()

        node_input = current_tensor
        node_output = f"tensor_{i}"
        current_tensor = node_output

        nodes.append({
            "name": name,
            "op_type": op_type,
            "inputs": [node_input],
            "outputs": [node_output],
            "attributes": attrs,
            "constants": constants,
            "output_shapes": {node_output: out_shape} if out_shape else {}
        })

    if nodes:
        last_out_name = nodes[-1]["outputs"][0]
        last_out_shape = nodes[-1]["output_shapes"].get(last_out_name, [])
        outputs.append({"name": last_out_name, "shape": last_out_shape})

    model_info = {
        "producer_name": "PyTorch (HEXForge Imported)",
        "producer_version": torch.__version__,
        "ir_version": "N/A",
        "model_version": "1",
        "description": str(model)
    }

    return {
        "model_info": model_info,
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes
    }
