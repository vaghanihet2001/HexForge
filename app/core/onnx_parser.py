import onnx
from onnx import shape_inference, numpy_helper

def parse_onnx_model(file_path):
    """
    Loads and parses an ONNX model, extracting metadata, inputs, outputs,
    and detailed node descriptions including attributes, outputs, weights,
    biases, and constants directly associated with their consumer nodes.
    """
    model = onnx.load(file_path)
    
    # Run shape inference to retrieve intermediate tensor shapes
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: ONNX shape inference failed: {e}")
        
    graph = model.graph
    
    # Identify model weights (initializers) to differentiate from actual inputs
    initializer_names = {init.name for init in graph.initializer}
    
    # Extract all initializers
    initializers = {}
    for init in graph.initializer:
        try:
            tensor = numpy_helper.to_array(init)
            initializers[init.name] = tensor.tolist()
        except Exception as e:
            initializers[init.name] = f"<Error parsing initializer: {e}>"

    # Extract all Constant node values and map them
    for node in graph.node:
        if node.op_type == "Constant":
            out_name = node.output[0] if node.output else None
            if out_name:
                for attr in node.attribute:
                    if attr.name == "value":
                        try:
                            tensor = numpy_helper.to_array(attr.t)
                            initializers[out_name] = tensor.tolist()
                        except Exception as e:
                            try:
                                initializers[out_name] = attr.floats or attr.ints or attr.strings or str(attr)
                            except Exception:
                                initializers[out_name] = f"<Error parsing constant: {e}>"

    # Helper to extract shape from ValueInfoProto
    def get_tensor_shape(value_info):
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField('shape'):
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            return shape
        return None

    # Extract model inputs (excluding weights)
    inputs = []
    for inp in graph.input:
        if inp.name not in initializer_names:
            inputs.append({
                "name": inp.name,
                "shape": get_tensor_shape(inp)
            })
            
    # Extract model outputs
    outputs = []
    for out in graph.output:
        outputs.append({
            "name": out.name,
            "shape": get_tensor_shape(out)
        })
        
    # Map to store tensor shapes (inputs + intermediate + outputs)
    tensor_shapes = {}
    for inp in graph.input:
        shape = get_tensor_shape(inp)
        if shape is not None:
            tensor_shapes[inp.name] = shape
            
    for val in graph.value_info:
        shape = get_tensor_shape(val)
        if shape is not None:
            tensor_shapes[val.name] = shape
            
    for out in graph.output:
        shape = get_tensor_shape(out)
        if shape is not None:
            tensor_shapes[out.name] = shape

    # Extract nodes
    nodes = []
    for i, node in enumerate(graph.node):
        # Skip Constant nodes from being rendered as separate nodes in playground
        if node.op_type == "Constant":
            continue

        # Extract attributes
        attrs = {}
        for attr in node.attribute:
            val = None
            if attr.HasField('f'):
                val = attr.f
            elif attr.HasField('i'):
                val = attr.i
            elif attr.HasField('s'):
                val = attr.s.decode('utf-8', errors='ignore') if isinstance(attr.s, bytes) else attr.s
            elif attr.floats:
                val = list(attr.floats)
            elif attr.ints:
                val = list(attr.ints)
            elif attr.strings:
                val = [s.decode('utf-8', errors='ignore') if isinstance(s, bytes) else s for s in attr.strings]
            elif attr.HasField('t'):
                try:
                    tensor = attr.t
                    total_elements = 1
                    for dim in tensor.dims:
                        total_elements *= dim
                    if total_elements < 100:
                        ndarray = numpy_helper.to_array(tensor)
                        val = ndarray.tolist()
                    else:
                        val = f"<Tensor shape={list(tensor.dims)}>"
                except Exception:
                    val = f"<Tensor shape={list(attr.t.dims)}>"
            attrs[attr.name] = val
            
        # Separate constants/weights from active connection inputs
        node_constants = {}
        active_inputs = []
        for inp_name in node.input:
            if inp_name in initializers:
                node_constants[inp_name] = initializers[inp_name]
            else:
                active_inputs.append(inp_name)

        # Get output shapes for this node's outputs
        node_output_shapes = {}
        for out_name in node.output:
            if out_name in tensor_shapes:
                node_output_shapes[out_name] = tensor_shapes[out_name]

        # Generate unique fallback name if none exists
        node_name = node.name if node.name else f"{node.op_type}_{i}"
        
        nodes.append({
            "name": node_name,
            "op_type": node.op_type,
            "inputs": active_inputs,
            "outputs": list(node.output),
            "attributes": attrs,
            "constants": node_constants,
            "output_shapes": node_output_shapes
        })
        
    model_info = {
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "model_version": model.model_version,
        "description": model.doc_string
    }
    
    return {
        "model_info": model_info,
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes
    }
