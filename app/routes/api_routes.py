from flask import Blueprint, jsonify, request

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "success", "message": "API is standing by."})

# Placeholder endpoints for future modules
@api_bp.route('/build', methods=['POST'])
def build_model_route():
    data = request.json
    graph_data = data.get('graph', [])
    
    try:
        from app.core.builder import build_model
        import torch
        
        # Build model dynamically
        model, input_shape, _ = build_model(graph_data)
        
        # Validate shapes by passing dummy data
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        dummy_input = torch.randn(*input_shape).to(device)
        
        with torch.no_grad():
            _ = model(dummy_input)
            
        return jsonify({
            "status": "success", 
            "message": "Model built and dimensions validated successfully!"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 400

@api_bp.route('/profile', methods=['POST'])
def profile_model_route():
    data = request.json
    graph_data = data.get('graph', [])
    
    try:
        from app.core.builder import build_model
        from app.core.profiler import profile_model
        import traceback
        
        # Build model dynamically
        model, input_shape, _ = build_model(graph_data)
        
        # Profile model
        results = profile_model(model, input_shape)
        
        return jsonify({
            "status": "success", 
            "message": "Profiling completed successfully",
            "results": results
        })
    except Exception as e:
        import traceback
        err_msg = str(e)
        error_node_id = None
        if "Node execution failed at node_id: " in err_msg:
            try:
                parts = err_msg.split("Node execution failed at node_id: ")
                error_node_id = parts[1].split(".")[0].strip()
            except Exception:
                pass
                
        return jsonify({
            "status": "error",
            "message": err_msg,
            "error_node_id": error_node_id,
            "traceback": traceback.format_exc()
        }), 400

@api_bp.route('/infer-shapes', methods=['POST'])
def infer_shapes_route():
    data = request.json
    graph_data = data.get('graph', [])
    results = []
    
    try:
        from app.core.builder import build_model
        import torch
        import traceback
        
        # Build model dynamically
        model, input_shape, _ = build_model(graph_data)
        
        # Trace shapes layer-by-layer
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        x = torch.randn(*input_shape).to(device)
        
        for name, layer in model.named_children():
            node_id = name.split('_')[-1] if '_' in name else name
            
            in_shape_val = list(x.shape) if isinstance(x, torch.Tensor) else None
            
            try:
                with torch.no_grad():
                    out = layer(x)
            except Exception as e:
                raise RuntimeError(f"Node execution failed at node_id: {node_id}. Error: {str(e)}") from e
                
            shape_val = list(out.shape) if isinstance(out, torch.Tensor) else None
            x = out
            
            results.append({
                "node_id": node_id,
                "shape": shape_val,
                "input_shape": in_shape_val
            })
            
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        err_msg = str(e)
        error_node_id = None
        if "node_id: " in err_msg:
            try:
                parts = err_msg.split("node_id: ")
                error_node_id = parts[1].split(".")[0].split(" ")[0].split("\n")[0].split(",")[0].strip()
            except Exception:
                pass
        return jsonify({
            "status": "error",
            "message": err_msg,
            "error_node_id": error_node_id,
            "results": results
        }), 400

@api_bp.route('/evaluate', methods=['POST'])
def evaluate_model_route():
    from werkzeug.utils import secure_filename
    import json
    import os
    from flask import current_app
    from app.core.builder import build_model
    from app.core.evaluator import extract_dataset, evaluate_model
    import traceback
    import uuid

    try:
        # Get graph data from form
        graph_data_str = request.form.get('graph')
        if not graph_data_str:
            return jsonify({"status": "error", "message": "Graph data missing"}), 400
            
        graph_data = json.loads(graph_data_str)
        
        # Get uploaded dataset
        if 'dataset' not in request.files:
            return jsonify({"status": "error", "message": "No dataset file uploaded"}), 400
            
        dataset_file = request.files['dataset']
        if dataset_file.filename == '':
            return jsonify({"status": "error", "message": "Empty filename"}), 400
            
        if not dataset_file.filename.endswith('.zip'):
            return jsonify({"status": "error", "message": "Dataset must be a .zip file"}), 400

        # Save and extract dataset safely
        unique_id = str(uuid.uuid4())
        zip_filename = secure_filename(dataset_file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{unique_id}_{zip_filename}")
        extract_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_id)
        
        dataset_file.save(upload_path)
        
        # Build Model
        model, input_shape, _ = build_model(graph_data)
        
        # Extract and Evaluate
        dataset_root = extract_dataset(upload_path, extract_path)
        results = evaluate_model(model, input_shape, dataset_root)
        
        return jsonify({
            "status": "success",
            "message": "Evaluation Complete",
            "results": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 400

@api_bp.route('/export', methods=['POST'])
def export_model():
    import traceback
    import io
    from flask import send_file
    from app.core.builder import build_model
    import torch
    
    try:
        data = request.json
        graph_data = data.get('graph', [])
        
        # Build model dynamically
        model, input_shape, _ = build_model(graph_data)
        model.eval()
        
        # Validate shapes by passing dummy data
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        dummy_input = torch.randn(*input_shape).to(device)
        
        # Export model to an in-memory buffer
        onnx_buffer = io.BytesIO()
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_buffer, 
            input_names=["input_tensor"], 
            output_names=["output_tensor"],
            opset_version=12
        )
        onnx_buffer.seek(0)
        
        return send_file(
            onnx_buffer,
            as_attachment=True,
            download_name="hexforge_model.onnx",
            mimetype="application/octet-stream"
        )
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 400

@api_bp.route('/inspect-onnx', methods=['POST'])
def inspect_onnx_route():
    from werkzeug.utils import secure_filename
    import os
    import traceback
    import uuid
    from flask import current_app
    from app.core.onnx_parser import parse_onnx_model

    try:
        if 'onnx_file' not in request.files:
            return jsonify({"status": "error", "message": "No ONNX file uploaded"}), 400
            
        file = request.files['onnx_file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "Empty filename"}), 400
            
        if not file.filename.endswith('.onnx'):
            return jsonify({"status": "error", "message": "File must have .onnx extension"}), 400
            
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        
        file.save(upload_path)
        
        # Parse model
        parsed_data = parse_onnx_model(upload_path)
        
        # Clean up file after parsing
        try:
            os.remove(upload_path)
        except Exception as ex:
            print(f"Warning: could not delete temporary ONNX file {upload_path}: {ex}")
            
        return jsonify({
            "status": "success",
            "message": "ONNX parsed successfully",
            "data": parsed_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 400

