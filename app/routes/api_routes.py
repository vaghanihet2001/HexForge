from flask import Blueprint, jsonify, request
import traceback
import logging

logger = logging.getLogger('hexforge.api')

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "success", "message": "API is standing by."})

# Placeholder endpoints for future modules
@api_bp.route('/build', methods=['POST'])
def build_model_route():
    data = request.json
    graph_data = data.get('graph', [])
    logger.info("[BUILD] Model build requested | nodes=%d", len(graph_data) if isinstance(graph_data, list) else 0)
    
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

        logger.info("[BUILD] Model built successfully | input_shape=%s", input_shape)
        return jsonify({
            "status": "success", 
            "message": "Model built and dimensions validated successfully!"
        })
    except Exception as e:
        logger.error("[BUILD] Model build failed: %s", str(e), exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 400

@api_bp.route('/profile', methods=['POST'])
def profile_model_route():
    data = request.json
    graph_data = data.get('graph', [])
    num_iterations = int(data.get('num_iterations', 100))
    
    try:
        from app.core.builder import build_model
        from app.core.profiler import profile_model
        import traceback
        
        # Build model dynamically
        model, input_shape, _ = build_model(graph_data)
        
        # Profile model
        results = profile_model(model, input_shape, num_iterations=num_iterations)
        
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
        
        version_id = request.form.get('version_id')
        upload_path = None
        extract_path = None
        unique_id = str(uuid.uuid4())

        if 'dataset' in request.files and request.files['dataset'].filename != '':
            dataset_file = request.files['dataset']
            if not dataset_file.filename.endswith('.zip'):
                return jsonify({"status": "error", "message": "Dataset must be a .zip file"}), 400
            zip_filename = secure_filename(dataset_file.filename)
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{unique_id}_{zip_filename}")
            extract_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_id)
            dataset_file.save(upload_path)
        else:
            if version_id:
                version = dbh.get_version(version_id)
                if version:
                    project = dbh.get_project(version["project_id"])
                    if project and project.get("dataset") and os.path.exists(project["dataset"]["zip_path"]):
                        upload_path = project["dataset"]["zip_path"]
                        extract_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_id)
                    else:
                        return jsonify({"status": "error", "message": "No dataset uploaded for this project, and no zip file was provided."}), 400
                else:
                    return jsonify({"status": "error", "message": "Version not found."}), 400
            else:
                return jsonify({"status": "error", "message": "No dataset file uploaded, and no version ID provided."}), 400

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

    try:
        if 'onnx_file' not in request.files:
            return jsonify({"status": "error", "message": "No model file uploaded"}), 400
            
        file = request.files['onnx_file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "Empty filename"}), 400
            
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.onnx', '.pt', '.pth']:
            return jsonify({"status": "error", "message": "File must have .onnx, .pt, or .pth extension"}), 400
            
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        
        file.save(upload_path)
        
        # Parse model depending on file extension
        if ext == '.onnx':
            from app.core.onnx_parser import parse_onnx_model
            parsed_data = parse_onnx_model(upload_path)
            model_type = "ONNX"
        else:
            from app.core.pt_parser import parse_pt_model
            parsed_data = parse_pt_model(upload_path)
            model_type = "PyTorch"
        
        # Clean up file after parsing
        try:
            os.remove(upload_path)
        except Exception as ex:
            print(f"Warning: could not delete temporary model file {upload_path}: {ex}")
            
        return jsonify({
            "status": "success",
            "message": f"{model_type} model parsed successfully",
            "data": parsed_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 400



# ══════════════════════════════════════════════════════════════
#  TRAINING  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@api_bp.route('/train/start', methods=['POST'])
def train_start():
    """Multipart form: graph (JSON str) + dataset (zip) + config fields."""
    import json, os, uuid, traceback
    from flask import current_app
    from app.core import trainer as tr
    from datetime import datetime

    try:
        state = tr.get_state()
        if state["status"] == "running":
            return jsonify({"status": "error", "message": "Training already in progress."}), 400

        graph_str = request.form.get("graph")
        if not graph_str:
            return jsonify({"status": "error", "message": "Graph data missing"}), 400
        graph_data = json.loads(graph_str)

        version_id = request.form.get("version_id")
        dataset_id = request.form.get("dataset_id")
        pretrained_run_id = request.form.get("pretrained_run_id")
        run_name = request.form.get("run_name")

        zip_path = None
        extract_path = None
        uid          = str(uuid.uuid4())
        upload_dir   = current_app.config["UPLOAD_FOLDER"]

        # Determine Dataset Zip and Extract paths
        if "dataset" in request.files and request.files["dataset"].filename != '':
            dataset_file = request.files["dataset"]
            if not dataset_file.filename.endswith(".zip"):
                return jsonify({"status": "error", "message": "Dataset must be a .zip file"}), 400
            zip_path     = os.path.join(upload_dir, f"{uid}_train.zip")
            extract_path = os.path.join(upload_dir, f"{uid}_train_ds")
            dataset_file.save(zip_path)
            
            # Create a dataset record automatically for this upload
            if version_id:
                version = dbh.get_version(version_id)
                if version:
                    proj_id = version["project_id"]
                    from app.core.evaluator import inspect_classification_dataset
                    temp_extract = os.path.join(upload_dir, f"temp_inspect_{uid}")
                    try:
                        import shutil
                        # Temporary copy for inspection
                        shutil.copy(zip_path, zip_path + ".temp")
                        metadata = inspect_classification_dataset(zip_path + ".temp", temp_extract)
                        dataset_name = f"Upload {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
                        # Save permanent copy in datasets folder
                        dataset_dir = os.path.join(upload_dir, 'datasets', uid)
                        os.makedirs(dataset_dir, exist_ok=True)
                        perm_zip_path = os.path.join(dataset_dir, 'dataset.zip')
                        perm_extract_to = os.path.join(dataset_dir, 'dataset_extracted')
                        shutil.move(zip_path, perm_zip_path)
                        shutil.move(temp_extract, perm_extract_to)
                        
                        dataset_id = dbh.create_dataset(
                            project_id=proj_id,
                            dataset_name=dataset_name,
                            zip_path=perm_zip_path,
                            extract_path=perm_extract_to,
                            classes=metadata["classes"],
                            samples=metadata["num_samples"]
                        )
                        zip_path = perm_zip_path
                        extract_path = os.path.join(upload_dir, f"runs_extracted_{uid}")
                    except Exception as e:
                        return jsonify({"status": "error", "message": f"Invalid dataset ZIP: {str(e)}"}), 400
        elif dataset_id:
            dataset_ver = dbh.get_dataset(dataset_id)
            if dataset_ver and os.path.exists(dataset_ver["zip_path"]):
                zip_path = dataset_ver["zip_path"]
                extract_path = os.path.join(upload_dir, f"runs_extracted_{uid}")
            else:
                return jsonify({"status": "error", "message": "Selected dataset version not found or zip file missing."}), 400
        else:
            if version_id:
                version = dbh.get_version(version_id)
                if version:
                    # Look up latest versioned dataset for the project
                    project_datasets = dbh.get_datasets(version["project_id"])
                    if project_datasets:
                        latest_ds = project_datasets[0]
                        dataset_id = latest_ds["_id"]
                        zip_path = latest_ds["zip_path"]
                        extract_path = os.path.join(upload_dir, f"runs_extracted_{uid}")
                    else:
                        # Try to fallback to project legacy dataset if any
                        project = dbh.get_project(version["project_id"])
                        if project and project.get("dataset") and os.path.exists(project["dataset"]["zip_path"]):
                            zip_path = project["dataset"]["zip_path"]
                            extract_path = os.path.join(upload_dir, f"{uid}_train_ds")
                        else:
                            return jsonify({"status": "error", "message": "No dataset version found. Please upload a dataset first."}), 400
                else:
                    return jsonify({"status": "error", "message": "Version not found."}), 400
            else:
                return jsonify({"status": "error", "message": "No version ID provided."}), 400

        cfg = {
            "epochs":      int(request.form.get("epochs",      10)),
            "batch_size":  int(request.form.get("batch_size",  32)),
            "lr":        float(request.form.get("lr",         1e-3)),
            "weight_decay":float(request.form.get("weight_decay", 0.0)),
            "momentum":  float(request.form.get("momentum",   0.9)),
            "grad_clip": float(request.form.get("grad_clip",  0.0)),
            "val_split": float(request.form.get("val_split",  0.2)),
            "optimizer":       request.form.get("optimizer", "adam"),
            "loss":            request.form.get("loss", "crossentropy"),
            "scheduler":       request.form.get("scheduler", "none"),
            "augment":         request.form.get("augment", "false").lower() == "true",
        }

        if version_id:
            cfg["version_id"] = version_id
            version = dbh.get_version(version_id)
            proj_id = version["project_id"] if version else None
            
            # Create Run record
            if not run_name:
                run_name = f"Run #{len(dbh.get_runs(version_id)) + 1}"
            
            run_id = dbh.create_run(
                version_id=version_id,
                project_id=proj_id,
                run_name=run_name,
                dataset_id=dataset_id,
                hyperparameters=cfg,
                pretrained_run_id=pretrained_run_id
            )
            cfg["run_id"] = run_id
            if pretrained_run_id:
                cfg["pretrained_run_id"] = pretrained_run_id
                
            dbh.update_run_status(run_id, status="training")
            
            # Make sure run directory exists
            run_dir = os.path.join(upload_dir, 'runs', run_id)
            os.makedirs(run_dir, exist_ok=True)
            model_path = os.path.join(run_dir, "model_best.pt")
        else:
            run_id = None
            model_path = os.path.join(upload_dir, f"{uid}_best.pt")

        logger.info(
            "[TRAIN/START] Starting training run | run_id=%s | version_id=%s | epochs=%d | batch=%d",
            run_id, version_id, cfg["epochs"], cfg["batch_size"]
        )
        tr.start_training(graph_data, zip_path, extract_path, cfg, model_path)
        logger.info("[TRAIN/START] Training thread launched | run_id=%s | uid=%s", run_id, uid)
        return jsonify({"status": "success", "message": "Training started", "uid": uid, "run_id": run_id})

    except Exception as e:
        logger.error("[TRAIN/START] Training start failed: %s", str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e), "traceback": traceback.format_exc()}), 400


@api_bp.route('/train/status', methods=['GET'])
def train_status():
    """Returns training state for polling."""
    from app.core import trainer as tr
    import math
    state = tr.get_state()
    since = request.args.get("since", 0, type=int)
    logs  = state["logs"][since:]
    state["logs"]       = logs
    state["log_offset"] = since + len(logs)

    # Sanitize non-finite floats — Infinity/NaN are not valid JSON
    # and cause res.json() to throw in the browser, silently killing the poll loop
    bvl = state.get("best_val_loss")
    state["best_val_loss"] = None if (bvl is None or not math.isfinite(bvl)) else round(bvl, 6)
    for key in ("elapsed", "eta"):
        v = state.get(key)
        state[key] = 0.0 if (v is None or not math.isfinite(v)) else v

    # Surface any background thread error
    if state.get("status") == "error" and state.get("error"):
        logger.error("[TRAIN/STATUS] Background thread error: %s", state["error"])
    return jsonify(state)


@api_bp.route('/train/stop', methods=['POST'])
def train_stop():
    from app.core import trainer as tr
    tr.stop_training()
    return jsonify({"status": "success", "message": "Stop signal sent"})


@api_bp.route('/train/download', methods=['GET'])
def train_download():
    import os
    from flask import send_file
    from app.core import trainer as tr
    state = tr.get_state()
    path  = state.get("model_path")
    if not path or not os.path.exists(path):
        return jsonify({"status": "error", "message": "No trained model available yet."}), 404
    return send_file(path, as_attachment=True, download_name="hexforge_trained.pt",
                     mimetype="application/octet-stream")


# ══════════════════════════════════════════════════════════════
#  PROJECTS AND MODEL VERSIONS ENDPOINTS
# ══════════════════════════════════════════════════════════════

from app.core import db_helpers as dbh

@api_bp.route('/projects', methods=['GET'])
def get_projects_route():
    try:
        projects = dbh.get_projects()
        return jsonify({"status": "success", "projects": projects})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/projects', methods=['POST'])
def create_project_route():
    try:
        data = request.json
        name = data.get("name")
        description = data.get("description", "")
        if not name:
            return jsonify({"status": "error", "message": "Project name is required"}), 400
        
        project_id = dbh.create_project(name, description)
        # Create default v1 version automatically
        version_id = dbh.create_version(project_id, "v1")
        logger.info("[PROJECT] Created project '%s' | project_id=%s | version_id=%s", name, project_id, version_id)
        return jsonify({
            "status": "success",
            "message": "Project created successfully with default v1 version.",
            "project_id": project_id,
            "version_id": version_id
        })
    except Exception as e:
        logger.error("[PROJECT] Failed to create project: %s", str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/projects/<project_id>', methods=['GET'])
def get_project_route(project_id):
    try:
        project = dbh.get_project(project_id)
        if not project:
            return jsonify({"status": "error", "message": "Project not found"}), 404
        versions = dbh.get_versions(project_id)
        return jsonify({
            "status": "success",
            "project": project,
            "versions": versions
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/projects/<project_id>', methods=['DELETE'])
def delete_project_route(project_id):
    logger.info("[PROJECT] Delete requested | project_id=%s", project_id)
    try:
        success = dbh.delete_project(project_id)
        if success:
            logger.info("[PROJECT] Deleted | project_id=%s", project_id)
            return jsonify({"status": "success", "message": "Project deleted successfully"})
        logger.error("[PROJECT] Delete failed | project_id=%s", project_id)
        return jsonify({"status": "error", "message": "Failed to delete project"}), 400
    except Exception as e:
        logger.error("[PROJECT] Delete exception | project_id=%s | %s", project_id, str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/projects/<project_id>', methods=['PUT'])
def update_project_route(project_id):
    try:
        data = request.json
        name = data.get("name")
        description = data.get("description")
        if not name:
            return jsonify({"status": "error", "message": "Project name is required"}), 400
        success = dbh.update_project(project_id, name, description)
        if success:
            return jsonify({"status": "success", "message": "Project updated successfully"})
        return jsonify({"status": "error", "message": "Failed to update project"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/projects/<project_id>/versions', methods=['POST'])
def create_version_route(project_id):
    try:
        data = request.json
        version_name = data.get("version_name")
        from_version_id = data.get("from_version_id") # Optional: copy graph from an existing version
        
        if not version_name:
            return jsonify({"status": "error", "message": "Version name is required"}), 400
            
        graph = None
        if from_version_id:
            src_version = dbh.get_version(from_version_id)
            if src_version:
                graph = src_version.get("graph")
                
        version_id = dbh.create_version(project_id, version_name, graph=graph)
        logger.info("[VERSION] Created '%s' | project_id=%s | version_id=%s | copied_from=%s",
                    version_name, project_id, version_id, from_version_id or 'none')
        return jsonify({
            "status": "success",
            "message": f"Version '{version_name}' created successfully.",
            "version_id": version_id
        })
    except ValueError as ve:
        logger.error("[VERSION] Validation error | project_id=%s | %s", project_id, str(ve))
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        logger.error("[VERSION] Create failed | project_id=%s | %s", project_id, str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/versions/<version_id>', methods=['GET'])
def get_version_route(version_id):
    try:
        version = dbh.get_version(version_id)
        if not version:
            return jsonify({"status": "error", "message": "Version not found"}), 404
        return jsonify({"status": "success", "version": version})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/versions/<version_id>/graph', methods=['POST'])
def update_version_graph_route(version_id):
    try:
        data = request.json
        graph = data.get("graph")
        if not graph:
            return jsonify({"status": "error", "message": "Graph data is required"}), 400
            
        success = dbh.update_version_graph(version_id, graph)
        if success:
            logger.info("[VERSION] Graph saved | version_id=%s | nodes=%d",
                        version_id, len(graph.get('nodes', [])) if isinstance(graph, dict) else 0)
            return jsonify({"status": "success", "message": "Graph saved successfully"})
        logger.error("[VERSION] Graph save failed | version_id=%s", version_id)
        return jsonify({"status": "error", "message": "Failed to update graph"}), 400
    except Exception as e:
        logger.error("[VERSION] Graph update exception | version_id=%s | %s", version_id, str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/versions/<version_id>', methods=['DELETE'])
def delete_version_route(version_id):
    try:
        success = dbh.delete_version(version_id)
        if success:
            return jsonify({"status": "success", "message": "Version deleted successfully"})
        return jsonify({"status": "error", "message": "Failed to delete version"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/playground/test', methods=['POST'])
def playground_test():
    import traceback
    import torch
    from PIL import Image
    from app.core.builder import build_model
    import torch.nn.functional as F
    import os
    
    try:
        run_id = request.form.get("run_id")
        version_id = request.form.get("version_id")
        
        run = None
        version = None
        
        if run_id:
            run = dbh.get_run(run_id)
            if not run:
                return jsonify({"status": "error", "message": "Training run not found"}), 404
            version_id = run["version_id"]
            version = dbh.get_version(version_id)
        elif version_id:
            version = dbh.get_version(version_id)
            if version:
                # Find latest trained run
                runs = dbh.get_runs(version_id)
                trained_runs = [r for r in runs if r.get("status") == "trained" and r.get("model_path")]
                if trained_runs:
                    run = trained_runs[0]

        logger.info("[PLAYGROUND] Inference requested | run_id=%s | version_id=%s", run_id, version_id)
        if not version:
            return jsonify({"status": "error", "message": "Version not found"}), 404
            
        model_path = run.get("model_path") if run else version.get("model_path")
        if not model_path or not os.path.exists(model_path):
            return jsonify({"status": "error", "message": "Model has not been trained yet. Please launch training first."}), 400
            
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image file uploaded"}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"status": "error", "message": "Empty image filename"}), 400
            
        # Build Model Structure
        model, input_shape, _ = build_model(version["graph"])
        
        # Load state dict
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Load Image and transform
        img = Image.open(image_file).convert("RGB")
        
        channels = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        
        from torchvision import transforms
        transform_list = [
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ]
        if channels == 1:
            transform_list.insert(1, transforms.Grayscale(num_output_channels=1))
            
        data_transform = transforms.Compose(transform_list)
        x = data_transform(img).unsqueeze(0).to(device) # Shape [1, C, H, W]
        
        # Inference
        with torch.no_grad():
            outputs = model(x)
            
        classes = run.get("classes") if (run and run.get("classes")) else version.get("classes", [])
        
        # Handle predictions depending on shape of outputs
        if outputs.ndim == 2:
            num_classes = outputs.shape[1]
            if num_classes == 1:
                # Binary / Regression single output
                prob = torch.sigmoid(outputs)[0][0].item()
                class_idx = 1 if prob >= 0.5 else 0
                confidence = prob if class_idx == 1 else 1.0 - prob
                
                label = classes[class_idx] if class_idx < len(classes) else ("Positive" if class_idx == 1 else "Negative")
                all_probs = [
                    {"class": classes[0] if len(classes) > 0 else "Negative", "confidence": round((1.0 - prob) * 100, 2)},
                    {"class": classes[1] if len(classes) > 1 else "Positive", "confidence": round(prob * 100, 2)}
                ]
            else:
                # Multi-class output
                probabilities = F.softmax(outputs, dim=1)[0]
                confidence_tensor, class_idx_tensor = torch.max(probabilities, 0)
                confidence = confidence_tensor.item()
                class_idx = class_idx_tensor.item()
                
                label = classes[class_idx] if class_idx < len(classes) else f"Class {class_idx}"
                all_probs = []
                for idx, prob in enumerate(probabilities.cpu().tolist()):
                    lbl = classes[idx] if idx < len(classes) else f"Class {idx}"
                    all_probs.append({"class": lbl, "confidence": round(prob * 100, 2)})
                    
                # Sort descending by confidence
                all_probs = sorted(all_probs, key=lambda x: x["confidence"], reverse=True)
        else:
            # Fallback for unexpected shapes
            label = "Unknown Output Format"
            confidence = 0.0
            all_probs = []
            
        logger.info("[PLAYGROUND] Inference done | version_id=%s | prediction=%s | confidence=%.2f%%",
                    version_id, label, round(confidence * 100, 2))
        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "all_predictions": all_probs
        })
    except Exception as e:
        logger.error("[PLAYGROUND] Inference failed | version_id=%s | %s", version_id, str(e), exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 400

@api_bp.route('/projects/<project_id>/dataset', methods=['POST'])
def upload_project_dataset(project_id):
    import os, shutil
    from werkzeug.utils import secure_filename
    from flask import current_app
    from app.core.evaluator import inspect_classification_dataset
    from datetime import datetime
    
    try:
        project = dbh.get_project(project_id)
        if not project:
            return jsonify({"status": "error", "message": "Project not found"}), 404
            
        if 'dataset' not in request.files:
            return jsonify({"status": "error", "message": "No dataset file uploaded"}), 400
            
        dataset_file = request.files['dataset']
        if dataset_file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400
            
        if not dataset_file.filename.endswith('.zip'):
            return jsonify({"status": "error", "message": "Dataset must be a ZIP file"}), 400

        logger.info("[DATASET] Upload started | project_id=%s | file=%s", project_id, dataset_file.filename)
            
        project_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'projects', project_id)
        os.makedirs(project_dir, exist_ok=True)
        
        zip_path = os.path.join(project_dir, 'dataset.zip')
        extract_to = os.path.join(project_dir, 'dataset_extracted')
        
        # Save zip
        dataset_file.save(zip_path)
        
        # Validate and inspect
        try:
            metadata = inspect_classification_dataset(zip_path, extract_to)
        except Exception as e:
            logger.error("[DATASET] Validation failed | project_id=%s | %s", project_id, str(e))
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(extract_to):
                shutil.rmtree(extract_to)
            return jsonify({"status": "error", "message": f"Invalid classification dataset: {str(e)}"}), 400
            
        dataset_info = {
            "filename": secure_filename(dataset_file.filename),
            "uploaded_at": datetime.utcnow().isoformat(),
            "num_classes": metadata["num_classes"],
            "classes": metadata["classes"],
            "num_samples": metadata["num_samples"],
            "distribution": metadata["distribution"],
            "zip_path": zip_path,
            "extract_path": extract_to
        }
        
        dbh.update_project_dataset(project_id, dataset_info)
        logger.info("[DATASET] Upload complete | project_id=%s | classes=%d | samples=%d",
                    project_id, metadata["num_classes"], metadata["num_samples"])
        return jsonify({
            "status": "success",
            "message": "Dataset uploaded and processed successfully",
            "dataset": dataset_info
        })
    except Exception as e:
        logger.error("[DATASET] Upload exception | project_id=%s | %s", project_id, str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/projects/<project_id>/dataset', methods=['DELETE'])
def delete_project_dataset_route(project_id):
    import os, shutil
    from flask import current_app
    
    logger.info("[DATASET] Delete requested | project_id=%s", project_id)
    try:
        project = dbh.get_project(project_id)
        if not project:
            return jsonify({"status": "error", "message": "Project not found"}), 404
            
        project_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'projects', project_id)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            
        dbh.delete_project_dataset(project_id)
        logger.info("[DATASET] Deleted | project_id=%s", project_id)
        return jsonify({"status": "success", "message": "Dataset deleted successfully"})
    except Exception as e:
        logger.error("[DATASET] Delete exception | project_id=%s | %s", project_id, str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# ─── DATASET VERSIONING ENDPOINTS ──────────────────────────────────────────────

@api_bp.route('/projects/<project_id>/datasets', methods=['POST'])
def upload_dataset_version(project_id):
    import os, shutil, uuid
    from werkzeug.utils import secure_filename
    from flask import current_app
    from app.core.evaluator import inspect_classification_dataset
    
    try:
        project = dbh.get_project(project_id)
        if not project:
            return jsonify({"status": "error", "message": "Project not found"}), 404
            
        if 'dataset' not in request.files:
            return jsonify({"status": "error", "message": "No dataset file uploaded"}), 400
            
        dataset_file = request.files['dataset']
        if dataset_file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400
            
        if not dataset_file.filename.endswith('.zip'):
            return jsonify({"status": "error", "message": "Dataset must be a ZIP file"}), 400

        dataset_name = request.form.get("name")
        if not dataset_name:
            dataset_name = secure_filename(dataset_file.filename).replace(".zip", "")

        logger.info("[DATASET] Version upload started | project_id=%s | name=%s", project_id, dataset_name)
            
        uid = str(uuid.uuid4())
        dataset_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'datasets', uid)
        os.makedirs(dataset_dir, exist_ok=True)
        
        zip_path = os.path.join(dataset_dir, 'dataset.zip')
        extract_to = os.path.join(dataset_dir, 'dataset_extracted')
        
        # Save zip
        dataset_file.save(zip_path)
        
        # Validate and inspect
        try:
            metadata = inspect_classification_dataset(zip_path, extract_to)
        except Exception as e:
            logger.error("[DATASET] Validation failed | project_id=%s | %s", project_id, str(e))
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
            return jsonify({"status": "error", "message": f"Invalid classification dataset: {str(e)}"}), 400
            
        dataset_id = dbh.create_dataset(
            project_id=project_id,
            dataset_name=dataset_name,
            zip_path=zip_path,
            extract_path=extract_to,
            classes=metadata["classes"],
            samples=metadata["num_samples"]
        )
        
        # Update extra metadata distribution info
        dbh.get_db().datasets.update_one(
            {"_id": dbh.ObjectId(dataset_id)},
            {"$set": {"distribution": metadata["distribution"]}}
        )
        
        logger.info("[DATASET] Upload complete | dataset_id=%s | classes=%d | samples=%d",
                    dataset_id, metadata["num_classes"], metadata["num_samples"])
        return jsonify({
            "status": "success",
            "message": "Dataset version uploaded successfully",
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "classes": metadata["classes"],
            "samples": metadata["num_samples"]
        })
    except Exception as e:
        logger.error("[DATASET] Version upload exception | project_id=%s | %s", project_id, str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/projects/<project_id>/datasets', methods=['GET'])
def list_project_datasets(project_id):
    try:
        datasets = dbh.get_datasets(project_id)
        return jsonify({"status": "success", "datasets": datasets})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/datasets/<dataset_id>', methods=['GET'])
def get_dataset_version(dataset_id):
    try:
        dataset = dbh.get_dataset(dataset_id)
        # print(dataset)
        if dataset is not  None:
            return jsonify({"status": "success", "dataset": dataset, "message": "Dataset version fetch successfully"})
        return jsonify({"status": "error", "message": "Failed to fetch dataset version"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/datasets/<dataset_id>', methods=['DELETE'])
def delete_dataset_version(dataset_id):
    try:
        success = dbh.delete_dataset(dataset_id)
        if success:
            return jsonify({"status": "success", "message": "Dataset version deleted successfully"})
        return jsonify({"status": "error", "message": "Failed to delete dataset version"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ─── TRAINING RUNS ENDPOINTS ───────────────────────────────────────────────────

@api_bp.route('/versions/<version_id>/runs', methods=['GET'])
def list_version_runs(version_id):
    try:
        runs = dbh.get_runs(version_id)
        return jsonify({"status": "success", "runs": runs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/projects/<project_id>/runs', methods=['GET'])
def list_project_runs(project_id):
    try:
        runs = dbh.get_project_runs(project_id)
        return jsonify({"status": "success", "runs": runs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/runs/<run_id>', methods=['GET'])
def get_run_details(run_id):
    try:
        run = dbh.get_run(run_id)
        if not run:
            return jsonify({"status": "error", "message": "Run not found"}), 404
        return jsonify({"status": "success", "run": run})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/runs/<run_id>/logs', methods=['GET'])
def get_run_logs(run_id):
    try:
        run = dbh.get_run(run_id)
        if not run:
            return jsonify({"status": "error", "message": "Run not found"}), 404
        return jsonify({"status": "success", "logs": run.get("train_logs", [])})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/runs/<run_id>', methods=['DELETE'])
def delete_training_run(run_id):
    try:
        success = dbh.delete_run(run_id)
        if success:
            return jsonify({"status": "success", "message": "Training run deleted successfully"})
        return jsonify({"status": "error", "message": "Failed to delete training run"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ─── RUN EXPORTS AND DOWNLOADS ─────────────────────────────────────────────────

@api_bp.route('/runs/<run_id>/export/onnx', methods=['POST'])
def export_run_onnx(run_id):
    import os, torch
    from app.core.builder import build_model
    try:
        run = dbh.get_run(run_id)
        if not run:
            return jsonify({"status": "error", "message": "Run not found"}), 404
            
        if not run.get("model_path") or not os.path.exists(run["model_path"]):
            return jsonify({"status": "error", "message": "Run is not successfully trained yet or model weights file is missing."}), 400
            
        version = dbh.get_version(run["version_id"])
        if not version:
            return jsonify({"status": "error", "message": "Model version not found"}), 404
            
        model, input_shape, _ = build_model(version["graph"])
        
        # Load weights on CPU for export
        device = torch.device('cpu')
        model.load_state_dict(torch.load(run["model_path"], map_location=device))
        model.eval()
        
        run_dir = os.path.dirname(run["model_path"])
        onnx_path = os.path.join(run_dir, "model_best.onnx")
        
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        dbh.update_run_status(run_id, status=run["status"], onnx_path=onnx_path)
        
        return jsonify({
            "status": "success",
            "message": "Model successfully exported to ONNX format",
            "onnx_path": f"/api/runs/{run_id}/download/onnx"
        })
    except Exception as e:
        logger.error("[RUN/ONNX] Export exception | run_id=%s | %s", run_id, str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/runs/<run_id>/download/onnx', methods=['GET'])
def download_run_onnx(run_id):
    import os
    from flask import send_file
    try:
        run = dbh.get_run(run_id)
        if not run or not run.get("onnx_path") or not os.path.exists(run["onnx_path"]):
            return "ONNX file not found or not exported yet.", 404
        return send_file(run["onnx_path"], as_attachment=True, download_name=f"{run['run_name']}_model.onnx")
    except Exception as e:
        return str(e), 500

@api_bp.route('/runs/<run_id>/download/pt', methods=['GET'])
def download_run_pt(run_id):
    import os
    from flask import send_file
    try:
        run = dbh.get_run(run_id)
        if not run or not run.get("model_path") or not os.path.exists(run["model_path"]):
            return "PyTorch checkpoint file not found.", 404
        return send_file(run["model_path"], as_attachment=True, download_name=f"{run['run_name']}_model.pt")
    except Exception as e:
        return str(e), 500

# ─── RELOCATED EVALUATE ENDPOINT ───────────────────────────────────────────────

@api_bp.route('/runs/<run_id>/evaluate', methods=['POST'])
def evaluate_run_route(run_id):
    import os, shutil, uuid, json, torch
    from flask import current_app, secure_filename
    from app.core.builder import build_model
    from app.core.evaluator import extract_dataset, evaluate_model
    
    try:
        run = dbh.get_run(run_id)
        if not run:
            return jsonify({"status": "error", "message": "Run not found"}), 404
            
        if not run.get("model_path") or not os.path.exists(run["model_path"]):
            return jsonify({"status": "error", "message": "This run has no successfully trained model weights yet."}), 400
            
        version = dbh.get_version(run["version_id"])
        if not version:
            return jsonify({"status": "error", "message": "Version not found"}), 404
            
        upload_path = None
        extract_path = None
        unique_id = str(uuid.uuid4())
        
        if 'dataset' in request.files and request.files['dataset'].filename != '':
            dataset_file = request.files['dataset']
            if not dataset_file.filename.endswith('.zip'):
                return jsonify({"status": "error", "message": "Dataset must be a .zip file"}), 400
            zip_filename = secure_filename(dataset_file.filename)
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"eval_{unique_id}_{zip_filename}")
            extract_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"eval_ext_{unique_id}")
            dataset_file.save(upload_path)
        else:
            # Fall back to dataset version linked with the run
            if run.get("dataset_id"):
                ds_ver = dbh.get_dataset(run["dataset_id"])
                if ds_ver and os.path.exists(ds_ver["zip_path"]):
                    upload_path = ds_ver["zip_path"]
                    extract_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"eval_ext_{unique_id}")
            # If still no dataset, check project-level
            if not upload_path:
                project = dbh.get_project(run["project_id"])
                if project and project.get("dataset") and os.path.exists(project["dataset"]["zip_path"]):
                    upload_path = project["dataset"]["zip_path"]
                    extract_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"eval_ext_{unique_id}")
                    
        if not upload_path:
            return jsonify({"status": "error", "message": "No dataset was provided and no dataset version is associated with this training run."}), 400
            
        # Rebuild Model Structure
        model, input_shape, _ = build_model(version["graph"])
        
        # Load Trained Weights
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model.load_state_dict(torch.load(run["model_path"], map_location=device))
        model = model.to(device)
        model.eval()
        
        # Extract and Evaluate
        dataset_root = extract_dataset(upload_path, extract_path)
        results = evaluate_model(model, input_shape, dataset_root)
        
        # Cleanup if we created temp directories
        try:
            if 'dataset' in request.files and os.path.exists(upload_path):
                os.remove(upload_path)
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)
        except Exception:
            pass
            
        return jsonify({
            "status": "success",
            "message": "Evaluation Complete",
            "results": results
        })
    except Exception as e:
        logger.error("[RUN/EVALUATE] Evaluation exception | run_id=%s | %s", run_id, str(e), exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
