from bson import ObjectId
from datetime import datetime

def get_db():
    from app import db
    return db

def create_project(name, description):
    db = get_db()
    if db is None:
        raise RuntimeError("Database connection not available")
    project = {
        "name": name,
        "description": description,
        "created_at": datetime.utcnow()
    }
    result = db.projects.insert_one(project)
    return str(result.inserted_id)

def get_projects():
    db = get_db()
    if db is None:
        return []
    projects = list(db.projects.find().sort("created_at", -1))
    for p in projects:
        p["_id"] = str(p["_id"])
    return projects

def get_project(project_id):
    db = get_db()
    if db is None:
        return None
    try:
        p = db.projects.find_one({"_id": ObjectId(project_id)})
        if p:
            p["_id"] = str(p["_id"])
        return p
    except Exception:
        return None

def delete_project(project_id):
    db = get_db()
    if db is None:
        return False
    # Also delete associated versions and their weights files if any
    versions = get_versions(project_id)
    for v in versions:
        delete_version(v["_id"])
    db.projects.delete_one({"_id": ObjectId(project_id)})
    return True

def create_version(project_id, version_name, graph=None):
    db = get_db()
    if db is None:
        raise RuntimeError("Database connection not available")
    
    # Check if version name already exists for this project
    existing = db.versions.find_one({"project_id": ObjectId(project_id), "version_name": version_name})
    if existing:
        raise ValueError(f"Version '{version_name}' already exists in this project.")

    if graph is None:
        # Default graph: just an input node
        graph = {
            "last_node_id": 1,
            "last_link_id": 0,
            "nodes": [
                {
                    "id": 1,
                    "type": "pytorch/input",
                    "pos": [100, 200],
                    "size": [210, 82],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "properties": {"shape": "[1, 3, 224, 224]"},
                    "outputs": [{"name": "out", "type": "tensor", "links": []}]
                }
            ],
            "links": [],
            "groups": [],
            "config": {},
            "version": 0.4
        }

    version = {
        "project_id": ObjectId(project_id),
        "version_name": version_name,
        "graph": graph,
        "status": "untrained",
        "metrics": {},
        "model_path": None,
        "classes": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    result = db.versions.insert_one(version)
    return str(result.inserted_id)

def get_versions(project_id):
    db = get_db()
    if db is None:
        return []
    try:
        versions = list(db.versions.find({"project_id": ObjectId(project_id)}).sort("created_at", 1))
        for v in versions:
            v["_id"] = str(v["_id"])
            v["project_id"] = str(v["project_id"])
        return versions
    except Exception:
        return []

def get_version(version_id):
    db = get_db()
    if db is None:
        return None
    try:
        v = db.versions.find_one({"_id": ObjectId(version_id)})
        if v:
            v["_id"] = str(v["_id"])
            v["project_id"] = str(v["project_id"])
        return v
    except Exception:
        return None

def update_version_graph(version_id, graph):
    db = get_db()
    if db is None:
        return False
    db.versions.update_one(
        {"_id": ObjectId(version_id)},
        {"$set": {"graph": graph, "updated_at": datetime.utcnow()}}
    )
    return True

def update_version_status(version_id, status, metrics=None, model_path=None, classes=None):
    db = get_db()
    if db is None:
        return False
    update_data = {
        "status": status,
        "updated_at": datetime.utcnow()
    }
    if metrics is not None:
        update_data["metrics"] = metrics
    if model_path is not None:
        update_data["model_path"] = model_path
    if classes is not None:
        update_data["classes"] = classes
        
    db.versions.update_one(
        {"_id": ObjectId(version_id)},
        {"$set": update_data}
    )
    return True

def delete_version(version_id):
    db = get_db()
    if db is None:
        return False
    
    # Delete associated training runs cascadingly
    runs = get_runs(version_id)
    for r in runs:
        delete_run(r["_id"])
        
    v = get_version(version_id)
    if v and v.get("model_path"):
        import os
        try:
            if os.path.exists(v["model_path"]):
                os.remove(v["model_path"])
        except Exception as e:
            print(f"Error removing model file {v['model_path']}: {e}")
    db.versions.delete_one({"_id": ObjectId(version_id)})
    return True

def update_project(project_id, name, description):
    db = get_db()
    if db is None:
        return False
    db.projects.update_one(
        {"_id": ObjectId(project_id)},
        {"$set": {"name": name, "description": description}}
    )
    return True

def update_project_dataset(project_id, dataset_info):
    db = get_db()
    if db is None:
        return False
    db.projects.update_one(
        {"_id": ObjectId(project_id)},
        {"$set": {"dataset": dataset_info}}
    )
    return True

def delete_project_dataset(project_id):
    db = get_db()
    if db is None:
        return False
    db.projects.update_one(
        {"_id": ObjectId(project_id)},
        {"$unset": {"dataset": ""}}
    )
    return True

# ─── DATASET VERSIONING HELPERS ────────────────────────────────────────────────

def create_dataset(project_id, dataset_name, zip_path, extract_path, classes, samples):
    db = get_db()
    if db is None:
        raise RuntimeError("Database connection not available")
    dataset = {
        "project_id": ObjectId(project_id),
        "dataset_name": dataset_name,
        "zip_path": zip_path,
        "extract_path": extract_path,
        "classes": classes,
        "samples": samples,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    result = db.datasets.insert_one(dataset)
    return str(result.inserted_id)

def get_datasets(project_id):
    db = get_db()
    if db is None:
        return []
    try:
        datasets = list(db.datasets.find({"project_id": ObjectId(project_id)}).sort("created_at", -1))
        for d in datasets:
            d["_id"] = str(d["_id"])
            d["project_id"] = str(d["project_id"])
        return datasets
    except Exception:
        return []

def get_dataset(dataset_id):
    db = get_db()
    if db is None:
        return None
    try:
        d = db.datasets.find_one({"_id": ObjectId(dataset_id)})
        if d:
            d["_id"] = str(d["_id"])
            d["project_id"] = str(d["project_id"])
        return d
    except Exception:
        return None

def delete_dataset(dataset_id):
    db = get_db()
    if db is None:
        return False
    d = get_dataset(dataset_id)
    if d:
        import os, shutil
        # delete zip file
        try:
            if d.get("zip_path") and os.path.exists(d["zip_path"]):
                os.remove(d["zip_path"])
        except Exception as e:
            print(f"Error removing dataset zip file: {e}")
        # delete extracted folder
        try:
            if d.get("extract_path") and os.path.exists(d["extract_path"]):
                shutil.rmtree(d["extract_path"])
        except Exception as e:
            print(f"Error removing dataset extracted folder: {e}")
        db.datasets.delete_one({"_id": ObjectId(dataset_id)})
        return True
    return False

# ─── TRAINING RUNS HELPERS ─────────────────────────────────────────────────────

def create_run(version_id, project_id, run_name, dataset_id, hyperparameters, pretrained_run_id=None):
    db = get_db()
    if db is None:
        raise RuntimeError("Database connection not available")
    run = {
        "version_id": ObjectId(version_id),
        "project_id": ObjectId(project_id),
        "run_name": run_name,
        "dataset_id": ObjectId(dataset_id) if dataset_id else None,
        "pretrained_run_id": ObjectId(pretrained_run_id) if pretrained_run_id else None,
        "status": "untrained",
        "hyperparameters": hyperparameters,
        "model_path": None,
        "onnx_path": None,
        "classes": [],
        "metrics": {},
        "train_logs": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    result = db.train_runs.insert_one(run)
    return str(result.inserted_id)

def get_runs(version_id):
    db = get_db()
    if db is None:
        return []
    try:
        runs = list(db.train_runs.find({"version_id": ObjectId(version_id)}).sort("created_at", -1))
        for r in runs:
            r["_id"] = str(r["_id"])
            r["version_id"] = str(r["version_id"])
            r["project_id"] = str(r["project_id"])
            if r.get("dataset_id"):
                r["dataset_id"] = str(r["dataset_id"])
            if r.get("pretrained_run_id"):
                r["pretrained_run_id"] = str(r["pretrained_run_id"])
        return runs
    except Exception:
        return []

def get_project_runs(project_id):
    db = get_db()
    if db is None:
        return []
    try:
        runs = list(db.train_runs.find({"project_id": ObjectId(project_id)}).sort("created_at", -1))
        for r in runs:
            r["_id"] = str(r["_id"])
            r["version_id"] = str(r["version_id"])
            r["project_id"] = str(r["project_id"])
            if r.get("dataset_id"):
                r["dataset_id"] = str(r["dataset_id"])
            if r.get("pretrained_run_id"):
                r["pretrained_run_id"] = str(r["pretrained_run_id"])
        return runs
    except Exception:
        return []

def get_run(run_id):
    db = get_db()
    if db is None:
        return None
    try:
        r = db.train_runs.find_one({"_id": ObjectId(run_id)})
        if r:
            r["_id"] = str(r["_id"])
            r["version_id"] = str(r["version_id"])
            r["project_id"] = str(r["project_id"])
            if r.get("dataset_id"):
                r["dataset_id"] = str(r["dataset_id"])
            if r.get("pretrained_run_id"):
                r["pretrained_run_id"] = str(r["pretrained_run_id"])
        return r
    except Exception:
        return None

def update_run_status(run_id, status, metrics=None, model_path=None, onnx_path=None, classes=None, train_logs=None):
    db = get_db()
    if db is None:
        return False
    update_data = {
        "status": status,
        "updated_at": datetime.utcnow()
    }
    if metrics is not None:
        update_data["metrics"] = metrics
    if model_path is not None:
        update_data["model_path"] = model_path
    if onnx_path is not None:
        update_data["onnx_path"] = onnx_path
    if classes is not None:
        update_data["classes"] = classes
    if train_logs is not None:
        update_data["train_logs"] = train_logs

    db.train_runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": update_data}
    )
    return True

def delete_run(run_id):
    db = get_db()
    if db is None:
        return False
    r = get_run(run_id)
    if r:
        import os
        # Delete weights
        if r.get("model_path"):
            try:
                if os.path.exists(r["model_path"]):
                    os.remove(r["model_path"])
            except Exception as e:
                print(f"Error removing run weights file: {e}")
        # Delete ONNX
        if r.get("onnx_path"):
            try:
                if os.path.exists(r["onnx_path"]):
                    os.remove(r["onnx_path"])
            except Exception as e:
                print(f"Error removing run ONNX file: {e}")
        db.train_runs.delete_one({"_id": ObjectId(run_id)})
        return True
    return False
