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
