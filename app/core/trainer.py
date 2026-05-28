"""
HEXForge Training Engine
Runs PyTorch training in a background thread, streams Ultralytics-style logs
and per-epoch metrics (loss, accuracy, precision, recall, F1) to a shared state.
"""

import os
import time
import threading
import zipfile
import shutil
import uuid
import traceback
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ─── Shared training state ──────────────────────────────────────────────────

_state_lock = threading.Lock()
_training_state = {
    "status": "idle",          # idle | running | stopped | done | error
    "epoch": 0,
    "total_epochs": 0,
    "batch": 0,
    "total_batches": 0,
    "logs": [],                # list of log-line strings
    "history": {               # per-epoch metrics lists
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
        "precision":  [],
        "recall":     [],
        "f1":         [],
        "lr":         [],
        "cpu_usage":  [],
        "gpu_usage":  [],
        "ram_usage":  [],
    },
    "best_val_loss": float("inf"),
    "device": "cpu",
    "classes": [],
    "error": None,
    "model_path": None,        # path to saved best model
    "elapsed": 0.0,
    "eta": 0.0,
    "progress_bar": None,
    "val_samples": [],
}

_stop_event = threading.Event()


def get_state():
    with _state_lock:
        import copy
        return copy.deepcopy(_training_state)


def stop_training():
    _stop_event.set()


def _log(msg: str, tag: str = "INFO"):
    """Append a log line to state. tag: INFO | EPOCH | BATCH | WARN | ERROR | SUCCESS"""
    with _state_lock:
        _training_state["logs"].append({"tag": tag, "msg": msg, "ts": time.time()})
    # Keep only the last 2000 lines
    with _state_lock:
        if len(_training_state["logs"]) > 2000:
            _training_state["logs"] = _training_state["logs"][-2000:]


def _reset_state(total_epochs: int, device_name: str, classes: list):
    with _state_lock:
        _training_state["status"]       = "running"
        _training_state["epoch"]        = 0
        _training_state["total_epochs"] = total_epochs
        _training_state["batch"]        = 0
        _training_state["total_batches"]= 0
        _training_state["logs"]         = []
        _training_state["history"]      = {k: [] for k in _training_state["history"]}
        _training_state["best_val_loss"]= float("inf")
        _training_state["device"]       = device_name
        _training_state["classes"]      = classes
        _training_state["error"]        = None
        _training_state["model_path"]   = None
        _training_state["elapsed"]      = 0.0
        _training_state["eta"]          = 0.0
        _training_state["progress_bar"] = None
        _training_state["val_samples"]  = []


# ─── Metrics helpers ─────────────────────────────────────────────────────────

def _compute_clf_metrics(all_preds, all_labels, num_classes):
    """Compute precision, recall, F1 (macro average) from lists of ints."""
    if not all_preds:
        return 0.0, 0.0, 0.0

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            tp[label] += 1
        else:
            fp[pred]   += 1
            fn[label]  += 1

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        p = tp[c] / (tp[c] + fp[c] + 1e-9)
        r = tp[c] / (tp[c] + fn[c] + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    macro_p  = sum(precisions) / len(precisions)
    macro_r  = sum(recalls)    / len(recalls)
    macro_f1 = sum(f1s)        / len(f1s)
    return round(macro_p, 4), round(macro_r, 4), round(macro_f1, 4)


# ─── Dataset helpers ─────────────────────────────────────────────────────────

def _extract_zip(zip_path: str, extract_to: str) -> str:
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    items = os.listdir(extract_to)
    if len(items) == 1 and os.path.isdir(os.path.join(extract_to, items[0])):
        return os.path.join(extract_to, items[0])
    return extract_to


def _build_transforms(input_shape: list, augment: bool = False):
    h, w = input_shape[2], input_shape[3]
    ch   = input_shape[1]

    if augment:
        tf_list = [
            transforms.Resize((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    else:
        tf_list = [transforms.Resize((h, w)), transforms.ToTensor()]

    if ch == 1:
        tf_list.insert(1, transforms.Grayscale(num_output_channels=1))

    return transforms.Compose(tf_list)


# ─── Optimizers / schedulers / losses ────────────────────────────────────────

def _build_optimizer(model, cfg: dict):
    name = cfg.get("optimizer", "adam").lower()
    lr   = float(cfg.get("lr", 1e-3))
    wd   = float(cfg.get("weight_decay", 0.0))
    mom  = float(cfg.get("momentum", 0.9))

    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    else:  # adam (default)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    name    = cfg.get("scheduler", "none").lower()
    epochs  = int(cfg.get("epochs", 10))

    if name == "steplr":
        step = max(1, epochs // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=False)
    return None


def _build_criterion(cfg: dict):
    name = cfg.get("loss", "crossentropy").lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "mse":
        return nn.MSELoss()
    elif name == "l1":
        return nn.L1Loss()
    elif name == "huber":
        return nn.HuberLoss()
    else:
        return nn.CrossEntropyLoss()


# ─── Ultralytics-style header ─────────────────────────────────────────────────

_HDR  = f"{'Epoch':>8}  {'GPU':>6}  {'T-Loss':>8}  {'V-Loss':>8}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'LR':>9}"
_DIV  = "─" * len(_HDR)


# ─── Core training thread ─────────────────────────────────────────────────────

def _train_thread(graph_data, zip_path, extract_path, cfg, model_save_path):
    """Runs inside a daemon thread."""
    try:
        from app.core.builder import build_model

        # ── Device ──────────────────────────────────────────────────────────
        use_cuda   = torch.cuda.is_available()
        device     = torch.device("cuda" if use_cuda else "cpu")
        dev_name   = torch.cuda.get_device_name(0) if use_cuda else "CPU"

        # ── Build model ──────────────────────────────────────────────────────
        _log("Building model from graph…", "INFO")
        model, input_shape, _ = build_model(graph_data)
        model = model.to(device)

        # ── Dataset ──────────────────────────────────────────────────────────
        _log("Extracting dataset…", "INFO")
        dataset_root = _extract_zip(zip_path, extract_path)

        train_tf = _build_transforms(input_shape, augment=cfg.get("augment", False))
        val_tf   = _build_transforms(input_shape, augment=False)

        try:
            full_dataset = datasets.ImageFolder(dataset_root, train_tf)
        except Exception as e:
            raise RuntimeError(f"Could not load ImageFolder dataset: {e}")

        if len(full_dataset) == 0:
            raise RuntimeError("Dataset is empty.")

        num_classes = len(full_dataset.classes)
        classes     = full_dataset.classes

        # Update DB if version_id is provided
        version_id = cfg.get("version_id")
        if version_id:
            from app.core.db_helpers import update_version_status
            update_version_status(version_id, status="training", classes=classes)

        # ── Reset state with class info ──────────────────────────────────────
        epochs = int(cfg.get("epochs", 10))
        _reset_state(epochs, dev_name, classes)

        # Post info logs AFTER reset so they aren't wiped
        _log(f"Device      : {dev_name}", "INFO")
        _log(f"Classes     : {num_classes}  →  {', '.join(classes)}", "INFO")
        _log(f"Images      : {len(full_dataset)}", "INFO")
        _log(f"Input shape : {input_shape}", "INFO")
        _log(f"Loss        : {cfg.get('loss','crossentropy').upper()}", "INFO")
        _log(f"Optimizer   : {cfg.get('optimizer','adam').upper()}  lr={cfg.get('lr',1e-3)}", "INFO")
        _log(f"Scheduler   : {cfg.get('scheduler','none').upper()}", "INFO")
        _log(f"Epochs      : {epochs}", "INFO")
        _log(f"Batch size  : {cfg.get('batch_size',32)}", "INFO")
        _log("", "INFO")

        # ── Train / Val split ────────────────────────────────────────────────
        val_frac  = float(cfg.get("val_split", 0.2))
        val_size  = max(1, int(len(full_dataset) * val_frac))
        train_size= len(full_dataset) - val_size

        train_ds, val_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Re-apply val transforms to val subset via a wrapper
        class _ValSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset    = subset
                self.transform = transform
            def __len__(self):
                return len(self.subset)
            def __getitem__(self, idx):
                img, label = self.subset[idx]
                # img is already a tensor from train_tf; re-load via dataset
                # Access underlying dataset
                actual_idx = self.subset.indices[idx]
                path, label = self.subset.dataset.samples[actual_idx]
                from PIL import Image
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label

        val_dataset = _ValSubset(val_ds, val_tf)

        bs         = int(cfg.get("batch_size", 32))
        num_workers= 0  # 0 = main thread (avoids pickle issues in threads)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=num_workers)
        val_loader   = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

        _log(f"Train samples : {train_size}   Val samples : {val_size}", "INFO")

        # ── Copy 3 validation samples for live visualization ──────────────────
        import random
        val_samples_info = []
        project_id = None
        if version_id:
            try:
                from app.core.db_helpers import get_version
                ver = get_version(version_id)
                if ver:
                    project_id = ver.get("project_id")
            except Exception:
                pass

        # Static uploads folder path
        static_uploads = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "uploads"))
        if project_id:
            samples_dir = os.path.join(static_uploads, "projects", str(project_id), "samples")
        else:
            samples_dir = os.path.join(static_uploads, "temp_samples")

        try:
            if os.path.exists(samples_dir):
                shutil.rmtree(samples_dir)
            os.makedirs(samples_dir, exist_ok=True)

            val_indices = list(val_ds.indices)
            if len(val_indices) > 3:
                rng = random.Random(42)
                sample_indices = rng.sample(val_indices, 3)
            else:
                sample_indices = val_indices

            for i, idx in enumerate(sample_indices):
                path, class_idx = full_dataset.samples[idx]
                ext = os.path.splitext(path)[1] or ".png"
                dest_name = f"sample_{i}{ext}"
                dest_path = os.path.join(samples_dir, dest_name)
                shutil.copy2(path, dest_path)

                val_samples_info.append({
                    "image_url": f"/static/uploads/projects/{project_id}/samples/{dest_name}" if project_id else f"/static/uploads/temp_samples/{dest_name}",
                    "label": classes[class_idx],
                    "path": path,
                    "prediction": "—",
                    "confidence": 0.0
                })
            with _state_lock:
                _training_state["val_samples"] = val_samples_info
        except Exception as se:
            _log(f"Warning: Could not extract validation samples for visualization: {se}", "WARN")

        _log(_DIV, "INFO")
        _log(_HDR, "INFO")
        _log(_DIV, "INFO")

        # ── Optimizer / scheduler / loss ─────────────────────────────────────
        optimizer = _build_optimizer(model, cfg)
        scheduler = _build_scheduler(optimizer, cfg, len(train_loader))
        criterion = _build_criterion(cfg)

        is_clf    = cfg.get("loss", "crossentropy").lower() in ("crossentropy", "bce")

        start_time = time.time()

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            if _stop_event.is_set():
                _log("Training interrupted by user.", "WARN")
                break

            ep_start = time.time()

            # ── Train phase ──────────────────────────────────────────────────
            model.train()
            running_loss  = 0.0
            correct_train = 0
            total_train   = 0
            batches       = len(train_loader)

            with _state_lock:
                _training_state["epoch"]        = epoch
                _training_state["total_batches"] = batches

            for b_idx, (inputs, labels) in enumerate(train_loader, 1):
                if _stop_event.is_set():
                    break

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                if cfg.get("loss","crossentropy").lower() == "bce":
                    loss = criterion(outputs.squeeze(), labels.float())
                else:
                    loss = criterion(outputs, labels)

                loss.backward()

                # Gradient clipping
                clip = float(cfg.get("grad_clip", 0.0))
                if clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)

                optimizer.step()

                running_loss  += loss.item()
                if is_clf:
                    _, preds = torch.max(outputs, 1)
                    correct_train += (preds == labels).sum().item()
                    total_train   += labels.size(0)

                with _state_lock:
                    _training_state["batch"] = b_idx

                # Update live tqdm-like progress bar
                try:
                    elapsed_b = time.time() - ep_start
                    rate = b_idx / elapsed_b if elapsed_b > 0 else 0
                    eta_b = (batches - b_idx) / rate if rate > 0 else 0
                    pct = int(b_idx / batches * 100)
                    
                    bar_len = 15
                    filled = int(b_idx / batches * bar_len)
                    bar_str = "█" * filled + "░" * (bar_len - filled)
                    
                    elapsed_str = f"{int(elapsed_b)//60:02d}:{int(elapsed_b)%60:02d}"
                    eta_str = f"{int(eta_b)//60:02d}:{int(eta_b)%60:02d}"
                    
                    curr_loss = running_loss / b_idx
                    prog_msg = f"Epoch {epoch}/{epochs}: {pct:3d}%|{bar_str}| {b_idx}/{batches} [{elapsed_str}<{eta_str}, {rate:.2f}it/s, loss={curr_loss:.4f}]"
                    with _state_lock:
                        _training_state["progress_bar"] = prog_msg
                except Exception:
                    pass

            # ── Val phase ────────────────────────────────────────────────────
            model.eval()
            val_loss_sum  = 0.0
            all_preds     = []
            all_labels_v  = []
            correct_val   = 0
            total_val     = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    if cfg.get("loss","crossentropy").lower() == "bce":
                        vloss = criterion(outputs.squeeze(), labels.float())
                    else:
                        vloss = criterion(outputs, labels)

                    val_loss_sum += vloss.item()

                    if is_clf:
                        _, preds = torch.max(outputs, 1)
                        correct_val   += (preds == labels).sum().item()
                        total_val     += labels.size(0)
                        all_preds.extend(preds.cpu().tolist())
                        all_labels_v.extend(labels.cpu().tolist())

            # ── Compute epoch metrics ────────────────────────────────────────
            train_loss = running_loss  / max(1, len(train_loader))
            val_loss   = val_loss_sum  / max(1, len(val_loader))
            train_acc  = (correct_train / max(1, total_train)) if is_clf else 0.0
            val_acc    = (correct_val   / max(1, total_val))   if is_clf else 0.0

            prec, rec, f1 = 0.0, 0.0, 0.0
            if is_clf and all_preds:
                prec, rec, f1 = _compute_clf_metrics(all_preds, all_labels_v, num_classes)

            # Run inference on the 3 visualization samples
            if val_samples_info:
                model.eval()
                with torch.no_grad():
                    for sample in val_samples_info:
                        try:
                            from PIL import Image
                            img = Image.open(sample["path"]).convert("RGB")
                            x = val_tf(img).unsqueeze(0).to(device)
                            outputs = model(x)
                            if is_clf:
                                if outputs.ndim == 2:
                                    num_classes_out = outputs.shape[1]
                                    if num_classes_out == 1:
                                        prob = torch.sigmoid(outputs)[0][0].item()
                                        class_idx = 1 if prob >= 0.5 else 0
                                        conf = prob if class_idx == 1 else 1.0 - prob
                                        pred_lbl = classes[class_idx] if class_idx < len(classes) else str(class_idx)
                                    else:
                                        probabilities = torch.softmax(outputs, dim=1)[0]
                                        conf_tensor, class_idx_tensor = torch.max(probabilities, 0)
                                        conf = conf_tensor.item()
                                        class_idx = class_idx_tensor.item()
                                        pred_lbl = classes[class_idx] if class_idx < len(classes) else str(class_idx)
                                else:
                                    pred_lbl = "Unknown"
                                    conf = 0.0
                            else:
                                pred_lbl = "Regression"
                                conf = 1.0

                            sample["prediction"] = pred_lbl
                            sample["confidence"] = round(conf * 100, 2)
                        except Exception as e_inf:
                            _log(f"Warning: Sample inference failed for {sample['image_url']}: {e_inf}", "WARN")

                with _state_lock:
                    import copy
                    _training_state["val_samples"] = copy.deepcopy(val_samples_info)

            current_lr = optimizer.param_groups[0]["lr"]

            # ── Scheduler step ───────────────────────────────────────────────
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # ── GPU memory ───────────────────────────────────────────────────
            if use_cuda:
                mem_gb = torch.cuda.memory_allocated(0) / 1e9
                gpu_str = f"{mem_gb:.2f}G"
            else:
                gpu_str = "0G"

            ep_time  = time.time() - ep_start
            elapsed  = time.time() - start_time
            eta      = (elapsed / epoch) * (epochs - epoch)

            # ── Best model checkpoint ────────────────────────────────────────
            saved = ""
            if val_loss < _training_state["best_val_loss"]:
                with _state_lock:
                    _training_state["best_val_loss"] = val_loss
                torch.save(model.state_dict(), model_save_path)
                saved = " ✔ saved"
                with _state_lock:
                    _training_state["model_path"] = model_save_path

            # ── System resource usage tracking ───────────────────────────────
            cpu_usage_pct = 0.0
            ram_usage_pct = 0.0
            gpu_usage_pct = 0.0
            try:
                import psutil
                cpu_usage_pct = psutil.cpu_percent()
                ram_usage_pct = psutil.virtual_memory().percent
            except Exception:
                pass

            try:
                if torch.cuda.is_available():
                    dev_idx = torch.cuda.current_device()
                    tot_mem = torch.cuda.get_device_properties(dev_idx).total_memory
                    alloc_mem = torch.cuda.memory_allocated(dev_idx)
                    if tot_mem > 0:
                        gpu_usage_pct = round((alloc_mem / tot_mem) * 100, 2)
            except Exception:
                pass

            # ── Update history ───────────────────────────────────────────────
            with _state_lock:
                h = _training_state["history"]
                h["train_loss"].append(round(train_loss, 5))
                h["val_loss"].append(round(val_loss, 5))
                h["train_acc"].append(round(train_acc, 4))
                h["val_acc"].append(round(val_acc, 4))
                h["precision"].append(prec)
                h["recall"].append(rec)
                h["f1"].append(f1)
                h["lr"].append(round(current_lr, 8))
                h["cpu_usage"].append(cpu_usage_pct)
                h["gpu_usage"].append(gpu_usage_pct)
                h["ram_usage"].append(ram_usage_pct)
                _training_state["elapsed"] = round(elapsed, 1)
                _training_state["eta"]     = round(eta, 1)

            # ── Ultralytics-style log line ────────────────────────────────────
            log_line = (
                f"{epoch:>4}/{epochs:<4}  "
                f"{gpu_str:>6}  "
                f"{train_loss:>8.4f}  "
                f"{val_loss:>8.4f}  "
                f"{val_acc*100:>6.2f}%  "
                f"{prec:>7.4f}  "
                f"{rec:>7.4f}  "
                f"{f1:>7.4f}  "
                f"{current_lr:>9.6f}"
                f"{saved}"
            )
            _log(log_line, "EPOCH")
            _log(f"  └─ epoch time: {ep_time:.1f}s  elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s", "INFO")

        # ── Training complete ─────────────────────────────────────────────────
        _log(_DIV, "INFO")
        total_elapsed = time.time() - start_time
        _log(f"Training complete in {total_elapsed:.1f}s  |  Best val loss: {_training_state['best_val_loss']:.5f}", "SUCCESS")
        if _training_state.get("model_path"):
            _log(f"Best model saved → {_training_state['model_path']}", "SUCCESS")

        with _state_lock:
            if _stop_event.is_set():
                _training_state["status"] = "stopped"
            else:
                _training_state["status"] = "done"

        # Update DB if version_id is provided
        version_id = cfg.get("version_id")
        if version_id:
            from app.core.db_helpers import update_version_status
            db_status = "stopped" if _stop_event.is_set() else "trained"
            # Get latest accuracy
            accuracy = 0.0
            if _training_state["history"]["val_acc"]:
                accuracy = _training_state["history"]["val_acc"][-1]
            metrics = {
                "val_loss": _training_state["best_val_loss"],
                "accuracy": round(accuracy * 100, 2), # convert to percentage
                "history": _training_state["history"]
            }
            update_version_status(
                version_id, 
                status=db_status, 
                metrics=metrics, 
                model_path=model_save_path,
                classes=classes
            )

    except Exception as e:
        tb = traceback.format_exc()
        _log(f"ERROR: {str(e)}", "ERROR")
        _log(tb, "ERROR")
        with _state_lock:
            _training_state["status"] = "error"
            _training_state["error"]  = str(e)
        
        # Update DB if version_id is provided
        version_id = cfg.get("version_id")
        if version_id:
            from app.core.db_helpers import update_version_status
            update_version_status(version_id, status="failed", metrics={"error": str(e)})
    finally:
        with _state_lock:
            _training_state["progress_bar"] = None
        # Only delete the zip if it was a temp upload (not the project's permanent dataset)
        try:
            if zip_path and "_train.zip" in zip_path and os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception:
            pass


# ─── Public start function ────────────────────────────────────────────────────

def start_training(graph_data, zip_path, extract_path, cfg, model_save_path):
    """
    Kicks off training in a background daemon thread.
    Safe to call multiple times (previous must be stopped first).
    """
    global _stop_event
    _stop_event = threading.Event()

    t = threading.Thread(
        target=_train_thread,
        args=(graph_data, zip_path, extract_path, cfg, model_save_path),
        daemon=True,
    )
    t.start()
    return t
