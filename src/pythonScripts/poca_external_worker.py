import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
import traceback
from multiprocessing import shared_memory

import numpy as np


class PocaComponent:
    def __init__(self, name, features):
        self.name = name
        self._features = features

    def __contains__(self, feature):
        return feature in self._features

    def __getitem__(self, feature):
        return self._features[feature]

    def get(self, feature, default=None):
        return self._features.get(feature, default)

    def features(self):
        return list(self._features.keys())


class PocaData:
    def __init__(self, inputs):
        self._components = {}
        self._actions = []
        for meta, arr in inputs:
            comp = meta.get("component", "Component")
            feat = meta.get("feature", meta.get("label", f"array_{len(self._components)}"))
            self._components.setdefault(comp, {})[feat] = arr

    def components(self):
        return list(self._components.keys())

    def component(self, name=None):
        if name is None:
            if "DetectionSet" in self._components:
                name = "DetectionSet"
            elif len(self._components) == 1:
                name = next(iter(self._components))
            else:
                raise KeyError("No component name provided and no unique/default component is available")
        if name not in self._components:
            raise KeyError(f"Component '{name}' is not available. Available: {self.components()}")
        return PocaComponent(name, self._components[name])

    def __getitem__(self, feature):
        return self.component()[feature]

    def get(self, feature, default=None):
        return self.component().get(feature, default)

    def has_feature(self, feature, component=None):
        try:
            return feature in self.component(component)
        except KeyError:
            return False

    def is_3d(self, component=None):
        return self.has_feature("z", component)

    def display(self, text):
        self._actions.append({"type": "display", "text": str(text)})

    def add_feature(self, component, feature, values):
        self._actions.append({
            "type": "add_feature",
            "component": component,
            "feature": feature,
            "values": np.asarray(values),
        })

    def create_dataset(self, name, components):
        self._actions.append({
            "type": "create_dataset",
            "name": name,
            "components": components,
        })

    def actions(self):
        return list(self._actions)


def _normalise_requirements(raw):
    """Return canonical script input requirements.

    User scripts may define either:
      POCA_INPUTS = {"DetectionSet": ["x", "y", {"name": "z", "optional": True}]}
      POCA_INPUTS = [{"component": "DetectionSet", "features": ["x", "y"]}]
      def poca_inputs(): return ...
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "component" in raw and "features" in raw:
            raw = [raw]
        else:
            raw = [{"component": comp, "features": feats} for comp, feats in raw.items()]
    if not isinstance(raw, (list, tuple)):
        raise TypeError("PoCA requirements must be a dict or list")
    result = []
    for req in raw:
        if not isinstance(req, dict):
            raise TypeError("Each PoCA requirement must be a dict")
        component = str(req.get("component", "DetectionSet"))
        features = []
        for f in req.get("features", []):
            if isinstance(f, str):
                features.append({"name": f, "optional": False})
            elif isinstance(f, dict):
                name = f.get("name", f.get("feature"))
                if not name:
                    raise ValueError("Feature requirement dict needs 'name' or 'feature'")
                features.append({"name": str(name), "optional": bool(f.get("optional", False))})
            else:
                raise TypeError("Feature requirements must be strings or dicts")
        result.append({"component": component, "features": features})
    return result


def _describe_user_script(script_path):
    module_name = "poca_user_script_describe_" + str(os.getpid()) + "_" + str(time.time_ns())
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "poca_inputs") and callable(module.poca_inputs):
        raw = module.poca_inputs()
    else:
        raw = getattr(module, "POCA_INPUTS", [])
    return _normalise_requirements(raw)


def _load_array(meta):
    shm = shared_memory.SharedMemory(name=meta["name"])
    dtype = np.dtype(meta.get("dtype", "float64"))
    shape = tuple(meta["shape"])
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr.setflags(write=False)
    return shm, arr


def _load_user_function(script_path, function_name):
    module_name = "poca_user_script_" + str(os.getpid()) + "_" + str(time.time_ns())
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, function_name, None)
    if func is None or not callable(func):
        raise RuntimeError(f"Function '{function_name}' was not found or is not callable in {script_path}")
    return func


def _normalise_legacy_outputs(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        if len(value) > 0 and all(np.ndim(v) > 0 for v in value):
            return value
    return [value]


def _append_array_payload(action, key, arr, output_shms):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype.kind not in ("f", "i", "u", "b"):
        raise TypeError(f"Unsupported array dtype for action {action.get('type')}: {arr.dtype}")
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
    output_shms.append(shm)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    action[key] = {
        "name": shm.name,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "nbytes": int(arr.nbytes),
    }


def _serialise_actions(actions, output_shms):
    result = []
    for raw in actions:
        if raw is None:
            continue
        if isinstance(raw, str):
            result.append({"type": "display", "text": raw})
            continue
        if not isinstance(raw, dict):
            raise TypeError(f"Unsupported action type returned by script: {type(raw)!r}")
        action = dict(raw)
        typ = action.get("type")
        if typ is None and "output_string" in action:
            result.append({"type": "display", "text": str(action["output_string"])})
            continue
        if typ == "display":
            action["text"] = str(action.get("text", action.get("output_string", "")))
            result.append(action)
        elif typ == "add_feature":
            values = action.pop("values", None)
            if values is None:
                raise ValueError("add_feature action requires a 'values' entry")
            _append_array_payload(action, "values_shm", values, output_shms)
            result.append(action)
        elif typ == "create_dataset":
            comps = action.get("components", {})
            serialised_components = {}
            for comp_name, features in comps.items():
                serialised_components[comp_name] = {}
                for feat_name, values in features.items():
                    payload_action = {"type": "array_payload"}
                    _append_array_payload(payload_action, "values_shm", values, output_shms)
                    serialised_components[comp_name][feat_name] = payload_action["values_shm"]
            action["components"] = serialised_components
            result.append(action)
        else:
            raise ValueError(f"Unknown PoCA action type: {typ}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()

    input_shms = []
    output_shms = []
    try:
        request = json.loads(sys.stdin.readline())
        if args.describe or request.get("api") == "poca_describe":
            response = {"ok": True, "requirements": _describe_user_script(args.script)}
            sys.__stdout__.write(json.dumps(response) + "\n")
            sys.__stdout__.flush()
            return 0

        loaded = []
        arrays = []
        for meta in request.get("inputs", []):
            shm, arr = _load_array(meta)
            input_shms.append(shm)
            loaded.append((meta, arr))
            arrays.append(arr)

        func = _load_user_function(args.script, args.function)
        stdout_capture = io.StringIO()

        if request.get("api") == "poca":
            poca = PocaData(loaded)
            with contextlib.redirect_stdout(stdout_capture):
                returned = func(poca)
            actions = poca.actions()
            if returned is not None:
                if isinstance(returned, dict):
                    if "actions" in returned:
                        actions.extend(returned["actions"])
                    else:
                        actions.append(returned)
                elif isinstance(returned, list):
                    actions.extend(returned)
                else:
                    actions.append(returned)
            printed = stdout_capture.getvalue()
            if printed:
                actions.insert(0, {"type": "display", "text": printed})
            response = {"ok": True, "actions": _serialise_actions(actions, output_shms)}
        else:
            with contextlib.redirect_stdout(stdout_capture):
                result = func(*arrays)
            response = {"ok": True, "outputs": [], "stdout": stdout_capture.getvalue()}
            for index, out in enumerate(_normalise_legacy_outputs(result)):
                action = {"type": "legacy_output", "index": index}
                _append_array_payload(action, "values_shm", out, output_shms)
                payload = action["values_shm"]
                payload["index"] = index
                response["outputs"].append(payload)

        sys.__stdout__.write(json.dumps(response) + "\n")
        sys.__stdout__.flush()

        sys.stdin.readline().strip()
        return 0
    except Exception as exc:
        sys.__stdout__.write(json.dumps({
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }) + "\n")
        sys.__stdout__.flush()
        return 1
    finally:
        for shm in input_shms:
            shm.close()
        for shm in output_shms:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    sys.exit(main())
