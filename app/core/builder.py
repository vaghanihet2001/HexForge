import torch
import torch.nn as nn
import ast
from collections import OrderedDict


def build_model(graph_data):
    """
    Parses a JSON graph from the frontend and returns:
    (PyTorch nn.Sequential model, input_shape_list, execution_order_nodes)
    """
    if not graph_data:
        raise ValueError("Empty graph data")

    nodes_by_id = {str(node['id']): node for node in graph_data}

    input_node = next((n for n in graph_data if n.get('type') == 'input'), None)
    if not input_node:
        raise ValueError("Graph must contain an Input node")

    shape_str = input_node['properties'].get('shape', '[1, 3, 224, 224]')
    try:
        input_shape = list(ast.literal_eval(shape_str))
    except (ValueError, SyntaxError):
        input_shape = [1, 3, 224, 224]

    # Topological traversal (sequential / first-child)
    execution_order, visited = [], set()
    current_node = input_node
    while current_node:
        nid = str(current_node['id'])
        if nid in visited:
            break
        visited.add(nid)
        execution_order.append(current_node)
        children = current_node.get('children', [])
        if not children:
            break
        current_node = nodes_by_id.get(str(children[0]))

    layers, layer_names = [], []

    for node in execution_order:
        node_type = node.get('type')
        props     = node.get('properties', {})
        node_id   = str(node.get('id', 'x'))

        if node_type == 'input':
            continue

        # ---- Convolutions ----
        elif node_type == 'conv1d':
            layers.append(nn.Conv1d(
                int(props.get('in_channels',  1)),
                int(props.get('out_channels', 16)),
                int(props.get('kernel_size',  3)),
                stride=int(props.get('stride',  1)),
                padding=int(props.get('padding', 1))
            ))
            layer_names.append(f"conv1d_{node_id}")

        elif node_type == 'conv2d':
            layers.append(nn.Conv2d(
                int(props.get('in_channels',  3)),
                int(props.get('out_channels', 16)),
                int(props.get('kernel_size',  3)),
                stride=int(props.get('stride',  1)),
                padding=int(props.get('padding', 1))
            ))
            layer_names.append(f"conv2d_{node_id}")

        elif node_type == 'convtranspose2d':
            layers.append(nn.ConvTranspose2d(
                int(props.get('in_channels',   16)),
                int(props.get('out_channels',   3)),
                int(props.get('kernel_size',    3)),
                stride=int(props.get('stride',         2)),
                padding=int(props.get('padding',       1)),
                output_padding=int(props.get('output_padding', 1))
            ))
            layer_names.append(f"convtranspose2d_{node_id}")

        # ---- Linear ----
        elif node_type == 'linear':
            layers.append(nn.Linear(
                int(props.get('in_features',  128)),
                int(props.get('out_features',  10))
            ))
            layer_names.append(f"linear_{node_id}")

        # ---- Activations ----
        elif node_type == 'relu':
            layers.append(nn.ReLU()); layer_names.append(f"relu_{node_id}")

        elif node_type == 'leakyrelu':
            layers.append(nn.LeakyReLU(float(props.get('negative_slope', 0.01))))
            layer_names.append(f"leakyrelu_{node_id}")

        elif node_type == 'prelu':
            layers.append(nn.PReLU(num_parameters=int(props.get('num_parameters', 1))))
            layer_names.append(f"prelu_{node_id}")

        elif node_type == 'sigmoid':
            layers.append(nn.Sigmoid()); layer_names.append(f"sigmoid_{node_id}")

        elif node_type == 'tanh':
            layers.append(nn.Tanh()); layer_names.append(f"tanh_{node_id}")

        elif node_type == 'gelu':
            layers.append(nn.GELU()); layer_names.append(f"gelu_{node_id}")

        elif node_type == 'silu':
            layers.append(nn.SiLU()); layer_names.append(f"silu_{node_id}")

        elif node_type == 'elu':
            layers.append(nn.ELU(alpha=float(props.get('alpha', 1.0))))
            layer_names.append(f"elu_{node_id}")

        elif node_type == 'softmax':
            layers.append(nn.Softmax(dim=int(props.get('dim', 1))))
            layer_names.append(f"softmax_{node_id}")

        # ---- Pooling ----
        elif node_type == 'maxpool2d':
            layers.append(nn.MaxPool2d(
                int(props.get('kernel_size', 2)),
                stride=int(props.get('stride', 2))
            ))
            layer_names.append(f"maxpool2d_{node_id}")

        elif node_type == 'avgpool2d':
            layers.append(nn.AvgPool2d(
                int(props.get('kernel_size', 2)),
                stride=int(props.get('stride',   2)),
                padding=int(props.get('padding', 0))
            ))
            layer_names.append(f"avgpool2d_{node_id}")

        elif node_type == 'adaptiveavgpool2d':
            size_raw = props.get('output_size', '[1, 1]')
            try:
                size = tuple(ast.literal_eval(size_raw))
            except Exception:
                size = (1, 1)
            layers.append(nn.AdaptiveAvgPool2d(size))
            layer_names.append(f"adaptiveavgpool2d_{node_id}")

        elif node_type == 'globalavgpool2d':
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layer_names.append(f"globalavgpool2d_{node_id}")

        elif node_type == 'maxpool1d':
            layers.append(nn.MaxPool1d(
                int(props.get('kernel_size', 2)),
                stride=int(props.get('stride', 2)),
                padding=int(props.get('padding', 0))
            ))
            layer_names.append(f"maxpool1d_{node_id}")

        elif node_type == 'avgpool1d':
            layers.append(nn.AvgPool1d(
                int(props.get('kernel_size', 2)),
                stride=int(props.get('stride', 2)),
                padding=int(props.get('padding', 0))
            ))
            layer_names.append(f"avgpool1d_{node_id}")

        # ---- Normalisation ----
        elif node_type == 'batchnorm2d':
            layers.append(nn.BatchNorm2d(
                int(props.get('num_features', 16)),
                eps=float(props.get('eps', 1e-5)),
                momentum=float(props.get('momentum', 0.1))
            ))
            layer_names.append(f"batchnorm2d_{node_id}")

        elif node_type == 'batchnorm1d':
            layers.append(nn.BatchNorm1d(
                int(props.get('num_features', 16)),
                eps=float(props.get('eps', 1e-5)),
                momentum=float(props.get('momentum', 0.1))
            ))
            layer_names.append(f"batchnorm1d_{node_id}")

        elif node_type == 'layernorm':
            shape_raw = props.get('normalized_shape', '[16]')
            try:
                norm_shape = list(ast.literal_eval(shape_raw))
            except Exception:
                norm_shape = [16]
            layers.append(nn.LayerNorm(norm_shape))
            layer_names.append(f"layernorm_{node_id}")

        elif node_type == 'dropout':
            layers.append(nn.Dropout(p=float(props.get('p', 0.5))))
            layer_names.append(f"dropout_{node_id}")

        # ---- Shape / Utility ----
        elif node_type == 'flatten':
            layers.append(nn.Flatten(start_dim=int(props.get('start_dim', 1))))
            layer_names.append(f"flatten_{node_id}")

        elif node_type == 'reshape':
            shape_raw = props.get('shape', '[-1, 128]')
            try:
                target = tuple(ast.literal_eval(shape_raw))
            except Exception:
                target = (-1, 128)
            # Wrap reshape in a Lambda
            layers.append(_ReshapeModule(target))
            layer_names.append(f"reshape_{node_id}")

        elif node_type == 'upsample':
            layers.append(nn.Upsample(
                scale_factor=float(props.get('scale_factor', 2)),
                mode=props.get('mode', 'nearest')
            ))
            layer_names.append(f"upsample_{node_id}")

        elif node_type == 'concat':
            layers.append(_ConcatModule(int(props.get('dim', 1))))
            layer_names.append(f"concat_{node_id}")

        elif node_type == 'split':
            layers.append(_SplitModule(int(props.get('split_size', 1)), int(props.get('dim', 1))))
            layer_names.append(f"split_{node_id}")

        elif node_type == 'transpose':
            layers.append(_TransposeModule(int(props.get('dim0', 0)), int(props.get('dim1', 1))))
            layer_names.append(f"transpose_{node_id}")

        elif node_type == 'squeeze':
            layers.append(_SqueezeModule(int(props.get('dim', 0))))
            layer_names.append(f"squeeze_{node_id}")

        elif node_type == 'unsqueeze':
            layers.append(_UnsqueezeModule(int(props.get('dim', 0))))
            layer_names.append(f"unsqueeze_{node_id}")

        # ---- Recurrent (wrapped for Sequential compatibility) ----
        elif node_type == 'lstm':
            layers.append(_LSTMWrapper(
                input_size=int(props.get('input_size',  128)),
                hidden_size=int(props.get('hidden_size', 256)),
                num_layers=int(props.get('num_layers',    1)),
                bidirectional=bool(props.get('bidirectional', False))
            ))
            layer_names.append(f"lstm_{node_id}")

        elif node_type == 'gru':
            layers.append(_GRUWrapper(
                input_size=int(props.get('input_size',  128)),
                hidden_size=int(props.get('hidden_size', 256)),
                num_layers=int(props.get('num_layers',    1)),
                bidirectional=bool(props.get('bidirectional', False))
            ))
            layer_names.append(f"gru_{node_id}")

        # Unknown / generic → identity
        else:
            layers.append(nn.Identity())
            layer_names.append(f"identity_{node_id}")

    if not layers:
        layers.append(nn.Identity())
        layer_names.append("identity_0")

    od    = OrderedDict(zip(layer_names, layers))
    model = nn.Sequential(od)
    return model, input_shape, execution_order


# ── helper modules ──────────────────────────────────────────

class _ReshapeModule(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class _LSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class _GRUWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class _ConcatModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat([x, x], dim=self.dim)


class _SplitModule(nn.Module):
    def __init__(self, split_size, dim):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        chunks = torch.split(x, self.split_size, dim=self.dim)
        return chunks[0] if chunks else x


class _TransposeModule(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class _SqueezeModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        try:
            return torch.squeeze(x, self.dim)
        except Exception:
            return torch.squeeze(x)


class _UnsqueezeModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        d = min(max(self.dim, -x.ndim - 1), x.ndim)
        return torch.unsqueeze(x, d)
