document.addEventListener("DOMContentLoaded", () => {
    var graph  = new LiteGraph.LGraph();
    var canvas = new LiteGraph.LGraphCanvas("#editor-canvas", graph);

    // Trigger shape inference on connection/node changes
    graph.onNodeConnectionChange = function(type, node, slot, target_node, target_slot) {
        if (window.TriggerShapeInference) {
            window.TriggerShapeInference();
        }
    };
    graph.onNodeAdded = function(node) {
        if (window.TriggerShapeInference) {
            window.TriggerShapeInference();
        }
    };
    graph.onNodeRemoved = function(node) {
        if (window.TriggerShapeInference) {
            window.TriggerShapeInference();
        }
    };
    
    // Trigger shape inference on widget changes
    LiteGraph.LGraphNode.prototype.onWidgetChanged = function(name, value, old_value, widget) {
        if (window.TriggerShapeInference) {
            window.TriggerShapeInference();
        }
    };

    // Trigger shape inference on property changes
    LiteGraph.LGraphNode.prototype.onPropertyChanged = function(name, value, prev_value) {
        if (window.TriggerShapeInference) {
            window.TriggerShapeInference();
        }
    };

    const container = document.getElementById('editor-canvas').parentElement;
    canvas.resize(container.clientWidth, container.clientHeight);

    // ── NODE REGISTRY (type → meta) ──────────────────────────
    const NODE_META = {
        "pytorch/input":             { label:"Input",            color:"#1a6b42", fields:["shape"] },
        "pytorch/conv1d":            { label:"Conv1d",           color:"#1565c0", fields:["in_channels","out_channels","kernel_size","stride","padding"] },
        "pytorch/conv2d":            { label:"Conv2d",           color:"#1565c0", fields:["in_channels","out_channels","kernel_size","stride","padding"] },
        "pytorch/convtranspose2d":   { label:"ConvTranspose2d",  color:"#1976d2", fields:["in_channels","out_channels","kernel_size","stride","padding","output_padding"] },
        "pytorch/linear":            { label:"Linear",           color:"#4527a0", fields:["in_features","out_features"] },
        "pytorch/relu":              { label:"ReLU",             color:"#b45309", fields:[] },
        "pytorch/leakyrelu":         { label:"LeakyReLU",        color:"#b45309", fields:["negative_slope"] },
        "pytorch/sigmoid":           { label:"Sigmoid",          color:"#9d174d", fields:[] },
        "pytorch/tanh":              { label:"Tanh",             color:"#9d174d", fields:[] },
        "pytorch/gelu":              { label:"GELU",             color:"#be185d", fields:[] },
        "pytorch/silu":              { label:"SiLU",             color:"#be185d", fields:[] },
        "pytorch/elu":               { label:"ELU",              color:"#9d174d", fields:["alpha"] },
        "pytorch/softmax":           { label:"Softmax",          color:"#9d174d", fields:["dim"] },
        "pytorch/maxpool2d":         { label:"MaxPool2d",        color:"#6a1b9a", fields:["kernel_size","stride"] },
        "pytorch/avgpool2d":         { label:"AvgPool2d",        color:"#7b1fa2", fields:["kernel_size","stride","padding"] },
        "pytorch/adaptiveavgpool2d": { label:"AdaptiveAvgPool2d",color:"#7b1fa2", fields:["output_size"] },
        "pytorch/globalavgpool2d":   { label:"GlobalAvgPool2d",  color:"#7b1fa2", fields:[] },
        "pytorch/maxpool1d":         { label:"MaxPool1d",        color:"#6a1b9a", fields:["kernel_size","stride","padding"] },
        "pytorch/avgpool1d":         { label:"AvgPool1d",        color:"#7b1fa2", fields:["kernel_size","stride","padding"] },
        "pytorch/batchnorm2d":       { label:"BatchNorm2d",      color:"#006064", fields:["num_features","momentum"] },
        "pytorch/batchnorm1d":       { label:"BatchNorm1d",      color:"#006064", fields:["num_features","momentum"] },
        "pytorch/prelu":             { label:"PReLU",            color:"#b45309", fields:["num_parameters"] },
        "pytorch/layernorm":         { label:"LayerNorm",        color:"#006064", fields:["normalized_shape"] },
        "pytorch/dropout":           { label:"Dropout",          color:"#00695c", fields:["p"] },
        "pytorch/flatten":           { label:"Flatten",          color:"#455a64", fields:["start_dim"] },
        "pytorch/reshape":           { label:"Reshape",          color:"#455a64", fields:["shape"] },
        "pytorch/upsample":          { label:"Upsample",         color:"#37474f", fields:["scale_factor","mode"] },
        "pytorch/concat":            { label:"Concat",           color:"#37474f", fields:["dim"] },
        "pytorch/split":             { label:"Split",            color:"#455a64", fields:["split_size","dim"] },
        "pytorch/transpose":         { label:"Transpose",        color:"#37474f", fields:["dim0","dim1"] },
        "pytorch/squeeze":           { label:"Squeeze",          color:"#455a64", fields:["dim"] },
        "pytorch/unsqueeze":         { label:"Unsqueeze",        color:"#455a64", fields:["dim"] },
        "pytorch/lstm":              { label:"LSTM",             color:"#bf360c", fields:["input_size","hidden_size","num_layers","bidirectional"] },
        "pytorch/gru":               { label:"GRU",              color:"#e64a19", fields:["input_size","hidden_size","num_layers","bidirectional"] },
        "pytorch/add":               { label:"Add",              color:"#0d47a1", fields:[] },
        "pytorch/sub":               { label:"Subtract",         color:"#0d47a1", fields:[] },
        "pytorch/mul":               { label:"Multiply",         color:"#0d47a1", fields:[] },
        "pytorch/div":               { label:"Divide",           color:"#0d47a1", fields:[] },
        "pytorch/matmul":            { label:"MatMul",           color:"#1565c0", fields:[] },
        "pytorch/generic":           { label:"Generic",          color:"#880e4f", fields:["op_type","name"] },
    };

    // ── FIELD CONFIGS ────────────────────────────────────────
    const FIELD_CONFIG = {
        shape:           { type:"text",  label:"Shape",          placeholder:"[1, 3, 224, 224]" },
        in_channels:     { type:"number",label:"In Channels",    min:1,  step:1  },
        out_channels:    { type:"number",label:"Out Channels",   min:1,  step:1  },
        kernel_size:     { type:"number",label:"Kernel Size",    min:1,  step:1  },
        stride:          { type:"number",label:"Stride",         min:1,  step:1  },
        padding:         { type:"number",label:"Padding",        min:0,  step:1  },
        output_padding:  { type:"number",label:"Output Padding", min:0,  step:1  },
        in_features:     { type:"number",label:"In Features",    min:1,  step:1  },
        out_features:    { type:"number",label:"Out Features",   min:1,  step:1  },
        negative_slope:  { type:"number",label:"Neg Slope",      min:0,  step:0.01 },
        alpha:           { type:"number",label:"Alpha",          min:0,  step:0.1  },
        dim:             { type:"number",label:"Dim",            step:1  },
        num_features:    { type:"number",label:"Num Features",   min:1,  step:1  },
        momentum:        { type:"number",label:"Momentum",       min:0,  max:1, step:0.01 },
        normalized_shape:{ type:"text",  label:"Norm Shape",     placeholder:"[64]" },
        p:               { type:"number",label:"Dropout p",      min:0,  max:1, step:0.05 },
        start_dim:       { type:"number",label:"Start Dim",      min:0,  step:1  },
        output_size:     { type:"text",  label:"Output Size",    placeholder:"[1, 1]" },
        scale_factor:    { type:"number",label:"Scale Factor",   min:0.5,step:0.5 },
        mode:            { type:"select",label:"Mode",           options:["nearest","bilinear","bicubic"] },
        input_size:      { type:"number",label:"Input Size",     min:1,  step:1  },
        hidden_size:     { type:"number",label:"Hidden Size",    min:1,  step:1  },
        num_layers:      { type:"number",label:"Num Layers",     min:1,  step:1  },
        bidirectional:   { type:"select",label:"Bidirectional",  options:["false","true"] },
        op_type:         { type:"text",  label:"Op Type",        placeholder:"Unknown" },
        name:            { type:"text",  label:"Name",           placeholder:"" },
        num_parameters:  { type:"number",label:"Num Parameters", min:1,  step:1  },
        split_size:      { type:"number",label:"Split Size",     min:1,  step:1  },
        dim0:            { type:"number",label:"Dim 0",          step:1  },
        dim1:            { type:"number",label:"Dim 1",          step:1  },
    };

    function getPropertyKeyFromWidgetName(wName) {
        const lower = wName.toLowerCase().trim();
        if (lower === "neg slope") return "negative_slope";
        if (lower === "out padding") return "output_padding";
        if (lower === "dropout p") return "p";
        if (lower === "dim 0") return "dim0";
        if (lower === "dim 1") return "dim1";
        return lower.replace(/\s/g, '_');
    }
    window.GetPropertyKeyFromWidgetName = getPropertyKeyFromWidgetName;

    function getBaseTitle(node) {
        let title = node.title || "";
        title = title.replace(/\s*\([^)]*ms\)(?:\s*\|\s*Out:\s*(?:\[[^\]]*\])?)?/g, "");
        title = title.replace(/\s*\(⚠️ Error\)/g, "");
        return title.trim();
    }
    window.GetBaseTitle = getBaseTitle;

    function syncAllNodeWidgets(g) {
        if (!g || !g._nodes) return;
        g._nodes.forEach(node => {
            if (node.widgets && node.properties) {
                node.widgets.forEach(w => {
                    if (w.name) {
                        const propKey = getPropertyKeyFromWidgetName(w.name);
                        if (node.properties[propKey] !== undefined) {
                            w.value = node.properties[propKey];
                        }
                    }
                });
            }
        });
    }
    window.SyncAllNodeWidgets = syncAllNodeWidgets;

    function refreshStatusWidgets(node) {
        if (!node.widgets) {
            node.widgets = [];
        }
        
        const drawStatusWidget = function(ctx, node, widget_width, y, H) {
            const margin = 15;
            const inner_width = widget_width - margin * 2;
            
            // Draw background box
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#666";
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR || "#222";
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(margin, y, inner_width, H, [H * 0.5]);
            } else {
                ctx.rect(margin, y, inner_width, H);
            }
            ctx.fill();
            if (!this.disabled) {
                ctx.stroke();
            }
            
            // Draw label
            ctx.font = "10px sans-serif";
            ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999";
            ctx.textAlign = "left";
            const label = this.label || this.name;
            ctx.fillText(label, margin * 2, y + H * 0.7);
            
            // Draw value on the right
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#FFF";
            ctx.textAlign = "right";
            
            const labelWidth = ctx.measureText(label).width;
            const maxValWidth = inner_width - labelWidth - 15;
            
            let valStr = String(this.value);
            ctx.font = "9px monospace";
            let valWidth = ctx.measureText(valStr).width;
            
            if (valWidth > maxValWidth) {
                // Truncate if still too long
                while (valStr.length > 5 && valWidth > maxValWidth) {
                    valStr = valStr.slice(0, -4) + "...";
                    valWidth = ctx.measureText(valStr).width;
                }
            }
            
            ctx.fillText(valStr, widget_width - margin * 2 - 8, y + H * 0.7);
        };
        
        // Output Shape widget
        if (node.properties && node.properties.output_shape) {
            let w = node.widgets.find(x => x.name === "Output Shape" || x.name === "Out Shape");
            if (!w) {
                w = node.addWidget("status", "Out Shape", node.properties.output_shape, () => {}, { disabled: true });
                if (w) {
                    w.disabled = true;
                    w.draw = drawStatusWidget;
                }
            } else {
                w.name = "Out Shape";
                w.value = node.properties.output_shape;
                w.draw = drawStatusWidget;
            }
        } else {
            node.widgets = node.widgets.filter(x => x.name !== "Output Shape" && x.name !== "Out Shape");
        }

        // Latency widget
        if (node.properties && node.properties.latency) {
            let w = node.widgets.find(x => x.name === "Latency");
            if (!w) {
                w = node.addWidget("status", "Latency", node.properties.latency, () => {}, { disabled: true });
                if (w) {
                    w.disabled = true;
                    w.draw = drawStatusWidget;
                }
            } else {
                w.value = node.properties.latency;
                w.draw = drawStatusWidget;
            }
        } else {
            node.widgets = node.widgets.filter(x => x.name !== "Latency");
        }
        
        node.setSize(node.computeSize());
    }
    window.RefreshStatusWidgets = refreshStatusWidgets;

    // ── RIGHT PANEL ──────────────────────────────────────────
    const panel      = document.getElementById('node-props-panel');
    const shell      = document.getElementById('editor-shell');
    const nppTitle   = document.getElementById('npp-title');
    const nppBody    = document.getElementById('npp-body');
    const nppDelete  = document.getElementById('npp-delete');
    const nppClose   = document.getElementById('npp-close');
    let   activeNode = null;

    function updateNodePanelShapes(node, isLoading = false) {
        if (!activeNode || activeNode.id !== node.id) return;
        
        let shapesContainer = document.getElementById("npp-shapes-container");
        if (!shapesContainer) {
            shapesContainer = document.createElement('div');
            shapesContainer.id = "npp-shapes-container";
            shapesContainer.style.cssText = 'display:flex; flex-direction:column; gap:6px; margin-bottom:12px;';
            
            const badge = nppBody.querySelector('.npp-type-badge');
            if (badge && badge.nextSibling) {
                nppBody.insertBefore(shapesContainer, badge.nextSibling);
            } else {
                nppBody.appendChild(shapesContainer);
            }
        }
        
        shapesContainer.innerHTML = '';
        
        if (isLoading) {
            const loadingBadge = document.createElement('div');
            loadingBadge.style.cssText = 'font-size:11.5px;padding:6px 10px;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);color:var(--yellow);border-radius:var(--radius-sm);font-family:var(--font-mono);display:flex;align-items:center;gap:10px;';
            loadingBadge.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> <span>Calculating shapes...</span>`;
            shapesContainer.appendChild(loadingBadge);
            return;
        }
        
        const inShape = node.properties ? node.properties.input_shape : null;
        const outShape = node.properties ? (node.properties.output_shape || (node.type === "pytorch/input" ? node.properties.shape : null)) : null;
        
        if (inShape || outShape) {
            if (inShape) {
                const inShapeBadge = document.createElement('div');
                inShapeBadge.style.cssText = 'font-size:11.5px;padding:6px 10px;background:rgba(51,204,255,0.08);border:1px solid rgba(51,204,255,0.2);color:var(--accent);border-radius:var(--radius-sm);font-family:var(--font-mono);display:flex;justify-content:space-between;align-items:center;gap:10px;';
                inShapeBadge.innerHTML = `<span>In Shape:</span><strong>${inShape}</strong>`;
                shapesContainer.appendChild(inShapeBadge);
            }
            if (outShape) {
                const outShapeBadge = document.createElement('div');
                outShapeBadge.style.cssText = 'font-size:11.5px;padding:6px 10px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);color:var(--green);border-radius:var(--radius-sm);font-family:var(--font-mono);display:flex;justify-content:space-between;align-items:center;gap:10px;';
                outShapeBadge.innerHTML = `<span>Out Shape:</span><strong>${outShape}</strong>`;
                shapesContainer.appendChild(outShapeBadge);
            }
        } else {
            shapesContainer.remove();
        }
    }
    window.UpdateNodePanelShapes = updateNodePanelShapes;
    window.GetActiveNode = () => activeNode;

    function openPanel(node) {
        activeNode = node;
        const meta = NODE_META[node.type] || { label: node.type, fields:[] };

        nppTitle.textContent = meta.label;
        nppBody.innerHTML = '';

        // Type badge
        const badge = document.createElement('div');
        badge.className = 'npp-type-badge';
        badge.textContent = node.type.replace('pytorch/', '').toUpperCase();
        nppBody.appendChild(badge);

        // Input & Output Shapes badges if present
        updateNodePanelShapes(node, false);

        // Node ID info
        const idRow = document.createElement('div');
        idRow.style.cssText = 'font-size:11px;color:var(--text-muted);margin-bottom:12px;font-family:var(--font-mono);';
        idRow.textContent = `ID: ${node.id}`;
        nppBody.appendChild(idRow);

        // Properties
        if (meta.fields.length === 0) {
            const info = document.createElement('p');
            info.style.cssText = 'font-size:12px;color:var(--text-muted);';
            info.textContent = 'No configurable properties.';
            nppBody.appendChild(info);
        } else {
            const section = document.createElement('div');
            section.className = 'npp-section';
            const stitle = document.createElement('p');
            stitle.className = 'npp-section-title';
            stitle.textContent = 'Properties';
            section.appendChild(stitle);

            meta.fields.forEach(fieldKey => {
                const cfg = FIELD_CONFIG[fieldKey];
                if (!cfg) return;
                const currentVal = node.properties ? node.properties[fieldKey] : undefined;

                const wrap = document.createElement('div');
                wrap.className = 'npp-field';

                const lbl = document.createElement('label');
                lbl.textContent = cfg.label;
                wrap.appendChild(lbl);

                let input;
                if (cfg.type === 'select') {
                    input = document.createElement('select');
                    cfg.options.forEach(opt => {
                        const o = document.createElement('option');
                        o.value = opt; o.textContent = opt;
                        if (String(currentVal) === opt) o.selected = true;
                        input.appendChild(o);
                    });
                } else {
                    input = document.createElement('input');
                    input.type = cfg.type === 'text' ? 'text' : 'number';
                    if (cfg.min  !== undefined) input.min  = cfg.min;
                    if (cfg.max  !== undefined) input.max  = cfg.max;
                    if (cfg.step !== undefined) input.step = cfg.step;
                    if (cfg.placeholder) input.placeholder = cfg.placeholder;
                    input.value = currentVal !== undefined ? currentVal : '';
                }

                input.addEventListener('change', () => {
                    if (!node.properties) node.properties = {};
                    const v = cfg.type === 'number' ? Number(input.value) : input.value;
                    node.properties[fieldKey] = v;
                    // sync LiteGraph widget if present
                    if (node.widgets) {
                        node.widgets.forEach(w => {
                            if (w.name && getPropertyKeyFromWidgetName(w.name) === fieldKey) {
                                w.value = v;
                            }
                        });
                    }
                    if (LiteGraph.LGraphCanvas.active_canvas)
                        LiteGraph.LGraphCanvas.active_canvas.setDirty(true, true);
                    if (window.TriggerShapeInference)
                        window.TriggerShapeInference();
                });

                wrap.appendChild(input);
                section.appendChild(wrap);
            });
            nppBody.appendChild(section);
        }

        // Render extra properties (like imported constant values or weights)
        if (node.properties) {
            const extraKeys = Object.keys(node.properties).filter(k => 
                !meta.fields.includes(k) && 
                k !== "op_type" && 
                k !== "name"
            );
            if (extraKeys.length > 0) {
                const constSection = document.createElement('div');
                constSection.className = 'npp-section';
                
                const constTitle = document.createElement('p');
                constTitle.className = 'npp-section-title';
                constTitle.textContent = 'Constants & Weights';
                constSection.appendChild(constTitle);

                extraKeys.forEach(k => {
                    const wrap = document.createElement('div');
                    wrap.className = 'npp-field';

                    const lbl = document.createElement('label');
                    lbl.textContent = k;
                    wrap.appendChild(lbl);

                    const input = document.createElement('input');
                    input.type = 'text';
                    const val = node.properties[k];
                    input.value = typeof val === 'object' ? JSON.stringify(val) : val;

                    input.addEventListener('change', () => {
                        try {
                            if (input.value.startsWith('[') || input.value.startsWith('{')) {
                                node.properties[k] = JSON.parse(input.value);
                            } else {
                                const num = Number(input.value);
                                node.properties[k] = !isNaN(num) ? num : input.value;
                            }
                        } catch(e) {
                            node.properties[k] = input.value;
                        }
                        
                        // Sync widget if it exists
                        if (node.widgets) {
                            node.widgets.forEach(w => {
                                if (w.name === k) {
                                    w.value = typeof node.properties[k] === 'object' ? JSON.stringify(node.properties[k]) : node.properties[k];
                                }
                            });
                        }
                        if (LiteGraph.LGraphCanvas.active_canvas) {
                            LiteGraph.LGraphCanvas.active_canvas.setDirty(true, true);
                        }
                        if (window.TriggerShapeInference) {
                            window.TriggerShapeInference();
                        }
                    });

                    wrap.appendChild(input);
                    constSection.appendChild(wrap);
                });
                nppBody.appendChild(constSection);
            }
        }

        panel.classList.remove('hidden');
        shell.classList.add('panel-open');
        canvas.resize(container.clientWidth, container.clientHeight);
    }

    function closePanel() {
        panel.classList.add('hidden');
        shell.classList.remove('panel-open');
        activeNode = null;
        canvas.resize(container.clientWidth, container.clientHeight);
    }

    nppClose.addEventListener('click', closePanel);

    nppDelete.addEventListener('click', () => {
        if (activeNode && window.AppGraph) {
            window.AppGraph.remove(activeNode);
            if (LiteGraph.LGraphCanvas.active_canvas)
                LiteGraph.LGraphCanvas.active_canvas.setDirty(true, true);
            closePanel();
        }
    });

    // Hook into LiteGraph node selection
    canvas.onNodeSelected = (node) => { if (node) openPanel(node); };
    canvas.onNodeDeselected = () => { closePanel(); };

    // Also handle double-click (LiteGraph fires onNodeDblClick)
    canvas.onShowNodePanel = (node) => { openPanel(node); };

    // ── NODE REGISTRATION ────────────────────────────────────
    function makeSimple(type, color) {
        function N() {
            this.addInput("Input","tensor");
            this.addOutput("Output","tensor");
            this.color = color;
        }
        N.title = NODE_META[type]?.label || type;
        LiteGraph.registerNodeType(type, N);
    }

    // Input
    function InputNode() {
        this.addOutput("Tensor","tensor");
        this.properties = { shape:"[1, 3, 224, 224]" };
        this.addWidget("text","Shape",this.properties.shape,v=>{ this.properties.shape=v; });
        this.color = "#1a6b42";
    }
    InputNode.title = "Input";
    LiteGraph.registerNodeType("pytorch/input", InputNode);

    // Conv1d
    function Conv1dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={in_channels:1,out_channels:16,kernel_size:3,stride:1,padding:1};
        this.addWidget("number","In Channels",1,v=>{this.properties.in_channels=v;},{step:1,min:1});
        this.addWidget("number","Out Channels",16,v=>{this.properties.out_channels=v;},{step:1,min:1});
        this.addWidget("number","Kernel Size",3,v=>{this.properties.kernel_size=v;},{step:1,min:1});
        this.addWidget("number","Stride",1,v=>{this.properties.stride=v;},{step:1,min:1});
        this.addWidget("number","Padding",1,v=>{this.properties.padding=v;},{step:1,min:0});
        this.color="#1565c0";
    }
    Conv1dNode.title="Conv1d"; LiteGraph.registerNodeType("pytorch/conv1d",Conv1dNode);

    // Conv2d
    function Conv2dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={in_channels:3,out_channels:16,kernel_size:3,stride:1,padding:1};
        this.addWidget("number","In Channels",3,v=>{this.properties.in_channels=v;},{step:1,min:1});
        this.addWidget("number","Out Channels",16,v=>{this.properties.out_channels=v;},{step:1,min:1});
        this.addWidget("number","Kernel Size",3,v=>{this.properties.kernel_size=v;},{step:1,min:1});
        this.addWidget("number","Stride",1,v=>{this.properties.stride=v;},{step:1,min:1});
        this.addWidget("number","Padding",1,v=>{this.properties.padding=v;},{step:1,min:0});
        this.color="#1565c0";
    }
    Conv2dNode.title="Conv2d"; LiteGraph.registerNodeType("pytorch/conv2d",Conv2dNode);

    // ConvTranspose2d
    function ConvTranspose2dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={in_channels:16,out_channels:3,kernel_size:3,stride:2,padding:1,output_padding:1};
        this.addWidget("number","In Channels",16,v=>{this.properties.in_channels=v;},{step:1,min:1});
        this.addWidget("number","Out Channels",3,v=>{this.properties.out_channels=v;},{step:1,min:1});
        this.addWidget("number","Kernel Size",3,v=>{this.properties.kernel_size=v;},{step:1,min:1});
        this.addWidget("number","Stride",2,v=>{this.properties.stride=v;},{step:1,min:1});
        this.addWidget("number","Padding",1,v=>{this.properties.padding=v;},{step:1,min:0});
        this.addWidget("number","Out Padding",1,v=>{this.properties.output_padding=v;},{step:1,min:0});
        this.color="#1976d2";
    }
    ConvTranspose2dNode.title="ConvTranspose2d"; LiteGraph.registerNodeType("pytorch/convtranspose2d",ConvTranspose2dNode);

    // Linear
    function LinearNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={in_features:128,out_features:10};
        this.addWidget("number","In Features",128,v=>{this.properties.in_features=v;},{step:1,min:1});
        this.addWidget("number","Out Features",10,v=>{this.properties.out_features=v;},{step:1,min:1});
        this.color="#4527a0";
    }
    LinearNode.title="Linear"; LiteGraph.registerNodeType("pytorch/linear",LinearNode);

    // Activations (simple)
    makeSimple("pytorch/relu",    "#b45309");
    makeSimple("pytorch/sigmoid", "#9d174d");
    makeSimple("pytorch/tanh",    "#9d174d");
    makeSimple("pytorch/gelu",    "#be185d");
    makeSimple("pytorch/silu",    "#be185d");

    function LeakyReLUNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={negative_slope:0.01};
        this.addWidget("number","Neg Slope",0.01,v=>{this.properties.negative_slope=v;},{step:0.01,min:0});
        this.color="#b45309";
    }
    LeakyReLUNode.title="LeakyReLU"; LiteGraph.registerNodeType("pytorch/leakyrelu",LeakyReLUNode);

    function SoftmaxNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={dim:1};
        this.addWidget("number","Dim",1,v=>{this.properties.dim=v;},{step:1});
        this.color="#9d174d";
    }
    SoftmaxNode.title="Softmax"; LiteGraph.registerNodeType("pytorch/softmax",SoftmaxNode);

    function ELUNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={alpha:1.0};
        this.addWidget("number","Alpha",1.0,v=>{this.properties.alpha=v;},{step:0.1,min:0});
        this.color="#9d174d";
    }
    ELUNode.title="ELU"; LiteGraph.registerNodeType("pytorch/elu",ELUNode);

    // Pooling
    function MaxPool2dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={kernel_size:2,stride:2};
        this.addWidget("number","Kernel Size",2,v=>{this.properties.kernel_size=v;},{step:1,min:1});
        this.addWidget("number","Stride",2,v=>{this.properties.stride=v;},{step:1,min:1});
        this.color="#6a1b9a";
    }
    MaxPool2dNode.title="MaxPool2d"; LiteGraph.registerNodeType("pytorch/maxpool2d",MaxPool2dNode);

    function AvgPool2dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={kernel_size:2,stride:2,padding:0};
        this.addWidget("number","Kernel Size",2,v=>{this.properties.kernel_size=v;},{step:1,min:1});
        this.addWidget("number","Stride",2,v=>{this.properties.stride=v;},{step:1,min:1});
        this.addWidget("number","Padding",0,v=>{this.properties.padding=v;},{step:1,min:0});
        this.color="#7b1fa2";
    }
    AvgPool2dNode.title="AvgPool2d"; LiteGraph.registerNodeType("pytorch/avgpool2d",AvgPool2dNode);

    function AdaptiveAvgPool2dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={output_size:"[1, 1]"};
        this.addWidget("text","Output Size","[1, 1]",v=>{this.properties.output_size=v;});
        this.color="#7b1fa2";
    }
    AdaptiveAvgPool2dNode.title="AdaptiveAvgPool2d"; LiteGraph.registerNodeType("pytorch/adaptiveavgpool2d",AdaptiveAvgPool2dNode);

    // Normalization
    function BatchNorm2dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={num_features:16,eps:1e-5,momentum:0.1};
        this.addWidget("number","Num Features",16,v=>{this.properties.num_features=v;},{step:1,min:1});
        this.addWidget("number","Momentum",0.1,v=>{this.properties.momentum=v;},{step:0.01,min:0,max:1});
        this.color="#006064";
    }
    BatchNorm2dNode.title="BatchNorm2d"; LiteGraph.registerNodeType("pytorch/batchnorm2d",BatchNorm2dNode);

    function LayerNormNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={normalized_shape:"[16]"};
        this.addWidget("text","Norm Shape","[16]",v=>{this.properties.normalized_shape=v;});
        this.color="#006064";
    }
    LayerNormNode.title="LayerNorm"; LiteGraph.registerNodeType("pytorch/layernorm",LayerNormNode);

    function DropoutNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={p:0.5};
        this.addWidget("number","Dropout p",0.5,v=>{this.properties.p=v;},{step:0.05,min:0,max:1});
        this.color="#00695c";
    }
    DropoutNode.title="Dropout"; LiteGraph.registerNodeType("pytorch/dropout",DropoutNode);

    // Shape
    function FlattenNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={start_dim:1};
        this.addWidget("number","Start Dim",1,v=>{this.properties.start_dim=v;},{step:1,min:0});
        this.color="#455a64";
    }
    FlattenNode.title="Flatten"; LiteGraph.registerNodeType("pytorch/flatten",FlattenNode);

    function ReshapeNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={shape:"[-1, 128]"};
        this.addWidget("text","Shape","[-1, 128]",v=>{this.properties.shape=v;});
        this.color="#455a64";
    }
    ReshapeNode.title="Reshape"; LiteGraph.registerNodeType("pytorch/reshape",ReshapeNode);

    function UpsampleNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={scale_factor:2,mode:"nearest"};
        this.addWidget("number","Scale Factor",2,v=>{this.properties.scale_factor=v;},{step:0.5,min:0.5});
        this.addWidget("combo","Mode","nearest",v=>{this.properties.mode=v;},{values:["nearest","bilinear","bicubic"]});
        this.color="#37474f";
    }
    UpsampleNode.title="Upsample"; LiteGraph.registerNodeType("pytorch/upsample",UpsampleNode);

    // Recurrent
    function LSTMNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={input_size:128,hidden_size:256,num_layers:1,bidirectional:false};
        this.addWidget("number","Input Size",128,v=>{this.properties.input_size=v;},{step:1,min:1});
        this.addWidget("number","Hidden Size",256,v=>{this.properties.hidden_size=v;},{step:1,min:1});
        this.addWidget("number","Num Layers",1,v=>{this.properties.num_layers=v;},{step:1,min:1});
        this.addWidget("toggle","Bidirectional",false,v=>{this.properties.bidirectional=v;});
        this.color="#bf360c";
    }
    LSTMNode.title="LSTM"; LiteGraph.registerNodeType("pytorch/lstm",LSTMNode);

    function GRUNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={input_size:128,hidden_size:256,num_layers:1,bidirectional:false};
        this.addWidget("number","Input Size",128,v=>{this.properties.input_size=v;},{step:1,min:1});
        this.addWidget("number","Hidden Size",256,v=>{this.properties.hidden_size=v;},{step:1,min:1});
        this.addWidget("number","Num Layers",1,v=>{this.properties.num_layers=v;},{step:1,min:1});
        this.addWidget("toggle","Bidirectional",false,v=>{this.properties.bidirectional=v;});
        this.color="#e64a19";
    }
    GRUNode.title="GRU"; LiteGraph.registerNodeType("pytorch/gru",GRUNode);

    // Math Operators
    function AddNode() {
        this.addInput("A","tensor");this.addInput("B","tensor");this.addOutput("Output","tensor");
        this.properties={};
        this.color="#0d47a1";
    }
    AddNode.title="Add"; LiteGraph.registerNodeType("pytorch/add",AddNode);

    function SubNode() {
        this.addInput("A","tensor");this.addInput("B","tensor");this.addOutput("Output","tensor");
        this.properties={};
        this.color="#0d47a1";
    }
    SubNode.title="Subtract"; LiteGraph.registerNodeType("pytorch/sub",SubNode);

    function MulNode() {
        this.addInput("A","tensor");this.addInput("B","tensor");this.addOutput("Output","tensor");
        this.properties={};
        this.color="#0d47a1";
    }
    MulNode.title="Multiply"; LiteGraph.registerNodeType("pytorch/mul",MulNode);

    function DivNode() {
        this.addInput("A","tensor");this.addInput("B","tensor");this.addOutput("Output","tensor");
        this.properties={};
        this.color="#0d47a1";
    }
    DivNode.title="Divide"; LiteGraph.registerNodeType("pytorch/div",DivNode);

    function MatMulNode() {
        this.addInput("A","tensor");this.addInput("B","tensor");this.addOutput("Output","tensor");
        this.properties={};
        this.color="#1565c0";
    }
    MatMulNode.title="MatMul"; LiteGraph.registerNodeType("pytorch/matmul",MatMulNode);

    // Generic fallback
    function GenericNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={op_type:"Unknown",name:""};
        this.color="#880e4f";
    }
    GenericNode.title="Generic Node"; LiteGraph.registerNodeType("pytorch/generic",GenericNode);

    // New pooling, norm, activation, and utility nodes
    function GlobalAvgPool2dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={};
        this.color="#7b1fa2";
    }
    GlobalAvgPool2dNode.title="GlobalAvgPool2d"; LiteGraph.registerNodeType("pytorch/globalavgpool2d",GlobalAvgPool2dNode);

    function MaxPool1dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={kernel_size:2,stride:2,padding:0};
        this.addWidget("number","Kernel Size",2,v=>{this.properties.kernel_size=v;},{step:1,min:1});
        this.addWidget("number","Stride",2,v=>{this.properties.stride=v;},{step:1,min:1});
        this.addWidget("number","Padding",0,v=>{this.properties.padding=v;},{step:1,min:0});
        this.color="#6a1b9a";
    }
    MaxPool1dNode.title="MaxPool1d"; LiteGraph.registerNodeType("pytorch/maxpool1d",MaxPool1dNode);

    function AvgPool1dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={kernel_size:2,stride:2,padding:0};
        this.addWidget("number","Kernel Size",2,v=>{this.properties.kernel_size=v;},{step:1,min:1});
        this.addWidget("number","Stride",2,v=>{this.properties.stride=v;},{step:1,min:1});
        this.addWidget("number","Padding",0,v=>{this.properties.padding=v;},{step:1,min:0});
        this.color="#7b1fa2";
    }
    AvgPool1dNode.title="AvgPool1d"; LiteGraph.registerNodeType("pytorch/avgpool1d",AvgPool1dNode);

    function BatchNorm1dNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={num_features:16,eps:1e-5,momentum:0.1};
        this.addWidget("number","Num Features",16,v=>{this.properties.num_features=v;},{step:1,min:1});
        this.addWidget("number","Momentum",0.1,v=>{this.properties.momentum=v;},{step:0.01,min:0,max:1});
        this.color="#006064";
    }
    BatchNorm1dNode.title="BatchNorm1d"; LiteGraph.registerNodeType("pytorch/batchnorm1d",BatchNorm1dNode);

    function PReLUNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={num_parameters:1};
        this.addWidget("number","Num Parameters",1,v=>{this.properties.num_parameters=v;},{step:1,min:1});
        this.color="#9d174d";
    }
    PReLUNode.title="PReLU"; LiteGraph.registerNodeType("pytorch/prelu",PReLUNode);

    function ConcatNode() {
        this.addInput("A","tensor");this.addInput("B","tensor");this.addOutput("Output","tensor");
        this.properties={dim:1};
        this.addWidget("number","Dim",1,v=>{this.properties.dim=v;},{step:1});
        this.color="#37474f";
    }
    ConcatNode.title="Concat"; LiteGraph.registerNodeType("pytorch/concat",ConcatNode);

    // Split node splits Input tensor to Output1 and Output2
    function SplitNode() {
        this.addInput("Input","tensor");this.addOutput("Output1","tensor");this.addOutput("Output2","tensor");
        this.properties={split_size:1,dim:1};
        this.addWidget("number","Split Size",1,v=>{this.properties.split_size=v;},{step:1,min:1});
        this.addWidget("number","Dim",1,v=>{this.properties.dim=v;},{step:1});
        this.color="#455a64";
    }
    SplitNode.title="Split"; LiteGraph.registerNodeType("pytorch/split",SplitNode);

    function TransposeNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={dim0:0,dim1:1};
        this.addWidget("number","Dim 0",0,v=>{this.properties.dim0=v;},{step:1});
        this.addWidget("number","Dim 1",1,v=>{this.properties.dim1=v;},{step:1});
        this.color="#37474f";
    }
    TransposeNode.title="Transpose"; LiteGraph.registerNodeType("pytorch/transpose",TransposeNode);

    function SqueezeNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={dim:0};
        this.addWidget("number","Dim",0,v=>{this.properties.dim=v;},{step:1});
        this.color="#455a64";
    }
    SqueezeNode.title="Squeeze"; LiteGraph.registerNodeType("pytorch/squeeze",SqueezeNode);

    function UnsqueezeNode() {
        this.addInput("Input","tensor");this.addOutput("Output","tensor");
        this.properties={dim:0};
        this.addWidget("number","Dim",0,v=>{this.properties.dim=v;},{step:1});
        this.color="#455a64";
    }
    UnsqueezeNode.title="Unsqueeze"; LiteGraph.registerNodeType("pytorch/unsqueeze",UnsqueezeNode);

    // ── INITIAL GRAPH LOAD ────────────────────────────────────
    let lastSavedData = "";
    const savedData = localStorage.getItem("hexforge-autosave");
    let loaded = false;
    if (savedData) {
        try {
            graph.configure(JSON.parse(savedData));
            if (graph._nodes) {
                graph._nodes.forEach(n => {
                    n.title = getBaseTitle(n);
                    n.originalTitle = n.title;
                    refreshStatusWidgets(n);
                });
            }
            syncAllNodeWidgets(graph);
            lastSavedData = savedData;
            loaded = true;
        } catch (e) {
            console.error("Failed to parse saved graph from localStorage:", e);
        }
    }
    
    if (!loaded) {
        // ── DEMO GRAPH fallback ───────────────────────────────
        var n_in  = LiteGraph.createNode("pytorch/input");    n_in.pos=[60,200];  graph.add(n_in);
        var n_c2  = LiteGraph.createNode("pytorch/conv2d");   n_c2.pos=[310,200]; graph.add(n_c2);
        var n_bn  = LiteGraph.createNode("pytorch/batchnorm2d"); n_bn.pos=[570,200]; graph.add(n_bn);
        var n_rl  = LiteGraph.createNode("pytorch/relu");     n_rl.pos=[810,200]; graph.add(n_rl);
        n_in.connect(0,n_c2,0); n_c2.connect(0,n_bn,0); n_bn.connect(0,n_rl,0);
        lastSavedData = JSON.stringify(graph.serialize());
    }

    graph.start();

    // ── DEBOUNCED / PERIODIC AUTOSAVE SYSTEM ──────────────────
    let saveTimeout = null;
    function triggerAutosave(immediate = false) {
        const indicator = document.getElementById("autosave-indicator");
        if (indicator) {
            indicator.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin" style="color:var(--yellow);font-size:10px;"></i> Saving...';
        }
        
        if (saveTimeout) clearTimeout(saveTimeout);
        
        const saveFunc = () => {
            const currentData = JSON.stringify(graph.serialize());
            if (currentData !== lastSavedData) {
                localStorage.setItem("hexforge-autosave", currentData);
                lastSavedData = currentData;
            }
            if (indicator) {
                indicator.innerHTML = '<i class="fa-solid fa-circle-check" style="color:var(--green);font-size:10px;"></i> Autosaved';
            }
        };

        if (immediate) {
            saveFunc();
        } else {
            saveTimeout = setTimeout(saveFunc, 1000);
        }
    }
    window.TriggerAutosave = triggerAutosave;

    // Periodic check (every 3 seconds) for positions, properties, or connection changes
    setInterval(() => {
        const currentData = JSON.stringify(graph.serialize());
        if (currentData !== lastSavedData) {
            triggerAutosave(false);
        }
    }, 3000);

    window.addEventListener("resize", ()=>canvas.resize(container.clientWidth, container.clientHeight));

    window.CurrentLayoutDirection = 'horizontal';

    // Sidebar chips → add node
    document.querySelectorAll('.node-btn').forEach(btn=>{
        btn.addEventListener('click', e=>{
            e.preventDefault();
            const t = btn.getAttribute('data-type');
            if(t){
                var node = LiteGraph.createNode(t);
                if(node){
                    node.horizontal = (window.CurrentLayoutDirection === 'vertical');
                    var cx=-canvas.ds.offset[0]+(canvas.canvas.width/2)/canvas.ds.scale;
                    var cy=-canvas.ds.offset[1]+(canvas.canvas.height/2)/canvas.ds.scale;
                    node.pos=[cx+(Math.random()*80-40), cy+(Math.random()*80-40)];
                    graph.add(node);
                    graph.setDirtyCanvas(true,true);
                    openPanel(node); // auto-open panel for the new node
                }
            }
        });
    });

    // Custom LiteGraph configurations to match HexFlow premium style
    function applyLiteGraphTheme(theme) {
        const isLight = theme === "light";
        
        // General LiteGraph configs
        LiteGraph.NODE_DEFAULT_BGCOLOR = isLight ? "#f8f9fa" : "#1e1e1e";
        LiteGraph.NODE_DEFAULT_BOXCOLOR = isLight ? "#dee2e6" : "#666666";
        LiteGraph.NODE_TEXT_COLOR = isLight ? "#212529" : "#ffffff";
        LiteGraph.NODE_SUBTEXT_COLOR = isLight ? "#6b7280" : "#94a3b8";
        LiteGraph.NODE_TITLE_COLOR = isLight ? "#212529" : "#ffffff";
        LiteGraph.NODE_SELECTED_TITLE_COLOR = isLight ? "#0066cc" : "#33ccff";
        
        if (canvas) {
            canvas.background_color = "transparent";
            canvas.clear_background_color = isLight ? "#ffffff" : "#121212";
            canvas.grid_color = isLight ? "#e9ecef" : "#2e2e2e";
            canvas.draw(true, true);
        }
    }

    document.addEventListener("themeChanged", (e) => {
        applyLiteGraphTheme(e.detail.theme);
    });

    // Set initial theme
    const initialTheme = localStorage.getItem('app-theme') || 'dark';
    applyLiteGraphTheme(initialTheme);

    window.AppGraph = graph;
    window.OpenNodePanel = openPanel;
    document.dispatchEvent(new Event("editorLoaded"));
    if (window.TriggerShapeInference) {
        window.TriggerShapeInference();
    }
});
