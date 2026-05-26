// We wait for the 'editorLoaded' custom event instead of just DOMContentLoaded
// this guarantees editor.js has finished setting up LiteGraph
document.addEventListener("editorLoaded", () => {
    console.log("API logic initialized after LiteGraph.");
    const btnProfile = document.getElementById("btn-profile");

    // Helper function to serialize the LiteGraph to a JSON array of nodes
    function serializeGraph() {
        if (!window.AppGraph) return null;

        const rawData = window.AppGraph.serialize();
        const nodes = rawData.nodes;
        const links = rawData.links;

        // Build a mapping of link IDs to target node IDs
        const linkMap = {};
        if (links) {
            links.forEach(link => {
                // link format: [id, origin_id, origin_slot, target_id, target_slot, type]
                const linkId = link[0];
                const targetNodeId = link[3];
                linkMap[linkId] = targetNodeId;
            });
        }

        // Process nodes into a clean list for the backend
        const cleanNodes = nodes.map(n => {
            // Find which nodes this node connects to
            const children = [];
            if (n.outputs) {
                n.outputs.forEach(opt => {
                    if (opt.links) {
                        opt.links.forEach(l_id => {
                            if (linkMap[l_id]) {
                                children.push(linkMap[l_id]);
                            }
                        });
                    }
                });
            }

            return {
                id: n.id,
                type: n.type.replace('pytorch/', ''), // e.g. "conv2d"
                properties: n.properties,
                children: children
            };
        });

        return cleanNodes;
    }
    window.SerializeGraph = serializeGraph;

    let shapeInferTimeout = null;
    function triggerShapeInference() {
        if (shapeInferTimeout) clearTimeout(shapeInferTimeout);
        shapeInferTimeout = setTimeout(() => {
            if (window.SerializeGraph && window.AppGraph) {
                const graphData = window.SerializeGraph();
                if (!graphData || graphData.length === 0) return;

                // Show loading indicator in the right-side panel for the active node
                const activeNode = window.GetActiveNode ? window.GetActiveNode() : null;
                if (activeNode && window.UpdateNodePanelShapes) {
                    window.UpdateNodePanelShapes(activeNode, true);
                }

                fetch('/api/infer-shapes', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ graph: graphData })
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => { throw err; });
                    }
                    return response.json();
                })
                .then(data => {
                    // Update shapes on nodes
                    if (window.AppGraph._nodes) {
                        window.AppGraph._nodes.forEach(node => {
                            if (node.type === "pytorch/input") {
                                // Keep input node shape
                                return;
                            }
                            
                            const res = data.results.find(r => Number(r.node_id) === node.id);
                            if (res) {
                                node.properties.output_shape = res.shape ? `[${res.shape.join(', ')}]` : '';
                                node.properties.input_shape = res.input_shape ? `[${res.input_shape.join(', ')}]` : '';
                            } else {
                                delete node.properties.output_shape;
                                delete node.properties.input_shape;
                            }
                            
                            if (window.RefreshStatusWidgets) {
                                window.RefreshStatusWidgets(node);
                            }
                        });
                    }
                    
                    // Clear red colors from all links
                    if (window.AppGraph.links) {
                        for (let linkId in window.AppGraph.links) {
                            window.AppGraph.links[linkId].color = null;
                        }
                    }
                    
                    // Update active node panel shapes
                    const currentActive = window.GetActiveNode ? window.GetActiveNode() : null;
                    if (currentActive && window.UpdateNodePanelShapes) {
                        window.UpdateNodePanelShapes(currentActive, false);
                    }
                    
                    // Force canvas redraw
                    if (LiteGraph.LGraphCanvas.active_canvas) {
                        LiteGraph.LGraphCanvas.active_canvas.setDirty(true, true);
                    }
                })
                .catch(err => {
                    console.warn("Shape inference failed:", err);
                    
                    // Update shapes on nodes with whatever results we got
                    const results = err.results || [];
                    if (window.AppGraph._nodes) {
                        window.AppGraph._nodes.forEach(node => {
                            if (node.type === "pytorch/input") {
                                return;
                            }
                            
                            const res = results.find(r => Number(r.node_id) === node.id);
                            if (res) {
                                node.properties.output_shape = res.shape ? `[${res.shape.join(', ')}]` : '';
                                node.properties.input_shape = res.input_shape ? `[${res.input_shape.join(', ')}]` : '';
                            } else {
                                delete node.properties.output_shape;
                                delete node.properties.input_shape;
                            }
                            
                            if (window.RefreshStatusWidgets) {
                                window.RefreshStatusWidgets(node);
                            }
                        });
                    }
                    
                    // Clear red colors from all links first
                    if (window.AppGraph.links) {
                        for (let linkId in window.AppGraph.links) {
                            window.AppGraph.links[linkId].color = null;
                        }
                    }
                    
                    // Highlight the connection to the failing node in red
                    if (err.error_node_id && window.AppGraph) {
                        const errorNode = window.AppGraph.getNodeById(Number(err.error_node_id));
                        if (errorNode && errorNode.inputs) {
                            errorNode.inputs.forEach(input => {
                                if (input.link !== null) {
                                    const link = window.AppGraph.links[input.link];
                                    if (link) {
                                        link.color = "#ff3333";
                                    }
                                }
                            });
                        }
                    }
                    
                    // Update active node panel shapes
                    const currentActive = window.GetActiveNode ? window.GetActiveNode() : null;
                    if (currentActive && window.UpdateNodePanelShapes) {
                        window.UpdateNodePanelShapes(currentActive, false);
                    }
                    
                    // Force canvas redraw
                    if (LiteGraph.LGraphCanvas.active_canvas) {
                        LiteGraph.LGraphCanvas.active_canvas.setDirty(true, true);
                    }
                });
            }
        }, 300);
    }
    window.TriggerShapeInference = triggerShapeInference;

    function getBaseTitle(node) {
        if (window.GetBaseTitle) {
            return window.GetBaseTitle(node);
        }
        let title = node.title || "";
        title = title.replace(/\s*\([^)]*ms\)(?:\s*\|\s*Out:\s*(?:\[[^\]]*\])?)?/g, "");
        title = title.replace(/\s*\(⚠️ Error\)/g, "");
        return title.trim();
    }

    if (btnProfile) {
        btnProfile.addEventListener("click", () => {
            if (window.CloseAllDropdowns) window.CloseAllDropdowns();

            // Clear previous error/profiling indicators from all nodes
            if (window.AppGraph && window.AppGraph._nodes) {
                window.AppGraph._nodes.forEach(n => {
                    n.has_error = false;
                    delete n.onDrawBackground;
                    n.boxcolor = null;
                    n.color = null;
                    const cleaned = getBaseTitle(n);
                    n.title = cleaned;
                    n.originalTitle = cleaned;
                    if (n.properties) {
                        delete n.properties.latency;
                    }
                    if (window.RefreshStatusWidgets) {
                        window.RefreshStatusWidgets(n);
                    }
                });
                if (window.AppGraph.links) {
                    for (let linkId in window.AppGraph.links) {
                        window.AppGraph.links[linkId].color = null;
                    }
                }
                if (LiteGraph.LGraphCanvas.active_canvas) {
                    LiteGraph.LGraphCanvas.active_canvas.setDirty(true, true);
                }
            }

            const graphData = serializeGraph();
            console.log("Sending graph data to backend...", graphData);

            // Update UI to show loading
            btnProfile.setAttribute("aria-busy", "true");
            btnProfile.textContent = "Profiling...";

            fetch('/api/profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ graph: graphData })
            })
                .then(response => response.json())
                .then(data => {
                    console.log("Profile Results:", data);
                    if (data.status === "success" && window.AppGraph) {
                        // Find max duration to scale colors
                        const maxTime = Math.max(...data.results.map(r => r.duration_ms), 0.001);

                        data.results.forEach(res => {
                            const node = window.AppGraph.getNodeById(Number(res.node_id));
                            if (node) {
                                // Reset error status
                                node.has_error = false;
                                delete node.onDrawBackground;
                                node.boxcolor = null;

                                // Save shape
                                const shapeStr = res.shape ? `[${res.shape.join(', ')}]` : '';
                                node.properties.output_shape = shapeStr;
                                const inShapeStr = res.input_shape ? `[${res.input_shape.join(', ')}]` : '';
                                node.properties.input_shape = inShapeStr;
                                node.properties.latency = `${res.duration_ms} ms`;

                                // Clean name first
                                const baseTitle = getBaseTitle(node);
                                node.originalTitle = baseTitle;
                                node.title = baseTitle;

                                if (window.RefreshStatusWidgets) {
                                    window.RefreshStatusWidgets(node);
                                }

                                // Color intensity based on relative slowness (red scale)
                                const ratio = res.duration_ms / maxTime;
                                const r = Math.floor(50 + (205 * ratio)); // 50 to 255
                                node.color = `rgb(${r}, 50, 50)`;
                            }
                        });
                        // Force canvas redraw
                        if (LiteGraph.LGraphCanvas.active_canvas) {
                            LiteGraph.LGraphCanvas.active_canvas.setDirty(true, true);
                        }
                        window.customAlert("Profiling complete! Slowest nodes highlighted in red, output shapes traced.", "success", "Profiling Complete");
                    } else {
                        // Highlight the failing node if error_node_id is provided
                        if (data.error_node_id && window.AppGraph) {
                            const node = window.AppGraph.getNodeById(Number(data.error_node_id));
                            if (node) {
                                const baseTitle = getBaseTitle(node);
                                node.originalTitle = baseTitle;
                                node.title = `${baseTitle} (⚠️ Error)`;
                                node.color = "#ff6b6b";
                                node.boxcolor = "#ff6b6b";
                                
                                // Glowing red border outline around the node
                                node.has_error = true;
                                node.onDrawBackground = function(ctx) {
                                    ctx.save();
                                    ctx.shadowColor = "#ff3333";
                                    ctx.shadowBlur = 25;
                                    ctx.strokeStyle = "#ff6b6b";
                                    ctx.lineWidth = 4;
                                    const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 20;
                                    ctx.strokeRect(-2, -titleHeight - 2, this.size[0] + 4, this.size[1] + titleHeight + 4);
                                    ctx.restore();
                                };

                                if (LiteGraph.LGraphCanvas.active_canvas) {
                                    LiteGraph.LGraphCanvas.active_canvas.setDirty(true, true);
                                }
                                if (window.OpenNodePanel) {
                                    window.OpenNodePanel(node);
                                }
                            }
                        }
                        window.customAlert(data.message, "error", "Profiling Error");
                    }
                })
                .catch(err => {
                    console.error("Error profiling model:", err);
                    window.customAlert("An error occurred during profiling.", "error", "Error");
                })
                .finally(() => {
                    btnProfile.removeAttribute("aria-busy");
                    btnProfile.textContent = "Profile Node-Wise Speed";
                });
        });
    }

    const btnEvaluate = document.getElementById("btn-evaluate");
    if (btnEvaluate) {
        btnEvaluate.addEventListener("click", () => {
            if (window.CloseAllDropdowns) window.CloseAllDropdowns();
            const graphData = serializeGraph();
            if (!graphData) return window.customAlert("Graph is empty!", "warning", "Warning");

            const fileInput = document.getElementById('dataset-upload');
            if (!fileInput.files.length) {
                return window.customAlert("Please upload a .zip dataset first before evaluating!", "warning", "Dataset Missing");
            }

            const formData = new FormData();
            formData.append('graph', JSON.stringify(graphData));
            formData.append('dataset', fileInput.files[0]);

            btnEvaluate.setAttribute("aria-busy", "true");
            btnEvaluate.textContent = "Evaluating...";

            fetch('/api/evaluate', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === "success") {
                        const res = data.results;
                        const overlay = document.getElementById('result-overlay');
                        const body = document.getElementById('result-body');
                        if (overlay && body) {
                            body.innerHTML = `
                                <div class="result-metric">
                                    <span class="metric-label">🎯 Accuracy</span>
                                    <span class="metric-value highlight">${res.accuracy}%</span>
                                </div>
                                <div class="result-metric">
                                    <span class="metric-label">🖼️ Total Images</span>
                                    <span class="metric-value">${res.total_images}</span>
                                </div>
                                <div class="result-metric">
                                    <span class="metric-label">✅ Correct Predictions</span>
                                    <span class="metric-value">${res.correct_predictions}</span>
                                </div>
                                <div class="result-metric">
                                    <span class="metric-label">⚡ Throughput</span>
                                    <span class="metric-value highlight">${res.fps} img/sec</span>
                                </div>
                                <div class="result-metric">
                                    <span class="metric-label">⏱️ Total Time</span>
                                    <span class="metric-value">${res.total_time_seconds}s</span>
                                </div>
                                <div class="result-metric">
                                    <span class="metric-label">📂 Classes</span>
                                    <span class="metric-value" style="font-size:12px;">${res.classes.join(", ")}</span>
                                </div>
                            `;
                            overlay.classList.add('visible');
                        }
                    } else {
                        window.customAlert(data.message, "error", "Evaluation Error");
                    }
                })
                .catch(err => {
                    console.error("Evaluation failed:", err);
                    window.customAlert("An error occurred during evaluation.", "error", "Error");
                })
                .finally(() => {
                    btnEvaluate.removeAttribute("aria-busy");
                    btnEvaluate.textContent = "🧪 Evaluate Accuracy";
                });
        });
    }

    // ONNX Upload and Inspection
    const onnxUploadInput = document.getElementById("onnx-upload");
    const btnInspectOnnx = document.getElementById("btn-inspect-onnx");
    const onnxFilename = document.getElementById("onnx-filename");
    const onnxUploadZone = document.getElementById("onnx-upload-zone");
    
    let lastParsedONNX = null;

    if (onnxUploadInput) {
        onnxUploadInput.addEventListener("change", function () {
            if (this.files.length) {
                const file = this.files[0];
                onnxFilename.textContent = file.name;
                onnxFilename.style.display = 'block';
                onnxUploadZone.style.borderColor = 'var(--accent)';
                onnxUploadZone.style.background = 'var(--accent-light)';
                btnInspectOnnx.removeAttribute("disabled");
            } else {
                onnxFilename.style.display = 'none';
                onnxUploadZone.style.borderColor = '';
                onnxUploadZone.style.background = '';
                btnInspectOnnx.setAttribute("disabled", "true");
            }
        });
    }

    if (btnInspectOnnx) {
        btnInspectOnnx.addEventListener("click", () => {
            if (window.CloseAllDropdowns) window.CloseAllDropdowns();
            if (!onnxUploadInput.files.length) return window.customAlert("Please upload an ONNX model first.", "warning", "ONNX Missing");

            const file = onnxUploadInput.files[0];
            const formData = new FormData();
            formData.append("onnx_file", file);

            btnInspectOnnx.setAttribute("aria-busy", "true");
            btnInspectOnnx.textContent = "Analyzing ONNX...";

            fetch("/api/inspect-onnx", {
                method: "POST",
                body: formData
            })
            .then(res => res.json())
            .then(res => {
                if (res.status === "success") {
                    lastParsedONNX = res.data;
                    window.LastParsedONNX = res.data; // Store globally
                    
                    // Render ONNX overlay
                    showONNXInspectorModal(res.data);
                } else {
                    window.customAlert(res.message, "error", "Inspection Error");
                }
            })
            .catch(err => {
                console.error("ONNX Parsing Error:", err);
                window.customAlert("An error occurred during ONNX parsing.", "error", "Error");
            })
            .finally(() => {
                btnInspectOnnx.removeAttribute("aria-busy");
                btnInspectOnnx.textContent = "🔍 Inspect ONNX Model";
            });
        });
    }

    function showONNXInspectorModal(data) {
        const overlay = document.getElementById("onnx-overlay");
        if (!overlay) return;

        // Populate metadata
        const metaGrid = document.getElementById("onnx-meta-grid");
        const info = data.model_info;
        metaGrid.innerHTML = `
            <div class="onnx-meta-item">
                <span class="onnx-meta-label">Producer</span>
                <span class="onnx-meta-value">${info.producer_name || 'Unknown'} (${info.producer_version || '?'})</span>
            </div>
            <div class="onnx-meta-item">
                <span class="onnx-meta-label">IR Version</span>
                <span class="onnx-meta-value">v${info.ir_version}</span>
            </div>
            <div class="onnx-meta-item">
                <span class="onnx-meta-label">Model Version</span>
                <span class="onnx-meta-value">${info.model_version || '0'}</span>
            </div>
            <div class="onnx-meta-item">
                <span class="onnx-meta-label">Total Nodes</span>
                <span class="onnx-meta-value">${data.nodes.length}</span>
            </div>
        `;

        // Populate inputs
        const inputsList = document.getElementById("onnx-inputs-list");
        inputsList.innerHTML = data.inputs.map(inp => {
            const shapeStr = inp.shape ? `[${inp.shape.join(', ')}]` : 'dynamic';
            return `<span class="onnx-io-badge">📥 ${inp.name} : ${shapeStr}</span>`;
        }).join('') || '<span style="font-size:12px; color:var(--text-muted);">None</span>';

        // Populate outputs
        const outputsList = document.getElementById("onnx-outputs-list");
        outputsList.innerHTML = data.outputs.map(out => {
            const shapeStr = out.shape ? `[${out.shape.join(', ')}]` : 'dynamic';
            return `<span class="onnx-io-badge output-badge">📤 ${out.name} : ${shapeStr}</span>`;
        }).join('') || '<span style="font-size:12px; color:var(--text-muted);">None</span>';

        // Render nodes
        renderONNXNodes(data.nodes);

        overlay.classList.add("visible");
    }

    function renderONNXNodes(nodes, filterText = "") {
        const nodesList = document.getElementById("onnx-nodes-list");
        if (!nodesList) return;

        const term = filterText.toLowerCase().trim();
        const filtered = nodes.filter(n => 
            n.name.toLowerCase().includes(term) || 
            n.op_type.toLowerCase().includes(term)
        );

        nodesList.innerHTML = filtered.map(n => {
            // Determine op class for styling
            const opLower = n.op_type.toLowerCase();
            let opClass = 'op-other';
            if (opLower.startsWith('conv'))                      opClass = 'op-conv';
            else if (['relu','leakyrelu','sigmoid','tanh','gelu','silu','elu','softmax'].includes(opLower)) opClass = 'op-relu';
            else if (opLower.includes('pool'))                   opClass = 'op-pool';
            else if (['gemm','matmul','linear'].includes(opLower)) opClass = 'op-linear';
            else if (['flatten','reshape','squeeze','unsqueeze'].includes(opLower)) opClass = 'op-flatten';
            else if (['batchnormalization','layernorm','dropout','instancenorm'].includes(opLower)) opClass = 'op-norm';

            // Attributes list
            const attrs = Object.entries(n.attributes).map(([key, val]) => {
                let valStr = JSON.stringify(val);
                if (valStr.length > 25) valStr = valStr.substring(0, 22) + '...';
                return `<span class="onnx-attr-badge">${key}: ${valStr}</span>`;
            }).join('');

            // Shape output text
            const outputShapes = Object.entries(n.output_shapes).map(([outName, shape]) => {
                return `${outName} [${shape.join(', ')}]`;
            }).join(', ');
            
            return `
                <div class="onnx-node-row">
                    <div class="onnx-node-header">
                        <span class="onnx-node-name">${n.name}</span>
                        <span class="onnx-node-op ${opClass}">${n.op_type}</span>
                    </div>
                    <div class="onnx-node-details">
                        <div class="onnx-node-io">
                            <div class="onnx-io-line">
                                <span class="onnx-io-label">Inputs:</span>
                                <span class="onnx-io-values">${n.inputs.join(', ')}</span>
                            </div>
                            <div class="onnx-io-line">
                                <span class="onnx-io-label">Outputs:</span>
                                <span class="onnx-io-values">${outputShapes || n.outputs.join(', ')}</span>
                            </div>
                        </div>
                        ${attrs ? `<div class="onnx-node-attrs">${attrs}</div>` : ''}
                    </div>
                </div>
            `;
        }).join('') || '<div style="padding:20px; text-align:center; color:var(--text-muted);">No nodes found.</div>';
    }

    // Search bar listener
    const searchInput = document.getElementById("onnx-search");
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            if (window.LastParsedONNX) {
                renderONNXNodes(window.LastParsedONNX.nodes, e.target.value);
            }
        });
    }
    
    const btnLoadCanvas = document.getElementById("btn-onnx-load-canvas");
    
    function performLoadONNX(data) {
        window.AppGraph.clear();
        const tensorProducer = {};
        data.inputs.forEach((inp, idx) => {
            const node = LiteGraph.createNode("pytorch/input");
            if (node) {
                node.title = inp.name;
                const shapeStr = inp.shape ? `[${inp.shape.join(', ')}]` : "[1, 3, 224, 224]";
                node.properties.shape = shapeStr;
                if (node.widgets && node.widgets[0]) {
                    node.widgets[0].value = shapeStr;
                }
                node.pos = [80, 150 + idx * 120];
                window.AppGraph.add(node);
                tensorProducer[inp.name] = { node: node, slot: 0 };
            }
        });

        const constantValues = {};
        data.nodes.forEach(n => {
            if (n.op_type === "Constant") {
                const outName = n.outputs[0];
                if (outName && n.attributes && n.attributes.value !== undefined) {
                    constantValues[outName] = n.attributes.value;
                }
            }
        });

        const getInputShape = (tensorName) => {
            const modelInput = data.inputs.find(i => i.name === tensorName);
            if (modelInput && modelInput.shape) return modelInput.shape;
            const producingNode = data.nodes.find(node => node.outputs.includes(tensorName));
            if (producingNode && producingNode.output_shapes && producingNode.output_shapes[tensorName]) {
                return producingNode.output_shapes[tensorName];
            }
            return null;
        };

        const nonConstantNodes = data.nodes.filter(n => n.op_type !== "Constant");
        const columns = {};

        nonConstantNodes.forEach((n, idx) => {
            let lgType = "pytorch/generic";
            const op = n.op_type.toLowerCase();

            if (op === "conv")                          lgType = "pytorch/conv2d";
            else if (op === "convtranspose")            lgType = "pytorch/convtranspose2d";
            else if (op === "relu")                     lgType = "pytorch/relu";
            else if (op === "leakyrelu")                lgType = "pytorch/leakyrelu";
            else if (op === "sigmoid")                  lgType = "pytorch/sigmoid";
            else if (op === "tanh")                     lgType = "pytorch/tanh";
            else if (op === "gelu")                     lgType = "pytorch/gelu";
            else if (op === "elu")                      lgType = "pytorch/elu";
            else if (op === "softmax")                  lgType = "pytorch/softmax";
            else if (op === "maxpool")                  lgType = "pytorch/maxpool2d";
            else if (op === "averagepool")              lgType = "pytorch/avgpool2d";
            else if (op === "globalaveragepool")        lgType = "pytorch/globalavgpool2d";
            else if (op === "flatten")                  lgType = "pytorch/flatten";
            else if (op === "reshape")                  lgType = "pytorch/reshape";
            else if (op === "gemm" || op === "matmul") lgType = "pytorch/linear";
            else if (op === "batchnormalization")       lgType = "pytorch/batchnorm2d";
            else if (op === "dropout")                  lgType = "pytorch/dropout";
            else if (op === "lstm")                     lgType = "pytorch/lstm";
            else if (op === "gru")                      lgType = "pytorch/gru";
            else if (op === "concat")                   lgType = "pytorch/concat";
            else if (op === "split")                    lgType = "pytorch/split";
            else if (op === "transpose")                lgType = "pytorch/transpose";
            else if (op === "squeeze")                  lgType = "pytorch/squeeze";
            else if (op === "unsqueeze")                lgType = "pytorch/unsqueeze";
            else if (op === "prelu")                    lgType = "pytorch/prelu";

            const node = LiteGraph.createNode(lgType);
            if (!node) return;

            node.title = n.name;
            
            // Set input & output shape properties for visual feedback
            const activeInputs = n.inputs.filter(inName => constantValues[inName] === undefined);
            if (activeInputs.length > 0) {
                const inShape = getInputShape(activeInputs[0]);
                if (inShape) {
                    if (!node.properties) node.properties = {};
                    node.properties.input_shape = `[${inShape.join(', ')}]`;
                }
            }
            if (n.outputs && n.outputs.length && n.output_shapes[n.outputs[0]]) {
                if (!node.properties) node.properties = {};
                node.properties.output_shape = `[${n.output_shapes[n.outputs[0]].join(', ')}]`;
            }

            if (lgType === "pytorch/conv2d") {
                if (n.attributes.strides && n.attributes.strides.length) {
                    node.properties.stride = n.attributes.strides[0];
                }
                if (n.attributes.pads && n.attributes.pads.length) {
                    node.properties.padding = n.attributes.pads[0];
                }
                if (n.attributes.kernel_shape && n.attributes.kernel_shape.length) {
                    node.properties.kernel_size = n.attributes.kernel_shape[0];
                }
                const outTensorName = n.outputs[0];
                const outShape = n.output_shapes[outTensorName];
                if (outShape && outShape.length >= 2) {
                    node.properties.out_channels = outShape[1];
                }
                const inTensorName = n.inputs[0];
                const parent = tensorProducer[inTensorName];
                if (parent) {
                    const parentShape = data.inputs.find(i => i.name === inTensorName)?.shape || 
                                        data.nodes.find(node => node.outputs.includes(inTensorName))?.output_shapes?.[inTensorName];
                    if (parentShape && parentShape.length >= 2) {
                        node.properties.in_channels = parentShape[1];
                    }
                }
                if (node.widgets) {
                    node.widgets.forEach(w => {
                        if (w.name === "In Channels") w.value = node.properties.in_channels;
                        else if (w.name === "Out Channels") w.value = node.properties.out_channels;
                        else if (w.name === "Kernel Size") w.value = node.properties.kernel_size;
                        else if (w.name === "Stride") w.value = node.properties.stride;
                        else if (w.name === "Padding") w.value = node.properties.padding;
                    });
                }

            } else if (lgType === "pytorch/linear") {
                const outTensorName = n.outputs[0];
                const outShape = n.output_shapes[outTensorName];
                if (outShape && outShape.length >= 1) {
                    node.properties.out_features = outShape[outShape.length - 1];
                }
                const inTensorName = n.inputs[0];
                const parentShape = data.inputs.find(i => i.name === inTensorName)?.shape || 
                                    data.nodes.find(node => node.outputs.includes(inTensorName))?.output_shapes?.[inTensorName];
                if (parentShape && parentShape.length >= 1) {
                    node.properties.in_features = parentShape[parentShape.length - 1];
                }
                if (node.widgets) {
                    node.widgets.forEach(w => {
                        if (w.name === "In Features") w.value = node.properties.in_features;
                        else if (w.name === "Out Features") w.value = node.properties.out_features;
                    });
                }

            } else if (lgType === "pytorch/maxpool2d") {
                if (n.attributes.kernel_shape && n.attributes.kernel_shape.length) {
                    node.properties.kernel_size = n.attributes.kernel_shape[0];
                }
                if (n.attributes.strides && n.attributes.strides.length) {
                    node.properties.stride = n.attributes.strides[0];
                }
                if (node.widgets) {
                    node.widgets.forEach(w => {
                        if (w.name === "Kernel Size") w.value = node.properties.kernel_size;
                        else if (w.name === "Stride") w.value = node.properties.stride;
                    });
                }

            } else if (lgType === "pytorch/flatten") {
                if (n.attributes.axis !== undefined) {
                    node.properties.start_dim = n.attributes.axis;
                }
                if (node.widgets && node.widgets[0]) {
                    node.widgets[0].value = node.properties.start_dim;
                }

            } else if (lgType === "pytorch/concat") {
                if (n.attributes.axis !== undefined) {
                    node.properties.dim = n.attributes.axis;
                }
                if (node.widgets && node.widgets[0]) {
                    node.widgets[0].value = node.properties.dim;
                }

            } else if (lgType === "pytorch/split") {
                if (n.attributes.axis !== undefined) {
                    node.properties.dim = n.attributes.axis;
                }
                if (n.attributes.split !== undefined && n.attributes.split.length) {
                    node.properties.split_size = n.attributes.split[0];
                }
                if (node.widgets) {
                    node.widgets.forEach(w => {
                        if (w.name === "Split Size") w.value = node.properties.split_size;
                        else if (w.name === "Dim") w.value = node.properties.dim;
                    });
                }

            } else if (lgType === "pytorch/transpose") {
                if (n.attributes.perm !== undefined && n.attributes.perm.length >= 2) {
                    node.properties.dim0 = n.attributes.perm[0];
                    node.properties.dim1 = n.attributes.perm[1];
                }
                if (node.widgets) {
                    node.widgets.forEach(w => {
                        if (w.name === "Dim 0") w.value = node.properties.dim0;
                        else if (w.name === "Dim 1") w.value = node.properties.dim1;
                    });
                }

            } else if (lgType === "pytorch/squeeze" || lgType === "pytorch/unsqueeze") {
                if (n.attributes.axes !== undefined && n.attributes.axes.length) {
                    node.properties.dim = n.attributes.axes[0];
                } else if (n.attributes.axis !== undefined) {
                    node.properties.dim = n.attributes.axis;
                }
                if (node.widgets && node.widgets[0]) {
                    node.widgets[0].value = node.properties.dim;
                }

            } else if (lgType === "pytorch/generic") {
                node.properties.op_type = n.op_type;
                node.properties.name = n.name;
                node.addWidget("text", "Op Type", n.op_type, () => {}, { disabled: true });
            }
            
            if (lgType === "pytorch/generic" && activeInputs.length > 1) {
                while (node.inputs && node.inputs.length > 0) {
                    node.removeInput(0);
                }
                activeInputs.forEach(inName => {
                    node.addInput(inName, "tensor");
                });
            }

            n.inputs.forEach(inName => {
                if (constantValues[inName] !== undefined) {
                    const val = constantValues[inName];
                    if (!node.properties) node.properties = {};
                    node.properties[inName] = val;
                    const displayVal = typeof val === 'object' ? JSON.stringify(val) : val;
                    node.addWidget("text", inName, displayVal, (newVal) => {
                        try {
                            if (newVal.startsWith('[') || newVal.startsWith('{')) {
                                node.properties[inName] = JSON.parse(newVal);
                            } else {
                                const num = Number(newVal);
                                node.properties[inName] = !isNaN(num) ? num : newVal;
                            }
                        } catch(e) {
                            node.properties[inName] = newVal;
                        }
                    });
                }
            });

            let parentXMax = 80;
            activeInputs.forEach(inName => {
                const prod = tensorProducer[inName];
                if (prod && prod.node.pos) {
                    parentXMax = Math.max(parentXMax, prod.node.pos[0]);
                }
            });

            let x = parentXMax + 260;
            if (!columns[x]) {
                columns[x] = [];
            }
            columns[x].push(node);
            let y = 180 + (columns[x].length - 1) * 155;
            node.pos = [x, y];

            window.AppGraph.add(node);

            if (window.RefreshStatusWidgets) {
                window.RefreshStatusWidgets(node);
            }

            activeInputs.forEach((inName, activeIdx) => {
                const prod = tensorProducer[inName];
                if (prod) {
                    const targetSlot = node.inputs && node.inputs.length > activeIdx ? activeIdx : 0;
                    prod.node.connect(prod.slot, node, targetSlot);
                }
            });

            if (n.outputs && n.outputs.length) {
                tensorProducer[n.outputs[0]] = { node: node, slot: 0 };
            }
        });

        if (LiteGraph.LGraphCanvas.active_canvas) {
            LiteGraph.LGraphCanvas.active_canvas.setDirty(true, true);
        }

        if (typeof arrangeGraph === "function") {
            arrangeGraph(window.CurrentLayoutDirection || "horizontal");
        }

        document.getElementById("onnx-overlay").classList.remove("visible");
        window.customAlert("ONNX Model imported successfully!\nConstant nodes are inlined inside target nodes.\nUnsupported operators were loaded as Red Generic Nodes.", "success", "Import Success");
    }

    if (btnLoadCanvas) {
        btnLoadCanvas.addEventListener("click", () => {
            const data = window.LastParsedONNX;
            if (!data || !window.AppGraph) return window.customAlert("No ONNX data loaded.", "warning", "Warning");

            window.customConfirm("Are you sure you want to clear the editor canvas and load the ONNX model layers?", () => {
                performLoadONNX(data);
            }, null, "Confirm Load");
        });
    }

    // Export Model to ONNX
    const btnExport = document.getElementById("btn-export");
    if (btnExport) {
        btnExport.addEventListener("click", () => {
            if (window.CloseAllDropdowns) window.CloseAllDropdowns();
            const graphData = serializeGraph();
            if (!graphData) return window.customAlert("Graph is empty!", "warning", "Warning");

            btnExport.setAttribute("aria-busy", "true");
            btnExport.textContent = "Exporting...";

            fetch('/api/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ graph: graphData })
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.message || "Export failed"); });
                }
                return response.blob();
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.style.display = "none";
                a.href = url;
                a.download = "hexforge_model.onnx";
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                window.customAlert("Model exported successfully as hexforge_model.onnx!", "success", "Export Success");
            })
            .catch(err => {
                console.error("Export Error:", err);
                window.customAlert("Failed to export model: " + err.message, "error", "Export Error");
            })
            .finally(() => {
                btnExport.removeAttribute("aria-busy");
                btnExport.textContent = "📦 Export ONNX";
            });
        });
    }
});
