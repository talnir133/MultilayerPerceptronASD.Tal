import sys, os, ast, json, inspect
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QLabel,
                             QLineEdit, QGroupBox, QFileDialog, QScrollArea, QFrame)
from PyQt5.QtCore import Qt

# Import the registry to dynamically build the GUI elements
from classification_rules import RULES_REGISTRY

DEFAULT_CONFIG = {
    "exp_name": "default_config",
    "features_types": [4, 4, 8], "hidden_size": 30, "n_hidden": 0,
    "b_scale_low": 0.0, "b_scale_high": 0.0, "w_scale_low": 0.1, "w_scale_high": 50.0,
    "optimizer_type": "Adam", "activation_type": "Identity", "batch_size": 32,
    "lr": 0.004,
    "seed": 0, "sd": 0.0,
    "exp_blocks": [
        {"block_name": "M1", "rule": "upper_half", "deciding_feature": 0, "zero_features": [2], "epochs": 25,
         "alpha_class": 1.0, "alpha_rec": 0.0}
    ]
}


def get_latest_config():
    os.makedirs("configs", exist_ok=True)
    files = [os.path.join("configs", f) for f in os.listdir("configs") if f.endswith('.json')]
    if files:
        try:
            with open(max(files, key=os.path.getmtime), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


class ConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.config = get_latest_config()
        self.inputs = {}
        self.block_widgets = []
        self.start_simulation = False
        self.init_ui()

    def init_ui(self):
        main_layout, left_layout, right_layout = QHBoxLayout(), QVBoxLayout(), QVBoxLayout()

        self.load_btn = QPushButton("📂 Load Configuration")
        self.load_btn.clicked.connect(self.load_config)
        left_layout.addWidget(self.load_btn)

        # --- 1. Set Input ---
        group1, form1 = QGroupBox("1. Set Input"), QFormLayout()
        self.add_line_edit(form1, "features_types", str(self.config.get("features_types", [4, 4, 8])))
        self.add_double_spinbox(form1, "sd", self.config.get("sd", 0.0))
        self.add_spinbox(form1, "batch_size", self.config.get("batch_size", 32), 1024)
        self.add_spinbox(form1, "seed", self.config.get("seed", 0))

        self.data_shape_label = QLabel("")
        self.data_shape_label.setStyleSheet("color: black; font-weight: bold; margin-top: 5px;")
        form1.addRow("", self.data_shape_label)
        group1.setLayout(form1)
        left_layout.addWidget(group1)

        # --- 2. Set Network ---
        group2, form2 = QGroupBox("2. Set Network"), QFormLayout()
        for k in ["hidden_size", "n_hidden"]:
            self.add_spinbox(form2, k, self.config.get(k, 0))
        for k in ["b_scale_low", "b_scale_high", "w_scale_low", "w_scale_high"]:
            self.add_double_spinbox(form2, k, self.config.get(k, 0.0))

        # Global learning rate with high precision
        self.add_double_spinbox(form2, "lr", self.config.get("lr", 0.004), step=0.0001, decimals=4)
        self.add_combobox(form2, "optimizer_type", ["Adam", "SGD"], self.config.get("optimizer_type", "Adam"))
        self.add_combobox(form2, "activation_type", ["Tanh", "RelU", "Sigmoid", "Identity"],
                          self.config.get("activation_type", "Identity"))
        group2.setLayout(form2)
        left_layout.addWidget(group2)

        # --- 3. Experiment blocks ---
        group3, gl = QGroupBox("3. Experiment blocks"), QVBoxLayout()
        sc = QScrollArea(widgetResizable=True)
        self.sw, self.sl = QWidget(), QVBoxLayout()
        self.sl.setAlignment(Qt.AlignTop)
        self.sw.setLayout(self.sl)
        sc.setWidget(self.sw)

        btn = QPushButton("➕ Add Block")
        btn.clicked.connect(lambda: self.add_block_row({}))
        gl.addWidget(sc)
        gl.addWidget(btn)
        group3.setLayout(gl)
        right_layout.addWidget(group3)

        # --- 4. Save & Run ---
        group4, f4 = QGroupBox("4. Save & Run"), QFormLayout()
        self.add_line_edit(f4, "exp_name", self.config.get("exp_name", "test"))
        self.run_btn = QPushButton("💾 Save & Run Simulation")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #0078D7; color: white;")
        self.run_btn.clicked.connect(self.on_run)
        f4.addRow(self.run_btn)
        group4.setLayout(f4)
        right_layout.addWidget(group4)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        self.setLayout(main_layout)
        self.setWindowTitle("Simulation Manager")
        self.resize(1050, 650)

        self.inputs["features_types"].textChanged.connect(self.update_all_shapes)
        self.populate_blocks()

    # --- Helper UI Builders ---
    def add_spinbox(self, layout, name, value, mx=1000):
        w = QSpinBox(minimum=0, maximum=mx, value=value)
        self.inputs[name] = w
        layout.addRow(name, w)

    def add_double_spinbox(self, layout, name, value, step=0.1, decimals=1):
        w = QDoubleSpinBox(minimum=0.0, maximum=100.0, singleStep=step, decimals=decimals, value=value)
        self.inputs[name] = w
        layout.addRow(name, w)

    def add_combobox(self, layout, name, options, default):
        w = QComboBox()
        w.addItems(options)
        w.setCurrentText(default)
        self.inputs[name] = w
        layout.addRow(name, w)

    def add_line_edit(self, layout, name, text):
        w = QLineEdit(text)
        self.inputs[name] = w
        layout.addRow(name, w)

    def _create_block_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #333; font-size: 11px; font-weight: bold;")
        return lbl

    def load_config(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load", "configs", "*.json")
        if p:
            with open(p, 'r') as f:
                self.config = json.load(f)
            for k, w in self.inputs.items():
                if k in self.config:
                    v = self.config[k]
                    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                        w.setValue(v)
                    elif isinstance(w, QComboBox):
                        w.setCurrentText(str(v))
                    elif isinstance(w, QLineEdit):
                        w.setText(str(v))
            self.populate_blocks()

    def populate_blocks(self):
        for w in self.block_widgets: w["row"].deleteLater()
        self.block_widgets.clear()
        for s in self.config.get("exp_blocks", []): self.add_block_row(s)
        self.update_all_shapes()

    def add_block_row(self, data):
        row = QFrame()
        row.setStyleSheet("QFrame { background-color: #f4f6f9; border-radius: 5px; margin-bottom: 2px; }")
        ly = QVBoxLayout(row)
        ly.setContentsMargins(10, 10, 10, 10)

        row_top = QHBoxLayout()

        # Build fields
        name = QLineEdit(data.get("block_name", f"S{len(self.block_widgets) + 1}"))
        name.setFixedWidth(65)

        ep = QSpinBox(minimum=1, maximum=10000, value=data.get("epochs", 25))

        rule_cb = QComboBox()
        rule_cb.addItems(list(RULES_REGISTRY.keys()))
        if data.get("rule", "upper_half") in RULES_REGISTRY:
            rule_cb.setCurrentText(data.get("rule", "upper_half"))

        zf_val = data.get("zero_features", [])
        zf = QLineEdit(",".join(map(str, zf_val)) if isinstance(zf_val, (list, tuple)) else str(zf_val))
        zf.setPlaceholderText("e.g. 2,3")
        zf.setFixedWidth(60)

        ac = QDoubleSpinBox(minimum=0.0, maximum=100.0, singleStep=0.1, value=data.get("alpha_class", 1.0))
        ac.setFixedWidth(45)

        ar = QDoubleSpinBox(minimum=0.0, maximum=100.0, singleStep=0.1, value=data.get("alpha_rec", 0.0))
        ar.setFixedWidth(45)

        dl = QPushButton("❌")
        dl.setFixedWidth(30)
        dl.setStyleSheet("background-color: transparent;")

        # Add to layout
        widgets_to_add = [
            (self._create_block_label("Name:"), name), (self._create_block_label("Epochs:"), ep),
            (self._create_block_label("Rule:"), rule_cb), (self._create_block_label("Zero Feats:"), zf),
            (self._create_block_label("a_c:"), ac), (self._create_block_label("a_r:"), ar)
        ]
        for lbl, w in widgets_to_add:
            row_top.addWidget(lbl)
            row_top.addWidget(w)

        row_top.addStretch()
        row_top.addWidget(dl)

        params_layout, params_widgets = QHBoxLayout(), {}
        params_layout.setAlignment(Qt.AlignLeft)

        def update_rule_params():
            for i in reversed(range(params_layout.count())):
                widget = params_layout.itemAt(i).widget()
                if widget: widget.deleteLater()
            params_widgets.clear()

            rule_func = RULES_REGISTRY[rule_cb.currentText()]
            for param_name, param in inspect.signature(rule_func).parameters.items():
                if param_name in ['names', 'features_types', 'kwargs']: continue

                lbl = QLabel(f"{param_name}:")
                lbl.setStyleSheet("color: #333; font-size: 11px;")

                default_val = param.default if param.default != inspect.Parameter.empty else 0
                val = data.get(param_name, default_val)

                spb = QDoubleSpinBox() if isinstance(default_val, float) else QSpinBox()
                spb.setRange(-1000.0 if isinstance(spb, QDoubleSpinBox) else -1000, 1000)
                spb.setValue(val)
                spb.setFixedWidth(60)

                params_layout.addWidget(lbl)
                params_layout.addWidget(spb)
                params_widgets[param_name] = spb

        rule_cb.currentTextChanged.connect(update_rule_params)
        update_rule_params()

        shape_lbl = QLabel("")
        shape_lbl.setStyleSheet("color: #555; font-size: 11px; font-weight: bold; margin-top: 3px;")

        ly.addLayout(row_top)
        ly.addLayout(params_layout)
        ly.addWidget(shape_lbl)

        d = {"row": row, "name": name, "ep": ep, "rule_cb": rule_cb, "zf": zf, "ac": ac, "ar": ar,
             "params_widgets": params_widgets, "shape_lbl": shape_lbl}
        self.block_widgets.append(d)
        self.sl.addWidget(row)

        dl.clicked.connect(lambda: self.remove_block(row, d))
        name.textChanged.connect(self.update_all_shapes)
        zf.textChanged.connect(self.update_all_shapes)
        self.update_all_shapes()

    def remove_block(self, r, d):
        self.sl.removeWidget(r)
        r.deleteLater()
        if d in self.block_widgets: self.block_widgets.remove(d)
        self.update_all_shapes()

    def parse_zf(self, text):
        return [int(x.strip()) for x in text.replace('(', '').replace(')', '').split(",") if x.strip().isdigit()]

    def update_all_shapes(self):
        try:
            ft = ast.literal_eval(self.inputs["features_types"].text())
            if not isinstance(ft, list): raise ValueError
            p_base, s_base = 1, sum(ft)
            for d in ft: p_base *= d
            self.data_shape_label.setText(f"Base Data Shape: ({p_base}, {s_base})")

            for w in self.block_widgets:
                zf_list = self.parse_zf(w["zf"].text())
                p_block = 1
                for i, dim in enumerate(ft):
                    if i not in zf_list: p_block *= dim
                w["shape_lbl"].setText(f"Data shape for '{w['name'].text()}' block: ({p_block}, {s_base})")
        except Exception:
            self.data_shape_label.setText("Exp Base Data Shape: (Invalid)")
            for w in self.block_widgets: w["shape_lbl"].setText("Data shape: (Invalid)")

    def on_run(self):
        for k, w in self.inputs.items():
            if isinstance(w, QSpinBox):
                self.config[k] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                self.config[k] = round(w.value(), 4)
            elif isinstance(w, QComboBox):
                self.config[k] = w.currentText()
            elif isinstance(w, QLineEdit):
                if k == "features_types":
                    try:
                        self.config[k] = ast.literal_eval(w.text())
                    except Exception:
                        pass
                else:
                    self.config[k] = w.text()

        self.config["exp_blocks"] = []
        for w in self.block_widgets:
            block_cfg = {
                "block_name": w["name"].text(), "rule": w["rule_cb"].currentText(),
                "zero_features": self.parse_zf(w["zf"].text()), "epochs": w["ep"].value(),
                "alpha_class": w["ac"].value(), "alpha_rec": w["ar"].value()
            }
            for param_name, param_widget in w["params_widgets"].items():
                block_cfg[param_name] = param_widget.value()
            self.config["exp_blocks"].append(block_cfg)

        os.makedirs("configs", exist_ok=True)
        with open(f"configs/{self.config['exp_name']}.json", 'w') as f:
            json.dump(self.config, f, indent=4)

        self.start_simulation = True
        self.close()


def launch_gui():
    app = QApplication.instance() or QApplication(sys.argv)
    g = ConfigGUI()
    g.show()
    app.exec_()
    return g.config if g.start_simulation else None