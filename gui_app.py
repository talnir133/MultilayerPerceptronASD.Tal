import sys, os, ast, json
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QLabel,
                             QLineEdit, QGroupBox, QMessageBox, QFileDialog, QScrollArea, QFrame)
from PyQt5.QtCore import Qt

DEFAULT_CONFIG = {
    "exp_name": "default_config",
    "features_types": [4, 4, 8], "hidden_size": 30, "n_hidden": 0, "output_size": 1,
    "b_scale_low": 0.0, "b_scale_high": 0.0, "w_scale_low": 0.1, "w_scale_high": 50.0,
    "optimizer_type": "Adam", "activation_type": "Identity", "batch_size": 32,
    "seed": 0, "sd": 0.0,
    "exp_blocks": [{"block_name": "M1", "deciding_feature": 0, "zero_features": [2], "epoches": 25}]
}


def get_latest_config():
    os.makedirs("configs", exist_ok=True)
    files = [os.path.join("configs", f) for f in os.listdir("configs") if f.endswith('.json')]
    if files:
        try:
            latest = max(files, key=os.path.getmtime)
            with open(latest, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_CONFIG.copy()


class ConfigGUI(QWidget):
    def __init__(self, run_callback):
        super().__init__()
        self.config = get_latest_config()
        self.run_callback = run_callback
        self.inputs = {}
        self.block_widgets = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_layout, right_layout = QVBoxLayout(), QVBoxLayout()

        self.load_btn = QPushButton("📂 Load Configuration")
        self.load_btn.clicked.connect(self.load_config)
        left_layout.addWidget(self.load_btn)

        # --- 1. Set Input ---
        group1 = QGroupBox("1. Set Input")
        form1 = QFormLayout()
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
        group2 = QGroupBox("2. Set Network")
        form2 = QFormLayout()
        for k in ["hidden_size", "n_hidden", "output_size"]: self.add_spinbox(form2, k, self.config.get(k, 0))
        for k in ["b_scale_low", "b_scale_high", "w_scale_low", "w_scale_high"]: self.add_double_spinbox(form2, k,
                                                                                                         self.config.get(
                                                                                                             k, 0.0))
        self.add_combobox(form2, "optimizer_type", ["Adam", "SGD"], self.config.get("optimizer_type", "Adam"))
        self.add_combobox(form2, "activation_type", ["Tanh", "RelU", "Sigmoid", "Identity"],
                          self.config.get("activation_type", "Identity"))
        group2.setLayout(form2)
        left_layout.addWidget(group2)

        # --- 3. Experiment blocks ---
        group3 = QGroupBox("3. Experiment blocks")
        gl = QVBoxLayout()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self.sw = QWidget()
        self.sl = QVBoxLayout(self.sw)
        self.sl.setAlignment(Qt.AlignTop)
        sc.setWidget(self.sw)
        gl.addWidget(sc)
        btn = QPushButton("➕ Add Block")
        btn.clicked.connect(lambda: self.add_block_row({}))
        gl.addWidget(btn)
        group3.setLayout(gl)
        right_layout.addWidget(group3)

        # --- 4. Save & Run ---
        group4 = QGroupBox("4. Save & Run")
        f4 = QFormLayout()
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
        self.resize(900, 650)

        self.inputs["features_types"].textChanged.connect(self.update_all_shapes)
        self.populate_blocks()

    def add_spinbox(self, l, n, v, mx=1000):
        w = QSpinBox();
        w.setRange(0, mx);
        w.setValue(v);
        self.inputs[n] = w;
        l.addRow(n, w)

    def add_double_spinbox(self, l, n, v):
        w = QDoubleSpinBox();
        w.setRange(0.0, 100.0);
        w.setSingleStep(0.1);
        w.setValue(v);
        self.inputs[n] = w;
        l.addRow(n, w)

    def add_combobox(self, l, n, o, d):
        w = QComboBox();
        w.addItems(o);
        w.setCurrentText(d);
        self.inputs[n] = w;
        l.addRow(n, w)

    def add_line_edit(self, l, n, t):
        w = QLineEdit(t);
        self.inputs[n] = w;
        l.addRow(n, w)

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
        name = QLineEdit(data.get("block_name", f"S{len(self.block_widgets) + 1}"))
        name.setFixedWidth(80)
        ep = QSpinBox();
        ep.setRange(1, 10000);
        ep.setValue(data.get("epoches", 25));
        ep.setPrefix("Ep: ")
        df = QSpinBox();
        df.setRange(0, 10);
        df.setValue(data.get("deciding_feature", 0));
        df.setPrefix("Feat: ")

        zf_val = data.get("zero_features", [])
        zf_str = ",".join(map(str, zf_val)) if isinstance(zf_val, (list, tuple)) else str(zf_val)
        zf = QLineEdit(zf_str);
        zf.setPlaceholderText("Zero Feats (e.g. 2)")

        dl = QPushButton("❌");
        dl.setFixedWidth(30)
        dl.setStyleSheet("background-color: transparent;")

        for w in [name, ep, df, zf, dl]: row_top.addWidget(w)

        shape_lbl = QLabel("")
        shape_lbl.setStyleSheet("color: #555; font-size: 11px; font-weight: bold; margin-top: 3px;")

        ly.addLayout(row_top)
        ly.addWidget(shape_lbl)

        d = {"row": row, "name": name, "ep": ep, "df": df, "zf": zf, "shape_lbl": shape_lbl}
        self.block_widgets.append(d)
        self.sl.addWidget(row)

        dl.clicked.connect(lambda: self.remove_block(row, d))

        name.textChanged.connect(self.update_all_shapes)
        zf.textChanged.connect(self.update_all_shapes)
        self.update_all_shapes()

    def remove_block(self, r, d):
        self.sl.removeWidget(r);
        r.deleteLater()
        if d in self.block_widgets: self.block_widgets.remove(d)
        self.update_all_shapes()

    def parse_zf(self, text):
        res = []
        for x in text.replace('(', '').replace(')', '').split(","):
            if x.strip().isdigit(): res.append(int(x.strip()))
        return res

    def update_all_shapes(self):
        try:
            ft = ast.literal_eval(self.inputs["features_types"].text())
            if not isinstance(ft, list): raise ValueError

            p_base, s_base = 1, sum(ft)
            for d in ft: p_base *= d
            self.data_shape_label.setText(f"Exp Base Data Shape: ({p_base}, {s_base})")

            for w in self.block_widgets:
                name = w["name"].text()
                zf_list = self.parse_zf(w["zf"].text())

                p_block = 1
                for i, dim in enumerate(ft):
                    if i not in zf_list:
                        p_block *= dim

                w["shape_lbl"].setText(f"Data shape for '{name}' block: ({p_block}, {s_base})")
        except:
            self.data_shape_label.setText("Exp Base Data Shape: (Invalid)")
            for w in self.block_widgets:
                w["shape_lbl"].setText("Data shape: (Invalid)")

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

    def on_run(self):
        for k, w in self.inputs.items():
            if isinstance(w, QSpinBox):
                self.config[k] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                self.config[k] = round(w.value(), 4)  # 4 ספרות זה מעולה ל-SD ול-Scales
            elif isinstance(w, QComboBox):
                self.config[k] = w.currentText()
            elif isinstance(w, QLineEdit):
                if k == "features_types":
                    try:
                        self.config[k] = ast.literal_eval(w.text())
                    except:
                        pass
                else:
                    self.config[k] = w.text()

        self.config["exp_blocks"] = [{"block_name": w["name"].text(), "deciding_feature": w["df"].value(),
                                      "zero_features": self.parse_zf(w["zf"].text()), "epoches": w["ep"].value()} for w
                                     in self.block_widgets]

        with open(f"configs/{self.config['exp_name']}.json", 'w') as f:
            json.dump(self.config, f, indent=4)

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        QApplication.processEvents()

        try:
            self.run_callback(self.config)
            QMessageBox.information(self, "Success", "Simulation finished and saved")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.run_btn.setEnabled(True)
            self.run_btn.setText("💾 Save & Run Simulation")


def launch_gui(cb):
    app = QApplication.instance() or QApplication(sys.argv)
    g = ConfigGUI(cb)
    g.show()
    app.exec_()
