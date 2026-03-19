import sys
import os
import ast
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QFormLayout,
                             QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
                             QLineEdit, QCheckBox, QGroupBox, QMessageBox, QFileDialog)

DEFAULT_CONFIG = {
    "config_name": "config_1",
    "num_epochs": 1000,
    "features_types": [4, 4],
    "odd_dim": 8,
    "hidden_size": 30,
    "n_hidden": 1,
    "output_size": 1,
    "b_scale_low": 0.1,
    "b_scale_high": 2.0,
    "w_scale_low": 1.0,
    "w_scale_high": 1.0,
    "optimizer_type": "Adam",
    "activation_type": "Tanh",
    "batch_size": 128,
    "unique_points_only": False,
    "seed": 0
}


class ConfigGUI(QWidget):
    def __init__(self, run_callback):
        super().__init__()
        self.config = DEFAULT_CONFIG.copy()
        self.run_callback = run_callback
        self.inputs = {}

        # Ensure configs directory exists
        os.makedirs("configs", exist_ok=True)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- Load Configuration Button ---
        self.load_btn = QPushButton("📂 Load Configuration")
        self.load_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        self.load_btn.clicked.connect(self.load_config)
        layout.addWidget(self.load_btn)

        # 1. Set Input
        group1 = QGroupBox("1. Set Input")
        form1 = QFormLayout()
        self.add_line_edit(form1, "features_types", str(self.config["features_types"]))
        self.add_spinbox(form1, "odd_dim", self.config["odd_dim"])
        self.add_spinbox(form1, "batch_size", self.config["batch_size"], max_val=1024)
        self.add_spinbox(form1, "seed", self.config["seed"])
        self.add_checkbox(form1, "unique_points_only", self.config["unique_points_only"])
        group1.setLayout(form1)
        layout.addWidget(group1)

        # 2. Set Network
        group2 = QGroupBox("2. Set Network")
        form2 = QFormLayout()
        self.add_spinbox(form2, "hidden_size", self.config["hidden_size"])
        self.add_spinbox(form2, "n_hidden", self.config["n_hidden"])
        self.add_spinbox(form2, "output_size", self.config["output_size"])
        self.add_double_spinbox(form2, "b_scale_low", self.config["b_scale_low"])
        self.add_double_spinbox(form2, "b_scale_high", self.config["b_scale_high"])
        self.add_double_spinbox(form2, "w_scale_low", self.config["w_scale_low"])
        self.add_double_spinbox(form2, "w_scale_high", self.config["w_scale_high"])
        self.add_combobox(form2, "optimizer_type", ["Adam", "SGD"], self.config["optimizer_type"])
        self.add_combobox(form2, "activation_type", ["Tanh", "RelU", "Sigmoid", "Identity"],
                          self.config["activation_type"])
        group2.setLayout(form2)
        layout.addWidget(group2)

        # 3. Set Experiment
        group3 = QGroupBox("3. Set Experiment")
        form3 = QFormLayout()
        self.add_spinbox(form3, "num_epochs", self.config["num_epochs"], max_val=10000, step=100)
        self.exp_stages_combo = QComboBox()
        self.exp_stages_combo.addItems(["Initial Only", "Initial + Flexibility", "Initial + Generalization"])
        form3.addRow("Experiment Stages:", self.exp_stages_combo)
        group3.setLayout(form3)
        layout.addWidget(group3)

        # 4. Save & Run
        group4 = QGroupBox("4. Save & Run")
        form4 = QFormLayout()
        self.add_line_edit(form4, "config_name", self.config["config_name"])
        self.run_btn = QPushButton("💾 Save & Run Simulation")
        # Changed background color to blue and text color to white for readability
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #0078D7; color: white;")
        self.run_btn.clicked.connect(self.on_run)
        form4.addRow(self.run_btn)
        group4.setLayout(form4)
        layout.addWidget(group4)

        self.setLayout(layout)
        self.setWindowTitle("Simulation Configuration Manager")
        self.resize(400, 700)

    # --- UI Generators ---
    def add_spinbox(self, layout, name, value, max_val=1000, step=1):
        w = QSpinBox();
        w.setRange(0, max_val);
        w.setSingleStep(step);
        w.setValue(value)
        self.inputs[name] = w;
        layout.addRow(name, w)

    def add_double_spinbox(self, layout, name, value):
        w = QDoubleSpinBox();
        w.setRange(0.0, 100.0);
        w.setSingleStep(0.1);
        w.setValue(value)
        self.inputs[name] = w;
        layout.addRow(name, w)

    def add_combobox(self, layout, name, options, default):
        w = QComboBox();
        w.addItems(options);
        w.setCurrentText(default)
        self.inputs[name] = w;
        layout.addRow(name, w)

    def add_line_edit(self, layout, name, text):
        w = QLineEdit(text);
        self.inputs[name] = w;
        layout.addRow(name, w)

    def add_checkbox(self, layout, name, checked):
        w = QCheckBox();
        w.setChecked(checked);
        self.inputs[name] = w;
        layout.addRow(name, w)

    # --- Actions ---
    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "configs", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                self.config.update(loaded_config)
                self.update_ui_from_config()
                # Success message removed for better UX
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    def update_ui_from_config(self):
        for key, widget in self.inputs.items():
            if key not in self.config: continue
            val = self.config[key]
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(val)
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(val)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(val)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(val) if key == "features_types" else str(val))

        # Update combo box for exp_stages
        stages = self.config.get("exp_stages", ["initial"])
        if len(stages) == 1:
            self.exp_stages_combo.setCurrentIndex(0)
        elif "flexibility" in stages[-1]:
            self.exp_stages_combo.setCurrentIndex(1)
        else:
            self.exp_stages_combo.setCurrentIndex(2)

    def on_run(self):
        # 1. Pull data from UI to config dict
        for key, widget in self.inputs.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                self.config[key] = widget.value()
            elif isinstance(widget, QComboBox):
                self.config[key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                self.config[key] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                val = widget.text()
                if key == "features_types":
                    try:
                        self.config[key] = ast.literal_eval(val)
                    except:
                        self.config[key] = [4, 4]
                else:
                    self.config[key] = val

        idx = self.exp_stages_combo.currentIndex()
        if idx == 0:
            self.config["exp_stages"] = ["initial"]
        elif idx == 1:
            self.config["exp_stages"] = ["initial", "flexibility"]
        else:
            self.config["exp_stages"] = ["initial", "generalization"]

        # 2. Save JSON file to configs folder
        config_path = f"configs/{self.config['config_name']}.json"
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not save JSON file:\n{e}")

        # 3. Run Simulation
        self.run_btn.setEnabled(False);
        self.run_btn.setText("Running...")
        QApplication.processEvents()
        try:
            self.run_callback(self.config)
            QMessageBox.information(self, "Success", "Simulation finished and saved!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{e}")
        finally:
            self.run_btn.setEnabled(True);
            self.run_btn.setText("💾 Save & Run Simulation")


def launch_gui(run_callback):
    app = QApplication.instance() or QApplication(sys.argv)
    gui = ConfigGUI(run_callback)
    gui.show()
    app.exec_()

