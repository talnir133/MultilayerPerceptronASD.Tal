import sys
import os
import ast
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
                             QLineEdit, QCheckBox, QGroupBox, QMessageBox, QFileDialog, QScrollArea)
from PyQt5.QtCore import Qt

DEFAULT_CONFIG = {
    "config_name": "config_1",
    "features_types": [4, 4], "odd_dim": 8, "hidden_size": 30, "n_hidden": 1, "output_size": 1,
    "b_scale_low": 0.1, "b_scale_high": 2.0, "w_scale_low": 1.0, "w_scale_high": 1.0,
    "optimizer_type": "Adam", "activation_type": "Tanh", "batch_size": 128,
    "unique_points_only": False, "seed": 0,
    "exp_stages": [{"stage_name": "M1", "deciding_feature": 0, "odd": False, "epoches": 100}]
}


def get_latest_config():
    os.makedirs("configs", exist_ok=True)
    files = [os.path.join("configs", f) for f in os.listdir("configs") if f.endswith('.json')]
    if files:
        latest = max(files, key=os.path.getmtime)
        try:
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
        self.stage_widgets = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.load_btn = QPushButton("📂 Load Configuration")
        self.load_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        self.load_btn.clicked.connect(self.load_config)
        left_layout.addWidget(self.load_btn)

        group1 = QGroupBox("1. Set Input")
        form1 = QFormLayout()
        self.add_line_edit(form1, "features_types", str(self.config.get("features_types", [4, 4])))
        self.add_spinbox(form1, "odd_dim", self.config.get("odd_dim", 8))
        self.add_spinbox(form1, "batch_size", self.config.get("batch_size", 128), 1024)
        self.add_spinbox(form1, "seed", self.config.get("seed", 0))
        self.add_checkbox(form1, "unique_points_only", self.config.get("unique_points_only", False))
        group1.setLayout(form1)
        left_layout.addWidget(group1)

        group2 = QGroupBox("2. Set Network")
        form2 = QFormLayout()
        self.add_spinbox(form2, "hidden_size", self.config.get("hidden_size", 30))
        self.add_spinbox(form2, "n_hidden", self.config.get("n_hidden", 1))
        self.add_spinbox(form2, "output_size", self.config.get("output_size", 1))
        self.add_double_spinbox(form2, "b_scale_low", self.config.get("b_scale_low", 0.1))
        self.add_double_spinbox(form2, "b_scale_high", self.config.get("b_scale_high", 2.0))
        self.add_double_spinbox(form2, "w_scale_low", self.config.get("w_scale_low", 1.0))
        self.add_double_spinbox(form2, "w_scale_high", self.config.get("w_scale_high", 1.0))
        self.add_combobox(form2, "optimizer_type", ["Adam", "SGD"], self.config.get("optimizer_type", "Adam"))
        self.add_combobox(form2, "activation_type", ["Tanh", "RelU", "Sigmoid", "Identity"],
                          self.config.get("activation_type", "Tanh"))
        group2.setLayout(form2)
        left_layout.addWidget(group2)

        group3 = QGroupBox("3. Experiment Stages")
        group3_layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.stages_container = QWidget()
        self.stages_layout = QVBoxLayout(self.stages_container)
        self.stages_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.stages_container)
        group3_layout.addWidget(scroll)

        self.add_stage_btn = QPushButton("➕ Add Stage")
        self.add_stage_btn.clicked.connect(lambda: self.add_stage_row({}))
        group3_layout.addWidget(self.add_stage_btn)
        group3.setLayout(group3_layout)
        right_layout.addWidget(group3)

        group4 = QGroupBox("4. Save & Run")
        form4 = QFormLayout()
        self.add_line_edit(form4, "config_name", self.config.get("config_name", "config_1"))
        self.run_btn = QPushButton("💾 Save & Run Simulation")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #0078D7; color: white;")
        self.run_btn.clicked.connect(self.on_run)
        form4.addRow(self.run_btn)
        group4.setLayout(form4)
        right_layout.addWidget(group4)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

        self.setLayout(main_layout)
        self.setWindowTitle("Simulation Configuration Manager")
        self.resize(850, 600)
        self.populate_stages()

    def add_spinbox(self, l, n, v, mx=1000, s=1):
        w = QSpinBox();
        w.setRange(0, mx);
        w.setSingleStep(s);
        w.setValue(v)
        self.inputs[n] = w;
        l.addRow(n, w)

    def add_double_spinbox(self, l, n, v):
        w = QDoubleSpinBox();
        w.setRange(0.0, 100.0);
        w.setSingleStep(0.1);
        w.setValue(v)
        self.inputs[n] = w;
        l.addRow(n, w)

    def add_combobox(self, l, n, o, d):
        w = QComboBox();
        w.addItems(o);
        w.setCurrentText(d)
        self.inputs[n] = w;
        l.addRow(n, w)

    def add_line_edit(self, l, n, t):
        w = QLineEdit(t);
        self.inputs[n] = w;
        l.addRow(n, w)

    def add_checkbox(self, l, n, c):
        w = QCheckBox();
        w.setChecked(c);
        self.inputs[n] = w;
        l.addRow(n, w)

    def populate_stages(self):
        for w in self.stage_widgets:
            w["row"].deleteLater()
        self.stage_widgets.clear()
        for stage in self.config.get("exp_stages", []):
            self.add_stage_row(stage)

    def add_stage_row(self, data):
        if len(self.stage_widgets) >= 10:
            QMessageBox.warning(self, "Limit", "Maximum of 10 stages allowed.")
            return

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        name_w = QLineEdit(data.get("stage_name", f"Stage_{len(self.stage_widgets) + 1}"))
        name_w.setPlaceholderText("Name")

        ep_w = QSpinBox()
        ep_w.setRange(1, 10000)
        ep_w.setValue(data.get("epoches", 100))
        ep_w.setPrefix("Ep: ")

        df_w = QSpinBox()
        df_w.setRange(0, 10)
        df_w.setValue(data.get("deciding_feature", 0))
        df_w.setPrefix("Feat: ")

        odd_w = QCheckBox("Odd")
        odd_w.setChecked(data.get("odd", False))

        del_btn = QPushButton("❌")
        del_btn.setFixedWidth(30)

        layout.addWidget(name_w, stretch=2)
        layout.addWidget(ep_w, stretch=1)
        layout.addWidget(df_w, stretch=1)
        layout.addWidget(odd_w)
        layout.addWidget(del_btn)

        stage_dict = {"row": row, "name": name_w, "ep": ep_w, "df": df_w, "odd": odd_w}
        self.stage_widgets.append(stage_dict)
        self.stages_layout.addWidget(row)

        del_btn.clicked.connect(lambda _, r=row, d=stage_dict: self.remove_stage(r, d))

    def remove_stage(self, row_widget, stage_dict):
        self.stages_layout.removeWidget(row_widget)
        row_widget.deleteLater()
        if stage_dict in self.stage_widgets:
            self.stage_widgets.remove(stage_dict)

    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "configs", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
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
                        widget.setText(str(val))
                self.populate_stages()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    def on_run(self):
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

        stages_data = []
        for w in self.stage_widgets:
            stages_data.append({
                "stage_name": w["name"].text(),
                "deciding_feature": w["df"].value(),
                "odd": w["odd"].isChecked(),
                "epoches": w["ep"].value()
            })

        if not stages_data:
            QMessageBox.warning(self, "Warning", "You must have at least one stage.")
            return

        self.config["exp_stages"] = stages_data

        config_path = f"configs/{self.config['config_name']}.json"
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not save JSON file:\n{e}")

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        QApplication.processEvents()
        try:
            self.run_callback(self.config)
            QMessageBox.information(self, "Success", "Simulation finished and saved!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{e}")
        finally:
            self.run_btn.setEnabled(True)
            self.run_btn.setText("💾 Save & Run Simulation")


def launch_gui(run_callback):
    app = QApplication.instance() or QApplication(sys.argv)
    gui = ConfigGUI(run_callback)
    gui.show()
    app.exec_()