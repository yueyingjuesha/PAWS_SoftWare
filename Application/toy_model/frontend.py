import sys
from os import getcwd, system
from os.path import join
from run_makedata import run_makedata




from pandas import read_csv, DataFrame


from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout, QFileDialog, QMessageBox


def load_csv(path):
    file = read_csv(path)
    return file.values

def save_csv(data, path):
    DataFrame(data).to_csv(path)

def count(data):
    return data ** 2

def calculate_sum(data):
    return data ** 2


class toy_runner():
    def __init__(self, mode, path):
        self.mode = mode
        #self.data = load_csv(path)
        self.path = path
        self.result = None
        self.run_flag = False

        self.warm_message = None

    def run(self):
        if self.mode == 1:
            result = run_makedata(self.path)
            if result == 'Finished!':
                self.run_flag = True
            else:
                self.run_flag = False
                self.warm_message = file
        elif self.mode == 2:
            self.result = calculate_sum(load_csv(path))
            self.run_flag = True

    def check_result(self):
        return self.run_flag

    def get_result(self):
        return self.result

    def save_result(self, dir):
        path = join(dir, "toy_output.csv")
        save_csv(self.result, path)


class MainForm(QWidget):
    def __init__(self, name = 'MainForm'):
        super(MainForm,self).__init__()
        self.setWindowTitle(name)
        self.cwd = getcwd() 
        self.resize(300, 100) 

        self.chosen_model = None
        self.has_result = None
        self.chosen_file = None
        self.save_path = None
        self.runner = None



        ## btn
        self.label1 = QLabel("Choose File:", self)
        self.btn_chooseFile = QPushButton("Choose File", self)  

        self.label2 = QLabel("Select Model:", self)
        self.btn_selectModel = QComboBox(self)  
        self.btn_selectModel.addItems(['model1','model2','model3'])

        self.btn_runModel = QPushButton("Run Model", self) 

        self.label3 = QLabel("Choose Save Path:", self)
        self.btn_chooseDir = QPushButton("Choose Save Path", self)  

        self.btn_exportResult = QPushButton("Export Result", self)  

        self.btn_runModel.setEnabled(False)
        self.btn_exportResult.setEnabled(False)





        layout = QVBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.btn_chooseFile)
        layout.addWidget(self.label2)
        layout.addWidget(self.btn_selectModel)
        layout.addWidget(self.btn_runModel)
        layout.addWidget(self.label3)
        layout.addWidget(self.btn_chooseDir)
        layout.addWidget(self.btn_exportResult)
        self.setLayout(layout)


        self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)
        self.btn_selectModel.activated[str].connect(self.slot_btn_selectModel)
        self.btn_runModel.clicked.connect(self.slot_btn_runModel)
        self.btn_chooseDir.clicked.connect(self.slot_btn_chooseDir)
        self.btn_exportResult.clicked.connect(self.slot_btn_exportResult)





    def slot_btn_chooseFile(self):
        self.chosen_file = QFileDialog.getExistingDirectory(self, "getExistingDirectory", "./") 
        self.btn_chooseFile.setText(self.chosen_file)
        if self.chosen_model:
            self.btn_runModel.setEnabled(True)
        return


    def slot_btn_selectModel(self, text):
        self.chosen_model = text
        if self.chosen_file:
            self.btn_runModel.setEnabled(True)
        return


    def slot_btn_runModel(self):
        QMessageBox.information(self, 'info1', 'Running {}, please wait'.format(self.chosen_model))
        model_type = int(self.chosen_model[-1])
        self.runner = toy_runner(model_type, self.chosen_file)
        self.runner.run()
        if self.runner.warm_message:
            QMessageBox.information(self, 'info3', 'No such a file in selected path: {}'.format(self.runner.warm_message))
            return 
        QMessageBox.information(self, 'info0', 'Running finished!'.format(self.chosen_model))
        self.has_result = self.runner.check_result()
        if self.save_path and self.has_result:
            self.btn_exportResult.setEnabled(True)
        return

    def slot_btn_exportResult(self):
        QMessageBox.information(self, 'info2', 'Results are saved in \n{}'.format(self.save_path))
        if self.chosen_model == 'model1':
            system('mv final.csv {}'.format(self.save_path))
            system('mv predictions1.txt {}'.format(self.save_path))
            system('mv predictions2.txt {}'.format(self.save_path))
            system('mv predictions_heatmap1.asc {}'.format(self.save_path))
        else:
            self.runner.save_result(self.save_path)
        return

    def slot_btn_chooseDir(self):
        dir_choose = QFileDialog.getExistingDirectory(self,  
                                    "Choose Path",  
                                    self.cwd) 

        self.save_path = dir_choose
        self.btn_chooseDir.setText(dir_choose)
        if self.save_path and self.has_result:
            self.btn_exportResult.setEnabled(True)
        return




    def closeEvent(self, event):  
        reply = QMessageBox.question(self,
                                               'exit',
                                               "Do you want to exit?",
                                               QMessageBox.Yes | QMessageBox.No,
                                               QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



if __name__=="__main__":
    app = QApplication(sys.argv)
    mainForm = MainForm('Demo V1.0')
    mainForm.show()
    sys.exit(app.exec_())
