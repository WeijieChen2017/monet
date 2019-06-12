# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import glob
from global_dict.w_global import gbl_set_value, gbl_get_value

from PyQt5 import QtCore, QtGui, QtWidgets
from blurring.w_blurring import blurring_data_generator


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1920, 1080)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(20, 10, 981, 751))
        self.tabWidget.setObjectName("tabWidget")
        self.Data = QtWidgets.QWidget()
        self.Data.setObjectName("Data")
        self.Input = QtWidgets.QGroupBox(self.Data)
        self.Input.setGeometry(QtCore.QRect(30, 20, 911, 81))
        self.Input.setObjectName("Input")
        self.label = QtWidgets.QLabel(self.Input)
        self.label.setGeometry(QtCore.QRect(20, 40, 51, 25))
        self.label.setObjectName("label")
        self.data_x = QtWidgets.QTextEdit(self.Input)
        self.data_x.setGeometry(QtCore.QRect(80, 40, 301, 25))
        self.data_x.setObjectName("data_x")
        self.Blurring = QtWidgets.QGroupBox(self.Data)
        self.Blurring.setGeometry(QtCore.QRect(30, 120, 911, 181))
        self.Blurring.setObjectName("Blurring")
        self.label_2 = QtWidgets.QLabel(self.Blurring)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 121, 25))
        self.label_2.setObjectName("label_2")
        self.fwhm4 = QtWidgets.QCheckBox(self.Blurring)
        self.fwhm4.setGeometry(QtCore.QRect(20, 60, 85, 21))
        self.fwhm4.setChecked(True)
        self.fwhm4.setObjectName("fwhm4")
        self.fwhm5 = QtWidgets.QCheckBox(self.Blurring)
        self.fwhm5.setGeometry(QtCore.QRect(20, 80, 85, 21))
        self.fwhm5.setChecked(True)
        self.fwhm5.setObjectName("fwhm5")
        self.fwhm6 = QtWidgets.QCheckBox(self.Blurring)
        self.fwhm6.setGeometry(QtCore.QRect(20, 100, 85, 21))
        self.fwhm6.setChecked(True)
        self.fwhm6.setObjectName("fwhm6")
        self.fwhm7 = QtWidgets.QCheckBox(self.Blurring)
        self.fwhm7.setGeometry(QtCore.QRect(20, 120, 85, 21))
        self.fwhm7.setChecked(True)
        self.fwhm7.setObjectName("fwhm7")
        self.fwhm8 = QtWidgets.QCheckBox(self.Blurring)
        self.fwhm8.setGeometry(QtCore.QRect(20, 140, 85, 21))
        self.fwhm8.setChecked(True)
        self.fwhm8.setObjectName("fwhm8")
        self.generate = QtWidgets.QPushButton(self.Data)
        self.generate.setGeometry(QtCore.QRect(30, 440, 80, 23))
        self.generate.setObjectName("generate")
        self.groupBox = QtWidgets.QGroupBox(self.Data)
        self.groupBox.setGeometry(QtCore.QRect(30, 320, 911, 101))
        self.groupBox.setObjectName("groupBox")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 30, 121, 25))
        self.label_3.setObjectName("label_3")
        self.slice1 = QtWidgets.QRadioButton(self.groupBox)
        self.slice1.setGeometry(QtCore.QRect(20, 50, 100, 21))
        self.slice1.setObjectName("slice1")
        self.slice3 = QtWidgets.QRadioButton(self.groupBox)
        self.slice3.setGeometry(QtCore.QRect(20, 70, 100, 21))
        self.slice3.setChecked(True)
        self.slice3.setObjectName("slice3")
        self.tabWidget.addTab(self.Data, "")
        self.Model = QtWidgets.QWidget()
        self.Model.setObjectName("Model")
        self.tabWidget.addTab(self.Model, "")
        self.Train = QtWidgets.QWidget()
        self.Train.setObjectName("Train")
        self.tabWidget.addTab(self.Train, "")
        self.Evaluate = QtWidgets.QWidget()
        self.Evaluate.setObjectName("Evaluate")
        self.tabWidget.addTab(self.Evaluate, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)

        # connect action with UDF
        self.generate.clicked.connect(self.data_generate)


        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.Input.setTitle(_translate("Dialog", "Input"))
        self.label.setText(_translate("Dialog", "Data X"))
        self.data_x.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">./data/X</p></body></html>"))
        self.Blurring.setTitle(_translate("Dialog", "Blurring"))
        self.label_2.setText(_translate("Dialog", "Nibabel smoothing"))
        self.fwhm4.setText(_translate("Dialog", "FWHM=4"))
        self.fwhm5.setText(_translate("Dialog", "FWHM=5"))
        self.fwhm6.setText(_translate("Dialog", "FWHM=6"))
        self.fwhm7.setText(_translate("Dialog", "FWHM=7"))
        self.fwhm8.setText(_translate("Dialog", "FWHM=8"))
        self.generate.setText(_translate("Dialog", "Generate"))
        self.groupBox.setTitle(_translate("Dialog", "Adjacent Slice"))
        self.label_3.setText(_translate("Dialog", "slice of images:"))
        self.slice1.setText(_translate("Dialog", "1"))
        self.slice3.setText(_translate("Dialog", "3"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Data), _translate("Dialog", "Data"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Model), _translate("Dialog", "Model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Train), _translate("Dialog", "Train"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Evaluate), _translate("Dialog", "Evaluate"))

    def data_generate(self):

        # input
        path_x = self.data_x.toPlainText()

        # blur
        fwhm_kernel = []
        if self.fwhm4.isChecked():
            fwhm_kernel.append(4)
        if self.fwhm5.isChecked():
            fwhm_kernel.append(5)
        if self.fwhm6.isChecked():
            fwhm_kernel.append(6)
        if self.fwhm7.isChecked():
            fwhm_kernel.append(7)
        if self.fwhm8.isChecked():
            fwhm_kernel.append(8)

        # adjacent slice
        slice_x = 1
        if self.slice3.isChecked():
            slice_x = 3
        gbl_set_value("slice_x", slice_x)

        # generate
        list_X = glob.glob(path_x+'/*.nii')
        X, Y = blurring_data_generator(list_X, fwhm_kernel)


        QtWidgets.QMessageBox.information(self.generate, "test", list_X[0])
