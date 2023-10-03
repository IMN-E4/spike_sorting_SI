#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the script to launch ephyviewer.

"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2021/10/1"  ### Date it was created
__status__ = (
    "Production"  ### Production = still being developed. Else: Concluded/Finished.
)


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
from ephyviewer.myqt import QT
import pyqtgraph as pg
from ephyviewer.tools import ParamDialog
from launch_ephyviewer import open_my_viewer

# Internal imports ### (Put here imports that are related to internal codes from the lab)
from recording_list_NP import get_main_index


################################################################################

main_index = get_main_index()
display_columns = ['session_name', 'node', 'experiment_number']

class MainWindow(QT.QMainWindow) :
    def __init__(self, parent = None,):
        QT.QMainWindow.__init__(self, parent)

        self.resize(800,1000)

        self.mainWidget = QT.QWidget()
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QT.QHBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)

        self.tree = pg.widgets.TreeWidget.TreeWidget()
        self.tree.setAcceptDrops(False)
        self.tree.setDragEnabled(False)

        self.mainLayout.addWidget(self.tree)
        self.refresh_tree()

        self.tree.setContextMenuPolicy(QT.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.open_menu)

        self.all_viewers = []

    def refresh_tree(self):


        group = main_index.groupby('bird_name')
        for bird_name, index in group.groups.items():
            item  = QT.QTreeWidgetItem([f'{bird_name}'])
            self.tree.addTopLevelItem(item)
            for key, row in main_index.loc[index].iterrows():
                text = u' '.join('{}={}'.format(k, row[k]) for k in display_columns)
                child = QT.QTreeWidgetItem([text])
                child.key = key
                item.addChild(child)

    def open_menu(self, position):

        indexes = self.tree.selectedIndexes()
        if len(indexes) ==0: return

        items = self.tree.selectedItems()

        index = indexes[0]
        level = 0
        index = indexes[0]
        while index.parent().isValid():
            index = index.parent()
            level += 1
        menu = QT.QMenu()

        if level == 0:
            return
        elif level == 1:
            act = menu.addAction('Open viewer')
            act.key = items[0].key
            act.triggered.connect(self.open_viewer)
            # act = menu.addAction('Open LFP')
            # act.key = items[0].key
            # act.triggered.connect(self.open_viewer_lfp)

        menu.exec_(self.tree.viewport().mapToGlobal(position))

    

    def open_viewer(self):
        key = self.sender().key
        print(key)

        brain_area = main_index.loc[key, 'brain_area']
        bird_name = main_index.loc[key, 'bird_name']
        session_name = main_index.loc[key, 'session_name']
        node = main_index.loc[key, 'node']
        experiment = main_index.loc[key, 'experiment_number']

        print(bird_name, session_name)

        params = [
        {"name": "mic_spectrogram", "type": "bool", "value": True},
        {"name": "raw_recording", "type": "bool", "value": False},
        {"name": "bandpassed_recording", "type": "bool", "value": False},
        {"name": "viz_sorting", "type": "bool", "value": False},
        ]

        dia = ParamDialog(params, title="Select streams")
        if dia.exec_():
            kwargs_streams = dia.get()
        else:
            return

        w = open_my_viewer(brain_area, bird_name, session_name, node, experiment, parent=self, **kwargs_streams)

        w.show()
        w.setWindowTitle(bird_name+' '+session_name)
        self.all_viewers.append(w)

        for w in [w  for w in self.all_viewers if w.isVisible()]:
            self.all_viewers.remove(w)

    # def open_viewer_lfp(self):
    #     key = self.sender().key
    #     print(key)

    #     brain_area = main_index.loc[key, 'brain_area']
    #     bird_name = main_index.loc[key, 'bird_name']
    #     session_name = main_index.loc[key, 'session_name']
    #     node = main_index.loc[key, 'node']
    #     experiment = main_index.loc[key, 'experiment_number']

    #     print(bird_name, session_name)

    #     w = open_my_viewer(brain_area, bird_name, session_name, node, experiment, parent=self)

    #     w.show()
    #     w.setWindowTitle(bird_name+' '+session_name)
    #     self.all_viewers.append(w)

    #     for w in [w  for w in self.all_viewers if w.isVisible()]:
    #         self.all_viewers.remove(w)
    
    
if __name__ == '__main__':
    app = pg.mkQApp()
    w = MainWindow()
    w.show()
    app.exec_()
