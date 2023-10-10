#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the script to open ephyviewer for a list of recordings.
It will expect to find some csv organized as 'from_sniffer.csv'

Also, it depends on launch_ephyviewer, where we have a hard-coded path structure (expected to be the case in the NAS).

If sorting is selected, user will have to navigate to sorting folder!

Final note: experiment number will be by index (n-1), therefore experiment1 = block0

"""

__author__ = "Eduarda Centeno"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2023/10/1"  ### Date it was created
__status__ = (
    "Production"  ### Production = still being developed. Else: Concluded/Finished.
)


###########
# To do   #
###########
## add video!


####################
# Libraries        #
####################

# Standard imports

# Third party imports
import pandas as pd
from ephyviewer.myqt import QT
import pyqtgraph as pg
from ephyviewer.tools import ParamDialog

# Internal imports
from params_viz import path_to_database
from launch_ephyviewer import open_my_viewer


################################################################################
main_index = pd.read_csv(path_to_database)

display_columns = ["rec_name", "node", "experiment"]


class MainWindow(QT.QMainWindow):
    def __init__(
        self,
        parent=None,
    ):
        QT.QMainWindow.__init__(self, parent)

        self.resize(800, 1000)

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
        group = main_index.groupby("implant_name") # pay attention!
        for implant_name, index in group.groups.items():
            item = QT.QTreeWidgetItem([f"{implant_name}"])
            self.tree.addTopLevelItem(item)
            for key, row in main_index.loc[index].iterrows():
                text = " ".join("{}={}".format(k, row[k]) for k in display_columns)
                child = QT.QTreeWidgetItem([text])
                child.key = key
                item.addChild(child)

    def open_menu(self, position):
        indexes = self.tree.selectedIndexes()
        if len(indexes) == 0:
            return

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
            act = menu.addAction("Open viewer")
            act.key = items[0].key
            act.triggered.connect(self.open_viewer)

        menu.exec_(self.tree.viewport().mapToGlobal(position))

    def open_viewer(self):
        key = self.sender().key
        brain_area = main_index.loc[key, "brain_area"]
        implant_name = main_index.loc[key, "implant_name"]
        rec_name = main_index.loc[key, "rec_name"]
        node = main_index.loc[key, "node"]
        experiment = main_index.loc[key, "experiment"]

        print(implant_name, rec_name)

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

        w = open_my_viewer(
            brain_area,
            implant_name,
            rec_name,
            node,
            experiment,
            parent=self,
            **kwargs_streams,
        )

        w.show()
        w.setWindowTitle(implant_name + " " + rec_name)
        self.all_viewers.append(w)

        for w in [w for w in self.all_viewers if w.isVisible()]:
            self.all_viewers.remove(w)


if __name__ == "__main__":
    app = pg.mkQApp()
    w = MainWindow()
    w.show()
    app.exec_()
