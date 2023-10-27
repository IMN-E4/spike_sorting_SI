#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the script to open ephyviewer for a list of recordings.
It will expect to find some csv organized as 'from_sniffer.csv'

Also, it depends on launch_ephyviewer, where we have a hard-coded path structure (expected to be the case in the NAS).

If sorting is selected, user will have to navigate to sorting folder!

"""

__author__ = "Eduarda Centeno"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2023/10/1"
__status__ = "Production"


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
from launch_ephyviewer import open_my_viewer
from path_handling_viz import concatenate_available_sorting_paths
from utils import find_data_in_nas


################################################################################
recordings_index = find_data_in_nas(root_to_data="/nas")
display_columns = ["brain_area", "implant_name", "rec_name"]


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
        group = recordings_index.groupby("brain_area")  # pay attention!
        for brain_area, index in group.groups.items():
            item = QT.QTreeWidgetItem([f"{brain_area}"])
            self.tree.addTopLevelItem(item)
            for key, row in recordings_index.loc[index].iterrows():
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

        menu.exec(self.tree.viewport().mapToGlobal(position))

    # Open Ephyviewer
    def open_viewer(self):
        key = self.sender().key
        brain_area = recordings_index.loc[key, "brain_area"]
        implant_name = recordings_index.loc[key, "implant_name"]
        rec_name = recordings_index.loc[key, "rec_name"]

        ## add list of available sortings
        all_available_sortings = ["None"]
        all_available_sortings += concatenate_available_sorting_paths(
            brain_area, implant_name, rec_name
        )

        params = [
            {"name": "mic_spectrogram", "type": "bool", "value": True},
            {"name": "ap_recording", "type": "bool", "value": False},
            {"name": "lf_recording", "type": "bool", "value": False},
            {"name": "align_streams", "type": "bool", "value": False},
            {"name": "load_sync_channel", "type": "bool", "value": False},
            {"name": "order_by_depth", "type": "bool", "value": False},
            {
                "name": "available_sortings",
                "type": "list",
                "values": all_available_sortings,
            },  ## add here the list of available sortings + blank, if blank, no sortings.
        ]

        dia = ParamDialog(params, title="Select streams")
        if dia.exec():
            kwargs_streams = dia.get()
        else:
            return

        available_sortings = kwargs_streams.pop("available_sortings")
        if available_sortings == "None":
            kwargs_streams["viz_sorting"] = False
        else:
            kwargs_streams["viz_sorting"] = available_sortings

        w = open_my_viewer(
            brain_area=brain_area,
            implant_name=implant_name,
            rec_name=rec_name,
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
    app.exec()