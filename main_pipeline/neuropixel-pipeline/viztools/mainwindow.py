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
## add video to Ephyvivwer?!
## finish song sorting integration


####################
# Libraries        #
####################

# Standard imports
from pathlib import Path


# Third party imports
from ephyviewer.myqt import QT
import pyqtgraph as pg
from ephyviewer.tools import ParamDialog
import spikeinterface_gui
import spikeinterface.full as si

# Internal imports
from launch_ephyviewer import open_my_viewer
from path_handling_viz import concatenate_available_sorting_paths
from utils import find_data_in_nas


##### for testing
# import pandas as pd
# recordings_index = pd.read_csv('/home/eduarda/python-related/github-repos/spike_sorting_with_samuel/main_pipeline/neuropixel-pipeline/viztools/testing.csv',
#                                index_col=0)

################################################################################
recordings_index = find_data_in_nas(root_to_data="/nas")
display_columns = ["rec_name"]


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
        group_brain_area = recordings_index.groupby("brain_area")  # pay attention!
        group_both = recordings_index.groupby(["brain_area", "implant_name"])
        for brain_area, _ in group_brain_area.groups.items():
            item_brain_area = QT.QTreeWidgetItem([f"{brain_area}"])
            self.tree.addTopLevelItem(item_brain_area)

            for area_and_implant, index2 in group_both.groups.items():
                area = area_and_implant[0]
                implant_name = area_and_implant[1]
                if area == brain_area:
                    item_implant = QT.QTreeWidgetItem([f"{implant_name}"])
                    item_implant.key = implant_name
                    item_brain_area.addChild(item_implant)

                    for rec_index, row in recordings_index.loc[index2].iterrows():
                        text = " ".join("{}={}".format(k, row[k]) for k in display_columns)
                        child = QT.QTreeWidgetItem([text])
                        child.key = rec_index
                        item_implant.addChild(child)

    def open_menu(self, position):
        indexes = self.tree.selectedIndexes()
        if len(indexes) == 0:
            return

        items = self.tree.selectedItems()
        index = indexes[0]
        level = 0
        while index.parent().isValid():
            index = index.parent()
            level += 1
        
        menu = QT.QMenu()

        if level == 0:
            return
        elif level == 1:
            # Option 1
            act_level_1 = menu.addAction("Open Song Sorting")
            act_level_1.key = items[0].key
            act_level_1.triggered.connect(self.open_song_sorting)

        elif level == 2:
            # Option 2
            act_level_2 = menu.addAction("Open Ephyviewer")
            act_level_2.key = items[0].key
            act_level_2.triggered.connect(self.open_ephyviewer)

            # Option 3
            act_level_2 = menu.addAction("Open SIGUI")
            act_level_2.key = items[0].key
            act_level_2.triggered.connect(self.open_sigui_viewer)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    # Open Song Sorting
    def open_song_sorting(self):
        implant_name = self.sender().key
        paths = recordings_index[recordings_index.implant_name == implant_name].apply(lambda x :'/'.join(x.astype(str)),1)
        paths_of_available_recs = [Path('/' + path) for path in paths]
        available_recs = [rec_name.stem for rec_name in paths_of_available_recs]
        main_path = paths_of_available_recs[0].parent


        # perhaps add step where person can find the threshold on the fly...

        # Create ParamDialog for Manual sorting and Canapy config
        params_config = [
            {"name": "threshold", "type": "float", "value": 0.01}
            # can add here some stuff to change canapy's config            
        ]
        
        dia_config = ParamDialog(params_config, title="Select params")
        dia_config.resize(800,1000)
        if dia_config.exec():
            kwargs_config = dia_config.get() ## to be used later!
        else:
            return

        # Create ParamDialog for Manual sorting and Canapy config
        if len(available_recs) == 0:
            print('No recordings available to select!')
            return
        else:
            params_recs = [{"name": rec_name, "type": "list", "values": ["train", "predict"]} for rec_name in available_recs]

        dia_recs = ParamDialog(params_recs, title="Select recordings")
        dia_recs.resize(800,1000)
        if dia_recs.exec():
            kwargs_recs = dia_recs.get()
        else:
            return

        recs_for_manual_labelling = [main_path / key for key, item in kwargs_recs.items() if item == 'train']
        recs_for_predicting = [main_path / key for key, item in kwargs_recs.items() if item == 'predict']
        
        # print('to train', recs_for_manual_labelling)
        # print('to predict', recs_for_predicting)
        # kwargs_recs["recs_for_manual_labelling"] = recs_for_manual_labelling
        # kwargs_recs["recs_for_predicting"] = recs_for_predicting

        # print(kwargs_recs)
        
        # ## check that nidqs indeed exist by looking for bin and meta
        #     assert [len(list(Path(rec).glob('*nidq*')))==2 for rec in recs_for_manual_labelling], 'Cannot find NIDQ in selected recs for manual labelling'
        #     assert [len(list(Path(rec).glob('*nidq*')))==2 for rec in recs_for_predicting], 'Cannot find NIDQ in selected recs for predicting'

        ## here I need to decide how to integrate with Manual labelling script
            
        # w = open_manual_labelling(
        #     parent=self,  # this needs to be there!
        #     **kwargs_streams,
        # )
        
        ## after manual labelling is finished, trigger canapy
        # w.show(kwargs_streams)
        # w.setWindowTitle(implant_name)
        # self.all_viewers.append(w)

        # for w in [w for w in self.all_viewers if w.isVisible()]:
        #     self.all_viewers.remove(w)


    # Open Ephyviewer
    def open_ephyviewer(self):
        rec_index = self.sender().key
        brain_area = recordings_index.loc[rec_index, "brain_area"]
        implant_name = recordings_index.loc[rec_index, "implant_name"]
        rec_name = recordings_index.loc[rec_index, "rec_name"]

        # Add list of available sortings
        all_available_sortings = ["None"]
        all_available_sortings += concatenate_available_sorting_paths(
            brain_area, implant_name, rec_name, target="NAS"
        )

        # Create params Dialog
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
            },
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
            parent=self,  # this needs to be there!
            **kwargs_streams,
        )

        w.show()
        w.setWindowTitle(implant_name + " " + rec_name)
        self.all_viewers.append(
            w
        )

        for w in [w for w in self.all_viewers if w.isVisible()]:
            self.all_viewers.remove(w)

    def open_sigui_viewer(self):
        rec_index = self.sender().key
        brain_area = recordings_index.loc[rec_index, "brain_area"]
        implant_name = recordings_index.loc[rec_index, "implant_name"]
        rec_name = recordings_index.loc[rec_index, "rec_name"]

        ## add list of available sortings
        all_available_sortings = ["None"]
        all_available_sortings += concatenate_available_sorting_paths(
            brain_area, implant_name, rec_name, target="CACHE"
        )
        params = [
            {
                "name": "available_sortings",
                "type": "list",
                "values": all_available_sortings,
            }
        ]

        dia = ParamDialog(params, title="Select sorting")
        if dia.exec():
            kwargs_streams = dia.get()
        else:
            return

        wf_folder = kwargs_streams.pop("available_sortings")
        print('this is the folder', wf_folder)

        we = si.WaveformExtractor.load_from_folder(wf_folder)
        w = spikeinterface_gui.MainWindow(we, parent=self)

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