from ephyviewer.myqt import QT

import pyqtgraph as pg

from recording_list_CN import get_main_index


from launch_ephyviewer import open_my_viewer

# from mainviewer import get_viewer_from_run_key
# from dataio import dataio, get_main_index
#~ from states_encoder import get_encoder


main_index = get_main_index()


# display_columns = ['bird_name', 'session_name', 'node']
display_columns = ['session_name', 'node']



class MainWindow(QT.QMainWindow) :
    def __init__(self, parent = None,):
        QT.QMainWindow.__init__(self, parent)

        self.resize(400,600)

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

        menu.exec_(self.tree.viewport().mapToGlobal(position))

    

    def open_viewer(self):
        key = self.sender().key
        print(key)

        bird_name = main_index.loc[key, 'bird_name']
        session_name = main_index.loc[key, 'session_name']

        print(bird_name, session_name)

        w = open_my_viewer(bird_name, session_name, parent=self)

        w.show()
        w.setWindowTitle(bird_name+' '+session_name)
        self.all_viewers.append(w)

        for w in [w  for w in self.all_viewers if w.isVisible()]:
            self.all_viewers.remove(w)
    
    
if __name__ == '__main__':
    app = pg.mkQApp()
    w = MainWindow()
    w.show()
    app.exec_()
