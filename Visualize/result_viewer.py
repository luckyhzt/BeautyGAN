import os
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from tkintertable import TableCanvas, TableModel
from PIL import Image, ImageTk
import pickle
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from threading import Timer


class Result_window(tk.Frame):

    def __init__(self, master):
        super(Result_window, self).__init__(master, width=900, height=600)
        self.grid_propagate(0)
        master.title('Result Viewer')
        master.resizable(height = False, width = False)
        self.btnFont = ('Microsoft YaHei UI', 12, 'bold')
        self.textFont = ('Microsoft YaHei UI', 13)

        # Plot parameters
        self.w = 750
        self.h = 500
        self.pressed = False
        self.drag = False
        self.press_x = 0 
        self.press_y = 0
        self.old_draw = []

        # Get all result folders
        thisDir = os.path.dirname(__file__)
        self.path = os.path.join(thisDir, 'Result')

        # Create table
        model = TableModel()
        self.table = TableCanvas(self, model=model, thefont=self.textFont, rowheight=26, editable=False)
        self.loadData()
        self.table.createTableFrame()
        self.updateTable()

        # Button
        self.UpdateBtn = tk.Button(self, text='Update', font=self.btnFont, bg='green', fg='white', command=self.updateAll)
        self.UpdateBtn.grid(column=1, row=3, padx=10)

        # Add binding
        self.display_win = tk.Toplevel(self)
        self.display_win.destroy()
        self.table.bind('<Button-1>', self.left_click)
        self.table.bind('<Button-3>', self.right_click)

    
    def left_click(self, event):

        def delete():
            self.display_win.destroy()
            directory = os.path.join(self.path, folder)
            for the_file in os.listdir(directory):
                file_path = os.path.join(directory, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
            os.rmdir(directory)
            self.updateAll()


        self.display_win.destroy()
        self.table.clearSelected()
        rclicked = self.table.get_row_clicked(event)
        cclicked = self.table.get_col_clicked(event)
        if rclicked == None or cclicked == None:
            return
        if 0 <= rclicked < self.table.rows and 0 <= cclicked < self.table.cols:
            self.table.setSelectedRow(rclicked)
            self.table.setSelectedCol(cclicked)
            self.table.drawSelectedRect(rclicked, cclicked)
            self.table.drawSelectedRow()
            self.table.tablerowheader.drawSelectedRows(rclicked)
        if rclicked >= self.table.rows or cclicked > 7 or cclicked < 1:
            return

        folder = self.table.model.getValueAt(rclicked, 0)
        x = self.table.winfo_rootx() + event.x
        y = self.table.winfo_rooty() + event.y + 15

        if cclicked == 5:
                img_array = np.load(os.path.join(self.path, folder, 'gen_imgs.npy'))
                img = Image.fromarray((img_array * 255).astype('uint8'), 'RGB')
                self.display_win = ImageDisplayWindow(img, img_array, x, y, init_w=700)

        if cclicked in (1, 2):
            with open(os.path.join(self.path, folder, 'config.pkl'), 'rb') as readFile:
                config = pickle.load(readFile)
            train_RMSE = np.load(os.path.join(self.path, folder, 'train_RMSE.npy'))
            test_RMSE = np.load(os.path.join(self.path, folder, 'test_RMSE.npy'))
            train_iter = np.arange(0, train_RMSE.shape[0] * config['log_step'], config['log_step'])
            test_iter = np.arange(0, test_RMSE.shape[0] * config['test_step'], config['test_step'])
            fig = Figure(figsize=(self.w/100, self.h/100), dpi=100, constrained_layout=True, linewidth=3, edgecolor='black')
            ax = fig.add_subplot(111)
            ax.set_title('Root Mean Squared Error')
            ax.set(xlabel='Iter', ylabel='RMSE')
            ax.plot(test_iter, test_RMSE, color='r', label='test')
            ax.plot(train_iter, train_RMSE, color='g', label='train')
            fig.legend(loc='upper right')
            self.display_win = GraphDisplayWindow(fig, ax, x, y)
        
        if cclicked == 3:
            with open(os.path.join(self.path, folder, 'config.pkl'), 'rb') as readFile:
                config = pickle.load(readFile)
            test_MAE = np.load(os.path.join(self.path, folder, 'test_MAE.npy'))
            test_iter = np.arange(0, test_MAE.shape[0] * config['test_step'], config['test_step'])
            fig = Figure(figsize=(self.w/100, self.h/100), dpi=100, constrained_layout=True, linewidth=3, edgecolor='black')
            ax = fig.add_subplot(111)
            ax.set_title('Mean Absolute Error')
            ax.set(xlabel='Iter', ylabel='MAE')
            ax.plot(test_iter, test_MAE, color='b')
            self.display_win = GraphDisplayWindow(fig, ax, x, y)
        
        if cclicked == 4:
            with open(os.path.join(self.path, folder, 'config.pkl'), 'rb') as readFile:
                config = pickle.load(readFile)
            test_PC = np.load(os.path.join(self.path, folder, 'test_PC.npy'))
            test_iter = np.arange(0, test_PC.shape[0] * config['test_step'], config['test_step'])
            fig = Figure(figsize=(self.w/100, self.h/100), dpi=100, constrained_layout=True, linewidth=3, edgecolor='black')
            ax = fig.add_subplot(111)
            ax.set_title('Pearson Correlation')
            ax.set(xlabel='Iter', ylabel='PC')
            ax.plot(test_iter, test_PC, color='b')
            self.display_win = GraphDisplayWindow(fig, ax, x, y)

        if cclicked == 6:
            with open(os.path.join(self.path, folder, 'config.pkl'), 'rb') as readFile:
                config = pickle.load(readFile)
            self.display_win = ConfigDisplayWindow(config, x, y)
        
        if cclicked == 7:
            self.display_win = tk.Toplevel(self, bg='seashell3')
            self.display_win.overrideredirect(True)
            self.display_win.geometry("%dx%d%+d%+d" % (200, 100, x, y))
            tk.Label(master=self.display_win, width=20, font=self.textFont, text='确认删除此记录？')\
                .grid(row=0, column=0, padx=0, pady=10, columnspan=2)
            tk.Button(self.display_win, text='取消', font=self.btnFont, bg='gray', fg='white', height=1, width=6,
                command=self.display_win.destroy).grid(row=1, column=0, padx=10, pady=10)
            tk.Button(self.display_win, text='确认', font=self.btnFont, bg='green', fg='white', height=1, width=6,
                command=delete).grid(row=1, column=1, padx=10, pady=10)


    def right_click(self, event):
        self.display_win.destroy()
        self.table.clearSelected()
        self.table.allrows = False
        rclicked = self.table.get_row_clicked(event)
        cclicked = self.table.get_col_clicked(event)

        if rclicked is None or cclicked is None:
            return

        if rclicked in range (0, self.table.rows) and cclicked in range (0, self.table.cols):
            self.table.setSelectedRow(rclicked)
            self.table.setSelectedCol(cclicked)
            self.table.drawSelectedRect(self.table.currentrow, self.table.currentcol)
            self.table.drawSelectedRow()
            self.table.tablerowheader.drawSelectedRows(rclicked)


    def updateAll(self):
        self.loadData()
        self.updateTable()

    
    def loadData(self):
        results = os.listdir(self.path)
        model = self.table.model
        rows = len(model.data)
        model.deleteRows(range(0, rows))

        model.addColumn('Folder')
        model.addColumn('Train_RMSE')
        model.addColumn('Test_RMSE')
        model.addColumn('Test_MAE')
        model.addColumn('Test_PC')
        model.addColumn('Mask')
        model.addColumn('Config')
        model.addColumn('Delete')

        if len(results) == 0:
            model.addRow(key='1', Folder='-', Train_RMSE='-', Test_RMSE='-', Test_MAE='-', Test_PC='-', Mask='-')
        else:
            for i, res in enumerate(results):
                res_path = os.path.join(self.path, res)
                train_RMSE = np.load(os.path.join(res_path, 'train_RMSE.npy'))[-1]
                test_RMSE = np.load(os.path.join(res_path, 'test_RMSE.npy'))[-1]
                test_MAE = np.load(os.path.join(res_path, 'test_MAE.npy'))[-1]
                test_PC = np.load(os.path.join(res_path, 'test_PC.npy'))[-1]
                model.addRow(key=i, Folder=res, Train_RMSE='{:.4f}'.format(train_RMSE), Test_RMSE='{:.4f}'.format(test_RMSE), 
                            Test_MAE='{:.4f}'.format(test_MAE), Test_PC='{:.4f}'.format(test_PC), Mask='View', Config='Detail',
                            Delete='Delete')


    def updateTable(self):
        self.table.adjustColumnWidths()
        self.table.autoResizeColumns()
        self.table.redraw()




class ImageDisplayWindow(tk.Toplevel):

    def __init__(self, img, img_array, x, y, init_w, *args, **kwargs):
        super(ImageDisplayWindow, self).__init__(*args, **kwargs)
        self.overrideredirect(True)
        self.img = img
        self.img_array = img_array

        self.ratio = float(self.img.size[0] / self.img.size[1])
        init_h = int(init_w / self.ratio)
        self.geometry("%dx%d%+d%+d" % (init_w, init_h, x, y))

        self.fig = Figure(figsize=(init_w/100, init_h/100), dpi=100, constrained_layout=True, linewidth=3, edgecolor='black')
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.img)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        grip = ttk.Sizegrip(self)
        grip.place(relx=1.0, rely=1.0, anchor="se")
        self.canvas.mpl_connect('button_press_event', self.click)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('motion_notify_event', self.onMotion)
        self.drag_win = False
        self.press_x = 0
        self.press_y = 0
        self.old_draw = []
        self.clicked = False
        self.timer = None


    def click(self, event):
        for d in self.old_draw:
            d.remove()
        self.old_draw = []

        if event.button == 1:
            for d in self.old_draw:
                    d.remove()
            self.old_draw = []
            self.clicked = True
            self.press_x = self.winfo_pointerx() - self.winfo_rootx()
            self.press_y = self.winfo_pointery() - self.winfo_rooty()
            self.timer = Timer(0.15, self.set_drag)
            self.timer.start()

                
    def release(self, event):
        if self.timer != None:
            self.timer.cancel()
            self.timer = None
        
        if not self.drag_win and event.xdata != None and event.ydata != None and self.clicked == True:
            cur_ylim = self.ax.get_ylim()
            cur_xlim = self.ax.get_xlim()
            x_range = cur_xlim[1] - cur_xlim[0]
            y_range = np.absolute(cur_ylim[1] - cur_ylim[0])
            pad_x = 0.07 * x_range
            pad_y = 0.07 * y_range
            x = int(event.xdata)
            y = int(event.ydata)
            y = y % 234

            if cur_xlim[1] - x >= 0.2*x_range:
                text_x = x + pad_x
            else:
                text_x = x - pad_x

            for this_y in (y, y+234):
                if cur_ylim[0] - this_y >= 0.2*y_range:
                    text_y = this_y + pad_y
                else:
                    text_y = this_y - pad_y
                value = self.img_array[this_y, x, :]
                str_value = '%.4f\n%.4f\n%.4f' % (value[0], value[1], value[2])
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                self.old_draw.append(self.ax.text(text_x, text_y, str_value, bbox=props))
                self.old_draw.append(self.ax.annotate("", xy=(x, this_y), xytext=(text_x, text_y), arrowprops=dict(arrowstyle="-")))
            
            self.canvas.draw()

        self.drag_win = False
        

    def onMotion(self, event):
        if self.drag_win:
            x = self.winfo_pointerx() - self.press_x
            y = self.winfo_pointery() - self.press_y
            self.geometry('+{x}+{y}'.format(x=x, y=y))
    

    def set_drag(self):
        self.drag_win = True



class GraphDisplayWindow(tk.Toplevel):

    def __init__(self, fig, ax, x, y, *args, **kwargs):
        super(GraphDisplayWindow, self).__init__(*args, **kwargs)
        self.overrideredirect(True)
        self.focus_set()

        # Figure
        self.fig = fig
        self.ax = ax
        # Size and position
        self.x = x
        self.y = y
        self.w = 780
        self.h = 520
        # Used for event trigger
        self.drag_graph = False
        self.drag_win = False
        self.clicked = False
        self.press_x = 0 
        self.press_y = 0
        self.old_draw = []
        self.timer = None
        # Draw graph
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.geometry("%dx%d%+d%+d" % (self.w, self.h, self.x, self.y))
        grip = ttk.Sizegrip(self)
        grip.place(relx=1.0, rely=1.0, anchor="se")
        # Connect event
        self.canvas.mpl_connect('scroll_event', self.zoom_fun)
        self.canvas.mpl_connect('button_press_event', self.click)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('motion_notify_event', self.onMotion)

    
    def zoom_fun(self, event):
        # Delete old draw and clear
        for d in self.old_draw:
            d.remove()
        self.old_draw = []
        base_scale = 1.5
        # get the current x and y limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        self.ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        self.ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        self.canvas.draw() # force re-draw

    def click(self, event):
        if event.button == 1:
            if event.xdata == None or event.ydata == None:
                self.drag_win = True
                self.press_x = self.winfo_pointerx() - self.winfo_rootx()
                self.press_y = self.winfo_pointery() - self.winfo_rooty()
            else:
                for d in self.old_draw:
                    d.remove()
                self.old_draw = []
                self.clicked = True
                self.press_x = event.xdata
                self.press_y = event.ydata
                self.timer = Timer(0.15, self.set_drag)
                self.timer.start()
                
    def set_drag(self):
        self.drag_graph = True

    def release(self, event):
        if self.timer != None:
            self.timer.cancel()
            self.timer = None

        if not self.drag_graph and event.xdata != None and event.ydata != None and self.clicked == True:
            cur_ylim = self.ax.get_ylim()
            cur_xlim = self.ax.get_xlim()
            x_range = cur_xlim[1] - cur_xlim[0]
            y_range = cur_ylim[1] - cur_ylim[0]
            text_pad_x = 0.02 * x_range
            text_pad_y = 0.02 * y_range
            # Get x axis value
            x_axis = self.ax.lines[0].get_data()[0]
            _, x_selected = self.nearest_value(x_axis, event.xdata)
            # Draw
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            for line in self.ax.lines:
                data = line.get_data()
                i, x_this = self.nearest_value(data[0], x_selected)
                self.old_draw.append(self.ax.text(x_this+text_pad_x, data[1][i]+text_pad_y, '{:.4f}'.format(data[1][i]), bbox=props))

            self.old_draw.append(self.ax.add_line(matplotlib.lines.Line2D([x_selected, x_selected], [cur_ylim[0], cur_ylim[1]],\
                                             linewidth = 1.0, color = 'gray')))
            self.old_draw.append(self.ax.text(x_selected+text_pad_x, cur_ylim[0]+text_pad_y, str(x_selected), bbox=props))
            # Update
            self.canvas.draw()

        self.clicked = False
        self.drag_graph = False
        self.drag_win = False
        
    def onMotion(self, event):
        if self.drag_graph:
            dx = event.xdata - self.press_x
            dy = event.ydata - self.press_y
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            # set new limits
            self.ax.set_xlim([cur_xlim[0] - dx, cur_xlim[1] - dx])
            self.ax.set_ylim([cur_ylim[0] - dy, cur_ylim[1] - dy])
            self.canvas.draw() # force re-draw
        if self.drag_win:
            x = self.winfo_pointerx() - self.press_x
            y = self.winfo_pointery() - self.press_y
            self.geometry('+{x}+{y}'.format(x=x, y=y))


    def nearest_value(self, array, value):
        diff = np.abs(array - value)
        index = np.argmin(diff)
        return index, array[index]



class ConfigDisplayWindow(tk.Toplevel):

    def __init__(self, config, x, y, *args, **kwargs):
        super(ConfigDisplayWindow, self).__init__(*args, **kwargs)
        self.overrideredirect(True)
        self.geometry("%dx%d%+d%+d" % (500, 300, x, y))

        self.press_x = 0
        self.press_y = 0
        self.bind('<Button-1>', self.click)
        self.bind('<B1-Motion>', self.onMotion)

        self.textFont = ('Microsoft YaHei UI', 16)
        self.label = tk.Label(self, text='loading...', anchor='w', justify='left', font=self.textFont, width=20)
        self.label.pack(fill='both')
        self.config = config
        self.showConfig()


    def showConfig(self):
        text =  'Train samples: %d / %d \n\n' \
                'Input size: %dx%d from %dx%d \n\n' \
                'Batch size: %d \n\n' \
                'Learning rate: %.4f \n(decrease by %.2f every %d epoch) \n\n' \
                'Weight decay: %f' \
                % (self.config['train_samples'], self.config['train_samples']+self.config['test_samples'],
                self.config['crop_size'], self.config['crop_size'], self.config['img_size'], self.config['img_size'],
                self.config['batch_size'],
                self.config['lr'], self.config['lr_decay'], self.config['lr_decay_epoch'],
                self.config['weight_decay'])
        self.label.config(text=text)


    def click(self, event):
        self.press_x = event.x
        self.press_y = event.y

    
    def onMotion(self, event):
        x = self.winfo_pointerx() - self.press_x
        y = self.winfo_pointery() - self.press_y
        self.geometry('+{x}+{y}'.format(x=x, y=y))






if __name__ == '__main__':
    root = tk.Tk()
    window = Result_window(root)
    window.pack(side="top", fill="both", expand="true", padx=4, pady=4)
    root.mainloop()

