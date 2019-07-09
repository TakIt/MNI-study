import tkinter
from tkinter import font
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import chainer
import chainer.links as cLinks
from chainer import serializers, Chain
import chainer.function as cFn

import sys

import nn

filename = "numfile.png"

window_width = 300
window_height = 300
canvas_width = 28 * 10
canvas_height = 28 * 10
button_width = 5
button_height = 1
BAR_SPACE = 3
BAR_WIDTH = 30
font=1

draw_depth = int(50. / 100. * 255)


class numNN(object):



    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y


    def on_dragged(self, event):
        self.canvas.create_line(self.sx, self.sy, event.x, event.y, width=5, tag="draw")
        self.draw.line(
            ((self.sx, self.sy), (event.x, event.y)),
            (draw_depth, draw_depth, draw_depth),
            int(window_width / 28 * 3)
        )
        self.sx = event.x
        self.sy = event.y


    def nnjudge(self):
        self.image1.save(filename)
        input_image=Image.open(filename)
        gray_image=ImageOps.grayscale(input_image)
        pr_resize = np.array(
            gray_image.resize((28, 28)).getdata()
        ).astype(np.float32)
        pr_resize /= 255.
        pr_resize = 1. - pr_resize

        y = self.mlp(pr_resize.reshape(1, 784))

        self.result.delete("result")  # clear the previous data
        self.val = []
        for i in range(10):
            self.val.append(
                max(np.array(y.data)[0][i], 0.) / np.max(np.array(y.data))
            )

        for i in range(10):

            self.result.create_text(
                15, i * BAR_WIDTH + BAR_SPACE + BAR_WIDTH / 2,
                text=str(i), tag="result"
            )
            self.result.create_text(
                window_width - 15,
                i * BAR_WIDTH + BAR_SPACE + BAR_WIDTH / 2,
                text=str("%.2f" % (self.val[i] / sum(self.val))),
                tag="result"
            )

    def resetcanvas(self):
        self.canvas.delete("draw")
        self.image1=Image.new("RGB", (window_width, window_height), (255,255,255))
        self.draw=ImageDraw.Draw(self.image1)
        self.result.delete("result")


    def windows(self):
        from tkinter import font
        root = tkinter.Tk()
        root.title("numberNN")
        root.geometry("300x300")

        label = tkinter.Label(root, text="入力された数字を推測します")
        label.pack(side="top")
        #font = font.Font(family='Helvetica', size=20, weight='bold')

        canvas_frame=tkinter.LabelFrame(
            root, bg="white",
            text="canvas",
            width=window_width, height=window_height,
            relief='groove', borderwidth=4
        )
        canvas_frame.pack(side=tkinter.LEFT)
        self.canvas = tkinter.Canvas(canvas_frame, bg="white",
                                     width=canvas_width, height=canvas_height,
                                     relief='groove', borderwidth=4)
        self.canvas.pack()
        quit_button = tkinter.Button(canvas_frame, text="exit",
                                     command=root.quit)
        quit_button.pack(side=tkinter.RIGHT)
        judge_button = tkinter.Button(canvas_frame, text="judge",
                                      width=button_width, height=button_height,
                                      command=self.judge)
        judge_button.pack(side=tkinter.LEFT)
        clear_button = tkinter.Button(canvas_frame, text="clear",
                                      command=self.clear)
        clear_button.pack(side=tkinter.LEFT)
        self.canvas.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas.bind("<B1-Motion>", self.on_dragged)

        c = tkinter.Canvas(root, bg="white", height=300, width=300)
        c.pack

        frame = tkinter.Frame(root)
        frame.pack()

        bottomframe = tkinter.Frame(root)
        bottomframe.pack(side=tkinter.BOTTOM)

        one = tkinter.Button(frame, text="1")
        one.pack(side=tkinter.LEFT)

        two = tkinter.Button(frame, text="2")
        two.pack(side=tkinter.LEFT)

        three = tkinter.Button(frame, text="3")
        three.pack(side=tkinter.LEFT)

        four = tkinter.Button(frame, text="4")
        four.pack(side=tkinter.LEFT)

        five = tkinter.Button(frame, text="5")
        five.pack(side=tkinter.LEFT)

        six = tkinter.Button(frame, text="6")
        six.pack(side=tkinter.LEFT)

        seven = tkinter.Button(frame, text="7")
        seven.pack(side=tkinter.LEFT)

        eight = tkinter.Button(frame, text="8")
        eight.pack(side=tkinter.LEFT)

        nine = tkinter.Button(frame, text="9")
        nine.pack(side=tkinter.LEFT)

        zero = tkinter.Button(frame, text="0")
        zero.pack(side=tkinter.LEFT)

        return root

    def _init_(self, modelName='20160818_MNIST.model'):
        self.modelName = modelName
        self.root = self.windows()

        self.image1 = Image.new("RGB", (window_width, window_height), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)

        self.mlp = nn.MLP(784, 1000, 10)
        model = cLinks.Classifier(self.mlp)
        serializers.load_hdf5(self.modelNamem, model)


    def run(self):
        self.root.mainloop()

def main():
    numNN().run()

if __name__ == '__main__':
    main()

