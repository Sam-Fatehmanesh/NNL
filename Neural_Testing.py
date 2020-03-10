import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import linalg, misc
import math
import os
clear = lambda: os.system('clear')
linalg.init()


class properties:
    def __init__(self):
        self.sig = "sig"
        self.relu = "relu"
        self.gradientdescent = "gradientdescent"
        self.naturalselection = "naturalselection"

class NeuralNet:
    def __init__(self,inputcount,trainIn,trainOut):
        self.trainindata = trainIn
        self.trainoutdata = trainOut
        self.inputcount = inputcount
        self.layerprimus = None

    def addLayer(self,size,func):
        
        if self.layerprimus == None:
            self.layerprimus = Layer(func,self.inputcount,size)
        else:
            self.layerprimus.addlayer(size,func)
        print("added new ", func," layer")

    def compute(self,indata):
        return self.layerprimus.fowardrun(indata)

    def correct(self,outdata,learnrate,momentum):
        self.layerprimus.update(outdata,learnrate,momentum)

    def train(self,epoch,learnrate,momentum):
        for i in range(epoch):
            for o in range(len(self.trainindata)):
                test = self.compute(self.trainindata[o])
                self.correct(self.trainoutdata[o],learnrate,momentum)
                if int(i % (epoch/100)) == 0:
                    self.diagnostics(self.trainoutdata[o],test)
        print("traing done")

    def error(self):
        error = 0.0
        for o in range(len(self.trainindata)):
            out = self.compute(self.trainindata[o])
            for i in range(out.size-1):
                error += (out[i][0]-self.trainoutdata[o][i])**2 
        print("error = ", error/2)

    def diagnostics(self,correct = "",output = ""):
        clear()
        self.layerprimus.interprint()
        print(correct)
        print(output)
        
class Layer:
    def __init__(self,func,inputcount,count = 0):
        self.inputcount = inputcount
        self.count = count
        self.func = func
        self.biasStart = 1.0
        #set recusive layer
        self.adjlayer = None
        self.isadjlayer = False
        #the neurons are stored in a matrix
        self.neurons = None
        #the momentum terms have their own matrix
        self.mterm = None
        #the matrix is set if the layer is ready
        if count != 0:
            #sets weights to inbetween -1 and 1
            self.neurons = matrix(count+1,inputcount+1,True)
            self.neurons = (self.neurons * 2.0) - 1.0
            self.mterm = matrix(count+1,inputcount+1)
            #these 'for' loops format the matrix to work for weights and neurons 
            for i in range(inputcount):
                self.neurons[count][i] = 0.0
            self.neurons[count][inputcount] = 1.0
        self.output = None
        self.input = None
        self.Z = None

    def addlayer(self,size,func):
        if self.isadjlayer == False:
            self.isadjlayer = True
            self.adjlayer = Layer(func,self.count,size)
        else:
            self.adjlayer.addlayer(size,func)

    def fowardrun(self,indata): 
        self.input = indata
        if self.isadjlayer == False:
            self.output = matrixdot(self.neurons,indata)
            self.Z = self.output
            for i in range(self.count):
                self.output[i][0] = actfunc(self.func,self.output[i][0])
            return self.output
        else:
            self.output = matrixdot(self.neurons,indata)
            self.Z = self.output
            for i in range(self.count):
                self.output[i][0] = actfunc(self.func,self.output[i][0])
            return self.adjlayer.fowardrun(self.output)

    def update(self,trainout,learnrate,momentum): 
        if self.isadjlayer == False:
            #find the cost derivitive with cost and output 2 time error
            costD = []
            for i in range(self.count):
                costD[i] = 2 * (self.output[i][0]-trainout[i])
            #finds the derivitive of the act func reletive to the neuron output
            ActD = []
            for i in range(self.count):
                ActD[i] = actfunc(self.func,self.Z[i][0],True)
            prevAct = self.output
            
            
        
            

    def interprint(self):
        print(self.neurons)
        if self.isadjlayer == False:
            return
        self.adjlayer.interprint()

#function definitions used by the classes
#turns a matrix of one collum into a sqaure matrix with values in diagonal pattern
def squarenpary(ary):
    new = matrix(ary.shape[0],ary.shape[0])
    for i in range(ary.shape[0]):
        new[i][i] = ary[i]
    return new
#scales matricies by a number
def scalemultiply(x,y):
    r = gpuarray.to_gpu(y)
    linalg.scale(x,r)
    return r.get()
#returns a matrix custum made
def matrix(x, y,isrand = False,ary = []):
    if isrand:
        return np.random.rand(x, y).astype(np.float)
    elif islistNull(ary) == False:
        return np.array(ary).astype(np.float)
    else:
        return np.zeros(shape = (x,y)).astype(np.float)
#GPU dot multiplicaiton of matricies
def matrixdot(x,y):
    X = gpuarray.to_gpu(x)
    Y = gpuarray.to_gpu(y)
    D = linalg.dot(X,Y)
    return D.get()
#returns a matrix in the GPU
def gpuary(x, y,isrand = False,ary = []):
    if isrand:
        return gpuarray.to_gpu(np.random.rand(x, y).astype(np.float))
    elif ~islistNull(ary):
        return gpuarray.to_gpu(np.array(ary).astype(np.float))
    else:
        return gpuarray.to_gpu(np.zeros(shape = (x,y)).astype(np.float))
#returns weather or not a list, matrix, or array is empty
def islistNull(list1):
    #finds if a list is empty
    if len(list1) == 0:
        return True
    else:
        return False
#an activation function that can be of choice provided and can be its derivitive
def actfunc(func,x,d=False):
    if func == "sig":
        return sigfunc(x,d)
    if func == "relu":
        return relufunc(x,d)
#the sigmoid funciton with choice of its derivitive
def sigfunc(x,d = False):
    num = 1.0 / (1.0+math.e**(-x))
    if d == False:
        return num
    else:
        return num * (1.0-num)
#the relu funciton with choice of its derivitive
def relufunc(x,d = False):
    if d == False:
        return max(0.0,x)
    else:
        if x >= 0.0:
            return 1.0
        else:
            return 0.0