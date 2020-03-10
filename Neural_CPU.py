import numpy as np
import math
import os
clear = lambda: os.system('clear')

class properties:
    def __init__(self):
        self.sig = "sig"
        self.relu = "relu"
        self.gradientdescent = "gradientdescent"
        self.genetic = "genetic"

class NeuralNet:
    def __init__(self,inputcount,trainIn,trainOut,diag):
        self.diag = diag
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
                if (self.diag and int(i % (epoch/100)) == 0):
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
        #clear()
        self.layerprimus.interprint()
        print(correct)
        print(output)
        
class Layer:
    def __init__(self,func,inputcount,count = 0):
        self.inputcount = inputcount
        self.count = count
        self.func = func
        self.biasStart = 0
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

    def updateneurons(self,learnrate,momentum,ActD):
        for i in range(self.count):
            for o in range(self.inputcount):
                change = (-learnrate * ActD[i][0] * self.input[o][0])
                self.mterm[i][o] = change + (self.mterm[i][o] * momentum)
                self.neurons[i][o] += self.mterm[i][o]

    def updatebias(self,learnrate,momentum,ActD):
        for i in range(self.count):
            change = (-learnrate * ActD[i][0])
            self.mterm[i][self.inputcount] = change + (self.mterm[i][self.inputcount] * momentum)
            self.neurons[i][self.inputcount] +=  self.mterm[i][self.inputcount]

    def updateActD(self,ActD,IcostD):
        ActD[self.count][0] = 1.0
        for i in range(self.count):
            ActD[i][0] = actfunc(self.func,self.output[i][0],True)
        #multiplies the activation derivitives by the derivitive of cost to output
        ActD = matrixdot(squarenpary(IcostD),ActD)
        ActD[self.count][0] = 1.0
        return ActD

    def updateD(self,ActD,currentD):
        #finish adding return cost to act part and finish other condition of upper if
        for i in range(self.inputcount):
            num = 0.0
            for o in range(self.count):
                num += ActD[o][0] * self.neurons[o][i]
            currentD[i][0] = num * self.input[i][0]
        return currentD

    def update(self,trainout,learnrate,momentum): 
        currentD = matrix(self.inputcount+1,1)
        currentD[self.inputcount][0] = 1.0
        if self.isadjlayer == False:
            #finds cost aka error for each neuron
            IcostD = matrix(self.count+1,1)
            IcostD[self.count][0] = 1
            for i in range(self.count):
                IcostD[i][0] = (self.output[i][0]-trainout[i]) * 2
            #finds the derivitrive of the activativation function with output in it
            ActD = matrix(self.count+1,1)
            ActD = self.updateActD(ActD,IcostD)
            #updating neuron's dependng on their individual inputs
            self.updateneurons(learnrate,momentum,ActD)
            #updates bias of each neuron
            self.updatebias(learnrate,momentum,ActD)
            #creating currentD to return for previouse layer of neuons
            currentD = self.updateD(ActD,currentD)
            #returns costtoactivation partial derivitive of back neurons 
            return currentD
        else:
            #passes correct train data foward and gets 
            #the parcial derivive of cost over layer output
            IcostD = self.adjlayer.update(trainout,learnrate,momentum)
            #finds the derivitrive of the activativation function with output in it
            ActD = matrix(self.count+1,1)
            ActD = self.updateActD(ActD,IcostD)
            #updating neuron's dependng on their individual inputs
            self.updateneurons(learnrate,momentum,ActD)
            #updates bias of each neuron
            self.updatebias(learnrate,momentum,ActD)
            #creating currentD to return for previouse layer of neuons
            currentD = self.updateD(ActD,currentD)
            #returns costtoactivation partial derivitive of back neurons 
            return currentD

    def interprint(self):
        print(self.neurons)
        if self.isadjlayer == False:
            return
        self.adjlayer.interprint()

#function definitions used by the classes
#turns a matrix of one collum into a sqaure matrix with values in diagonal pattern
def squarenpary(ary):
    new = matrix(ary.shape[0], ary.shape[0])
    for i in range(ary.shape[0]):
        new[i][i] = ary[i]
    return new
#scales matricies by a number
def scalemultiply(x,y):
    return x * y
#returns a matrix custom made
def matrix(x, y,isrand = False,ary = []):
    if isrand:
        return np.random.rand(x, y).astype(np.float)
    elif islistNull(ary) == False:
        return np.array(ary).astype(np.float)
    else:
        return np.zeros(shape = (x,y)).astype(np.float)
#CPU dot multiplicaiton of matricies
def matrixdot(x,y):
    return np.dot(x,y)
#returns wheather or not a list, matrix, or array is empty
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
def relufunc(x, d = False):
    if d == False:
        return max(0.0,x)
    else:
        if x >= 0.0:
            return 1.0
        else:
            return 0.0