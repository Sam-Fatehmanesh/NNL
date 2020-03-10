#change Neural_CPU to the CUDAGPU one if you want but you need the libraries imported in the Neural CUDAGPU version
from Neural_CPU import NeuralNet, properties, np

#traindata
#the last ones in this are just for the matrix multiplication
C1 = np.array([[1],[1],[1.0]])
C2 = np.array([[1],[0],[1.0]])
C3 = np.array([[0],[1],[1.0]])
C4 = np.array([[0],[0],[1.0]])

R1 = np.array([1])
R2 = np.array([1])
R3 = np.array([1])
R4 = np.array([0])


trainIn = [C1,C2,C3,C4]
trainOut = [R1,R2,R3,R4]

#properties
props = properties()

#configuration
Neuralnetwork = NeuralNet(2,trainIn,trainOut,True)
Neuralnetwork.addLayer(2,props.sig)
Neuralnetwork.addLayer(1,props.sig)

#train and testq
Neuralnetwork.train(10000,.0125,.2)
print(Neuralnetwork.compute(C1))
print(Neuralnetwork.compute(C2))
print(Neuralnetwork.compute(C3))
print(Neuralnetwork.compute(C4))
Neuralnetwork.error()
Neuralnetwork.diagnostics()