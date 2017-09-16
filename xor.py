import numpy as np
import matplotlib.pyplot as plot

INPUT_NODES = 2       
HIDDEN_NODES = 3       
OUTPUT_NODES = 1

MAX_ITER = 30000
ALPHA = .3

print ("XOR Neural Learning")

class xor(object):
    def __init__(self): # __init__ methood is used for instantiating constants and varialbles
        self.inputLayerSize = INPUT_NODES
        self.outputLayerSize = OUTPUT_NODES
        self.hiddenLayerSize = HIDDEN_NODES

        #Weights (Parameters)
        self.W1 = np.random.random((self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.random((self.hiddenLayerSize, self.outputLayerSize))

    def forward(self, X):           			# X matrix is 4X2 matrix
        self.z2 = np.dot(X, self.W1)			#
        self.a2 = self.sigmoid(self.z2)			#
        self.z3 = np.dot(self.a2, self.W2)		#
        yHat = self.sigmoid(self.z3)			#
        return yHat

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def sigmoidPrime(self,z):
        return np.exp(-z)/((1 + np.exp(-z))**2)

    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y - self.yHat)**2)
        #print J
        return J

    def costFunctionPrime(self, X, y):

        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2


#---------------------------------NUMERICAL GRADIENT CHECKING------------------------------------


# Formula = numericalGradient = (f(x+epsilon) - f(x-epsilon))/(2*epsilon)




a = xor()
X = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[0],[1],[1],[0]]);
# Training Sets

#Iterations
for i in range(30000):
    #yHat = a.forward(X)
    cost1 = a.costFunction(X,y)
    dJdW1, dJdW2 = a.costFunctionPrime(X,y)
    a.W1 = a.W1 - ALPHA*dJdW1
    a.W2 = a.W2 - ALPHA*dJdW2
    
print("Input:")
print(X)
print("Output Predictions:")
print (a.yHat)




