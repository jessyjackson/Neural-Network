from layers.layer import Layer
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.lossPrime = None
    def add(self, layer):
        self.layers.append(layer)
    def setLoss(self, loss, lossPrime):
        self.loss = loss
        self.lossPrime = lossPrime

    def predict(self, input):
        samples = len(input)
        result = []
        for i in range(samples):
            output = input[i]

            for layer in self.layers:
                output = layer.forwardPropagation(output)
            result.append(output)

        return result
    
    def train(self,xTrain,yTrain,epochs,learningRate):
        samples = len(xTrain)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = xTrain[j]
                for layer in self.layers:
                    output = layer.forwardPropagation(output)
                err += self.loss(yTrain[j],output)
                error = self.lossPrime(yTrain[j],output)
                for layer in reversed(self.layers):
                    error = layer.backwardPropagation(error,learningRate)
            err /= samples
            print('epoch %d/%d   error=%f' %(i+1,epochs,err))
    