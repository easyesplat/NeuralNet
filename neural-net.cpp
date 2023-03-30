#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

struct Connection
{
   double weight;
   double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

class Neuron
{
public:
   Neuron(int numOutputs, int myNeurIndex);
   void feedForward(const Layer &prevLayer);
   void calcOutputGradients(double targetVal);
   void calcHiddenGradients(const Layer &nextLayer);
   void updateInputWeights(Layer &prevLayer);
   void setOutputVal(double outputVal);
   double getOutputVal(void) const;
private:
   static double eta;
   static double alpha;
   static double transferFunction(double x);
   static double transferFunctionDerivative(double x);
   static double randomWeight(void) { return rand() / double(RAND_MAX); }
   double sumDOW(const Layer &nextLayer) const;
   double m_outputVal;
   vector<Connection> m_outputWeights;
   int m_myNeurIndex;
   double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(int numOutputs, int myNeurIndex)
{
   for (int i = 0; i < numOutputs; i++) {
      m_outputWeights.push_back(Connection());
      m_outputWeights.back().weight = randomWeight();
   }
   m_myNeurIndex = myNeurIndex;
}

double Neuron::transferFunction(double x)
{
   return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
   return 1.0 - x*x;
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
   double sum = 0.0;

   for (int nVal = 0; nVal < nextLayer.size() - 1; nVal++) {
      sum += m_outputWeights[nVal].weight * nextLayer[nVal].m_gradient;
   }

   return sum;
}

void Neuron::feedForward(const Layer &prevLayer)
{
   // Function is going to simply be the sum of all the inputs from the previous layer
   double sum = 0.0;

   for (int nVal = 0; nVal < prevLayer.size(); nVal++) {
      sum += prevLayer[nVal].getOutputVal() * prevLayer[nVal].m_outputWeights[m_myNeurIndex].weight;
   }

   m_outputVal = transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
   double delta = targetVal - m_outputVal;
   m_gradient = delta * transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
   double dow = sumDOW(nextLayer);
   m_gradient = dow * transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
   for (int nVal = 0; nVal < prevLayer.size(); nVal++) {
      Neuron &neuron = prevLayer[nVal];
      double oldDeltaWeight = neuron.m_outputWeights[m_myNeurIndex].deltaWeight;

      double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
      neuron.m_outputWeights[m_myNeurIndex].deltaWeight = newDeltaWeight;
      neuron.m_outputWeights[m_myNeurIndex].weight += newDeltaWeight;
   }
}

void Neuron::setOutputVal(double outputVal)
{
   m_outputVal = outputVal;
}

double Neuron::getOutputVal(void) const
{
   return m_outputVal;
}

class Net
{
public:
   Net(const vector<int> &topology);
   void feedForward(const vector<double> &inputVals);
   void backProp(const vector<double> &targetVals); 
   void getResults(vector<double> &resultVals) const;

private:
   vector<Layer> m_layers;
   double m_error;
   double m_recentAverageError;
   double m_recentAverageSmoothingFactor;
};

Net::Net(const vector<int> &topology)
{
   // Topology input has a vector of a integers that denote the amount of neurons per layer.
   int numLayers = topology.size();
   for (int layerVal = 0; layerVal < numLayers; layerVal++) {
      m_layers.push_back(Layer());
      // Need the number of neurons for next layer
      int numOutputs = layerVal == topology.size() - 1 ? 0 : topology[layerVal + 1];

      // Adding a bias neuron on top of the amount of neurons specified.
      for (int neuronVal = 0; neuronVal <= topology[layerVal]; neuronVal++) {
         m_layers.back().push_back(Neuron(numOutputs, neuronVal));
      }
   }
}

void Net::feedForward(const vector<double> &inputVals)
{
   // Confirming that the input neurons match the number of neurons present in the input layer
   assert(inputVals.size() == m_layers[0].size() - 1);

   // Setting up the values ("latching")
   for (int i = 0; i < inputVals.size(); i++) {
      m_layers[0][i].setOutputVal(inputVals[i]);
   }

   // Forward propogate
   for (int layerVal = 1; layerVal < m_layers.size(); layerVal++) {
      Layer &prevLayer = m_layers[layerVal - 1];
      for (int nVal = 0; nVal < m_layers[layerVal].size() - 1; nVal++) {
         m_layers[layerVal][nVal].feedForward(prevLayer);
      } 
   }
}

void Net::backProp(const vector<double> &targetVals)
{
   // Calculate overall net error (RMS of output neuron errors)
   // Minimize net error
   Layer &outputLayer = m_layers.back();
   m_error = 0.0;

   for (int nVal = 0; nVal < outputLayer.size() - 1; nVal++) {
      double delta = targetVals[nVal] - outputLayer[nVal].getOutputVal();
      m_error += delta * delta;
   }
   m_error /= outputLayer.size() - 1;
   m_error = sqrt(m_error);

   // Implement recent average measurement;
   m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
      / (m_recentAverageSmoothingFactor + 1.0);

   // Calculate output layer gradients
   for (int nVal = 0; nVal < outputLayer.size() - 1; nVal++) {
      outputLayer[nVal].calcOutputGradients(targetVals[nVal]);
   }

   // Calculate gradients on hidden layers
   for (int layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) {
      Layer &hiddenLayer = m_layers[layerNum];
      Layer &nextLayer = m_layers[layerNum + 1];

      for (int nVal = 0; nVal < hiddenLayer.size(); nVal++) {
         hiddenLayer[nVal].calcHiddenGradients(nextLayer);
      }
   }

   // For all layers from outputs to first hidden layer, update connection weights
   for (int layerNum = m_layers.size() - 1; layerNum > 0; layerNum--) {
      Layer &layer = m_layers[layerNum];
      Layer &prevLayer = m_layers[layerNum - 1];

      for (int nVal = 0; nVal < layer.size() - 1; nVal++) {
         layer[nVal].updateInputWeights(prevLayer);
      }
   }
}

void Net::getResults(vector<double> &resultVals) const {
   resultVals.clear();

   for (int nVal = 0; nVal < m_layers.back().size() - 1; nVal++) {
      resultVals.push_back(m_layers.back()[nVal].getOutputVal());
   }
}

// Test Main()
int main()
{
   vector<int> topology;
   topology.push_back(3);
   topology.push_back(2);
   topology.push_back(1);
   Net testingNet(topology);

   vector<double> inputVals;
   testingNet.feedForward(inputVals);

   vector<double> targetVals;
   testingNet.backProp(targetVals);

   vector<double> resultVals;
   testingNet.getResults(resultVals);
}