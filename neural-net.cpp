#include <vector>
#include <iostream>

using namespace std;

class Neuron {};
typedef vector<Neuron> Layer;

class Net
{
public:
   Net(const vector<int> &topology);
   void feedForward(const vector<double> &inputVals) {};
   void backProp(const vector<double> &targetVals) {}; 
   void getResults(vector<double> &resultVals) const {};

private:
   vector<Layer> m_layers;
};

Net::Net(const vector<int> &topology)
{
   // Topology input has a vector of a integers that denote the amount of neurons per layer.
   int numLayers = topology.size();
   for (int layerVal = 0; layerVal < numLayers; layerVal++) {
      m_layers.push_back(Layer());

      // Adding a bias neuron on top of the amount of neurons specified.
      for (int neuronVal = 0; neuronVal <= topology[layerVal]; neuronVal++) {
         m_layers.back().push_back(Neuron());
         cout << "Made a neuron!" << endl;
      }
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