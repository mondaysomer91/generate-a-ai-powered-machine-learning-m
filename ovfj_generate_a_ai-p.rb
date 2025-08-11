require 'rubygems'
require 'mlinalg'
require 'ai4r'

class AIPoweredMLMSimulator
  attr_accessor :data, :target, :model, :predictions

  def initialize(data, target)
    @data = data
    @target = target
  end

  def train
    @model = Ai4r::DecisionTree::C45.new
    @model.build(@data, @target)
  end

  def predict(test_data)
    @predictions = []
    test_data.each do |row|
      prediction = @model.eval(row)
      @predictions << prediction
    end
    @predictions
  end

  def evaluate
    correct = 0
    @predictions.each_with_index do |prediction, index|
      correct += 1 if prediction == @target[index]
    end
    accuracy = (correct.to_f / @predictions.size) * 100
    puts "Model Accuracy: #{accuracy}%"
  end
end

# Example usage
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
target = [0, 1, 2]
simulator = AIPoweredMLMSimulator.new(data, target)
simulator.train
test_data = [[10, 11, 12], [13, 14, 15]]
predictions = simulator.predict(test_data)
simulator.evaluate