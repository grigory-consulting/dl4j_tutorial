package org.deeplearning4j.examples;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class MNISTClassifier_Tutorial {
}
