package org.deeplearning4j.examples;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class WineQualityClassifier_Tutorial {
    public static void main(String[] args) throws Exception {
        int batchSize = 128;
        int labelIndex = 11; // "quality" is the 12th column (0-based)
        int numClasses = 2;
        int numInputs = 11;
        int seed = 123;

        // 1. Read and preprocess the data
        List<DataSet> allDataList = new ArrayList<>();
        try (CSVRecordReader rr = new CSVRecordReader(1, ';')) {
            rr.initialize(new FileSplit(new File("data/winequality-red.csv")));
            while (rr.hasNext()) {
                List<org.datavec.api.writable.Writable> record = rr.next();
                double[] features = new double[numInputs];
                for (int i = 0; i < numInputs; i++) {
                    features[i] = Double.parseDouble(record.get(i).toString());
                }
                int quality = Integer.parseInt(record.get(labelIndex).toString());
                int label = (quality >= 6) ? 1 : 0;
                double[] labelArr = new double[numClasses];
                labelArr[label] = 1.0;
                DataSet ds = new DataSet(
                        org.nd4j.linalg.factory.Nd4j.create(features, new long[]{1, numInputs}),
                        org.nd4j.linalg.factory.Nd4j.create(labelArr, new long[]{1, numClasses})
                );
                allDataList.add(ds);
            }
        }

        // 2. Shuffle and split into train/test
        Collections.shuffle(allDataList, new java.util.Random(seed));
        int trainSize = (int) (allDataList.size() * 0.8);
        List<DataSet> trainList = allDataList.subList(0, trainSize);
        List<DataSet> testList = allDataList.subList(trainSize, allDataList.size());

        DataSet trainData = DataSet.merge(trainList);
        DataSet testData = DataSet.merge(testList);

        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        normalizer.transform(trainData);
        normalizer.transform(testData);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(32).nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(16).nOut(numClasses)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(20));

        for (int epoch = 0; epoch < 20; epoch++) {
            model.fit(trainData);
            System.out.println("Epoch " + (epoch + 1) + " complete");
        }
        Evaluation eval = new Evaluation(numClasses);
        INDArray output = model.output(testData.getFeatures());

        System.out.println(output);

        eval.eval(testData.getLabels(),output);
        System.out.printf("Accuracy: %.2f%%\n", 100 * eval.accuracy());


        double[] learningRates = {0.1, 0.001};
        int[] nEpochs = {50, 100, 200};
        int[] hidden1Sizes = {128};
        int[] hidden2Sizes = {16,32,64};

        for (double lr: learningRates){
                for(int nepoch: nEpochs){
                    for(int h1: hidden1Sizes){
                        for(int h2: hidden1Sizes) {

                        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                                .seed(seed)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                .updater(new Adam(lr))
                                .list()
                                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(h1)
                                        .activation(Activation.RELU)
                                        .build())
                                .layer(new DenseLayer.Builder().nIn(h1).nOut(h2)
                                        .activation(Activation.RELU)
                                        .build())
                                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                        .nIn(h2).nOut(numClasses)
                                        .activation(Activation.SIGMOID)
                                        .build())
                                .build();
                        MultiLayerNetwork model2 = new MultiLayerNetwork(conf2);
                        model2.init();
                        for (int epoch = 0; epoch < nepoch; epoch++) {
                            model2.fit(trainData);

                        }
                        Evaluation eval2 = new Evaluation(numClasses);
                        INDArray output2 = model2.output(testData.getFeatures());
                        eval2.eval(testData.getLabels(),output2);
                        System.out.println(lr);
                        System.out.println(nepoch);
                        System.out.println(h1);
                        System.out.printf("Accuracy: %.2f%%\n", 100 * eval2.accuracy());

                        }
                    }
                }
            }

    }





    }

