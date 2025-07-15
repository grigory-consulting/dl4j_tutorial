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



        

    }
}
