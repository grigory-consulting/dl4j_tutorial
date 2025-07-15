package org.deeplearning4j.examples;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.xml.crypto.Data;
import java.util.Random;


public class LinResTutorial {
    public static void main(String[] args){
        Nd4j.getRandom().setSeed(1);
        Random rng = new Random();

        int m = 100; // Anzahl der Datenpunte

        INDArray X = Nd4j.rand(DataType.FLOAT, m ,1 ).mul(2); // Zufallszahlen aus [0,2]
        INDArray y = X.mul(3).add(4).add(Nd4j.randn(DataType.FLOAT,m,1));
        INDArray X_b = Nd4j.hstack(Nd4j.ones(DataType.FLOAT,m,1), X); // Design-Matrix

        // Analytische Lösung
        INDArray w_best = InvertMatrix.invert(X_b.transpose().mmul(X_b),false).mmul(X_b.transpose()).mmul(y);

        // Batch GD

        float alphaBatch = 0.1f;
        int nEpochsBatch = 50; // Anzahl der Epochen ( 1 Epoche = 1 mal hat ML ALgorithmus den Datensatz "gesehen")
        INDArray w_batch = Nd4j.randn(DataType.FLOAT, 2,1 );

        for (int epoch=0; epoch < nEpochsBatch; epoch++){

            INDArray gradients = X_b.transpose().mmul(X_b.mmul(w_batch).sub(y)).mul(2.0f / m);
            w_batch = w_batch.sub(gradients.mul(alphaBatch));

        }

        // Stochastic Gradient Descent (SGD)

        float alphaSGD = 0.1f;
        int nEpochSGD = 50;
        INDArray w_sgd = Nd4j.randn(DataType.FLOAT, 2,1 );

        for (int epoch=0; epoch < nEpochsBatch; epoch++){

            for (int iter = 0; iter < m ; iter++){
                int random_index = rng.nextInt(m);
                INDArray xi = X_b.getRow(random_index).reshape(1, 2);
                INDArray yi = y.getRow(random_index).reshape(1, 1);
                INDArray gradient = xi.transpose().mmul(xi.mmul(w_sgd).sub(yi)).mul(2f);
                w_sgd = w_sgd.sub(gradient.mul(alphaSGD));
            }

        }

        // Mini-Batch GD
        float alphaMB = 0.1f;
        int nEpochsMB = 50;
        int miniBatchSize = 16;
        INDArray w_mb = Nd4j.randn(DataType.FLOAT, 2, 1);
        float[] w0_mb = new float[nEpochsMB];
        float[] w1_mb = new float[nEpochsMB];
        for (int epoch = 0; epoch < nEpochsMB; epoch++) {
            w0_mb[epoch] = w_mb.getFloat(0);
            w1_mb[epoch] = w_mb.getFloat(1);
            // Shuffle indices using Collections.shuffle
            java.util.List<Integer> indices = new java.util.ArrayList<>();
            for (int i = 0; i < m; i++) indices.add(i);
            java.util.Collections.shuffle(indices, rng);
            for (int start = 0; start < m; start += miniBatchSize) {
                int end = Math.min(start + miniBatchSize, m);
                int size = end - start;
                INDArray xi = Nd4j.create(DataType.FLOAT, size, 2);
                INDArray yi = Nd4j.create(DataType.FLOAT, size, 1);
                for (int k = 0; k < size; k++) {
                    xi.putRow(k, X_b.getRow(indices.get(start + k)));
                    yi.putRow(k, y.getRow(indices.get(start + k)));
                }
                INDArray gradient = xi.transpose().mmul(xi.mmul(w_mb).sub(yi)).mul(2.0f / size);
                w_mb = w_mb.sub(gradient.mul(alphaMB));
            }
        }



        System.out.println("Analytisch: " + w_best);
        System.out.println("BGD: " + w_batch);
        System.out.println("SGD: " + w_sgd);
        System.out.println("MBGD: " + w_mb);

        INDArray y_pred_analytisch = X_b.mmul(w_best);
        INDArray y_pred_bgd = X_b.mmul(w_batch);
        INDArray y_pred_sgd = X_b.mmul(w_sgd);
        INDArray y_pred_mbgd = X_b.mmul(w_mb);

        System.out.println("\n--- Regression Report ---");
        System.out.printf("%-12s %-10s %-10s %-10s %-10s\n", "Methode", "MSE", "RMSE", "MAD", "R²");
        System.out.printf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f\n", "Analytisch", mse(y, y_pred_analytisch), rmse(y, y_pred_analytisch), mad(y, y_pred_analytisch), r2(y, y_pred_analytisch));
        System.out.printf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f\n", "BGD", mse(y, y_pred_bgd), rmse(y, y_pred_bgd), mad(y, y_pred_bgd), r2(y, y_pred_bgd));
        System.out.printf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f\n", "SGD", mse(y, y_pred_sgd), rmse(y, y_pred_sgd), mad(y, y_pred_sgd), r2(y, y_pred_sgd));
        System.out.printf("%-12s %-10.4f %-10.4f %-10.4f %-10.4f\n", "MBGD", mse(y, y_pred_mbgd), rmse(y, y_pred_mbgd), mad(y, y_pred_mbgd), r2(y, y_pred_mbgd));




    }

    public static float mse(INDArray yTrue, INDArray yPred) {
        INDArray diff = yTrue.sub(yPred);
        INDArray sq = diff.mul(diff);
        return ((Number) sq.meanNumber()).floatValue();
    }

    public static float rmse(INDArray yTrue, INDArray yPred) {
        return (float) Math.sqrt(mse(yTrue, yPred));
    }

    public static float mad(INDArray yTrue, INDArray yPred) {
        return ((Number) Transforms.abs(yTrue.sub(yPred)).meanNumber()).floatValue();
    }

    public static float r2(INDArray yTrue, INDArray yPred) {
        float ssRes = ((Number) yTrue.squaredDistance(yPred)).floatValue();
        float mean = ((Number) yTrue.meanNumber()).floatValue();
        float ssTot = ((Number) yTrue.sub(mean).mul(yTrue.sub(mean)).sumNumber()).floatValue();
        return 1 - ssRes / ssTot;
    }







}
