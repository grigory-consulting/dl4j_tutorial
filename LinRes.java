package org.deeplearning4j.examples;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.jfree.chart.*;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.*;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

import javax.swing.*;
import java.awt.*;
import java.util.Random;

public class LinRes {
    public static void main(String[] args) {
        Nd4j.getRandom().setSeed(1);
        Random rng = new Random(1);

        int m = 100;
        INDArray X = Nd4j.rand(DataType.FLOAT, m, 1).mul(2);
        INDArray y = X.mul(3).add(4).add(Nd4j.randn(DataType.FLOAT, m, 1));
        INDArray X_b = Nd4j.hstack(Nd4j.ones(DataType.FLOAT, m, 1), X);

        // Closed-form (Normal Equation)
        INDArray w_best = InvertMatrix.invert(X_b.transpose().mmul(X_b), false)
                .mmul(X_b.transpose()).mmul(y);

        // Batch GD
        float alphaBatch = 0.1f;
        int nEpochsBatch = 50;
        INDArray w_batch = Nd4j.randn(DataType.FLOAT, 2, 1);
        float[] w0_batch = new float[nEpochsBatch];
        float[] w1_batch = new float[nEpochsBatch];
        for (int epoch = 0; epoch < nEpochsBatch; epoch++) {
            w0_batch[epoch] = w_batch.getFloat(0);
            w1_batch[epoch] = w_batch.getFloat(1);
            INDArray gradients = X_b.transpose().mmul(X_b.mmul(w_batch).sub(y)).mul(2.0f / m);
            w_batch = w_batch.sub(gradients.mul(alphaBatch));
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

        // Stochastic GD
        float alphaSGD = 0.1f;
        int nEpochsSGD = 50;
        INDArray w_sgd = Nd4j.randn(DataType.FLOAT, 2, 1);
        float[] w0_sgd = new float[nEpochsSGD];
        float[] w1_sgd = new float[nEpochsSGD];
        for (int epoch = 0; epoch < nEpochsSGD; epoch++) {
            w0_sgd[epoch] = w_sgd.getFloat(0);
            w1_sgd[epoch] = w_sgd.getFloat(1);
            for (int iter = 0; iter < m; iter++) {
                int random_index = rng.nextInt(m);
                INDArray xi = X_b.getRow(random_index).reshape(1, 2);
                INDArray yi = y.getRow(random_index).reshape(1, 1);
                INDArray gradient = xi.transpose().mmul(xi.mmul(w_sgd).sub(yi)).mul(2f);
                w_sgd = w_sgd.sub(gradient.mul(alphaSGD));
            }
        }

        // For plotting: use ND4J min/max directly
        float xMin = Nd4j.min(X).getFloat(0);
        float xMax = Nd4j.max(X).getFloat(0);

        // Get train data for scatter plot
        float[] xVals = X.data().asFloat();
        float[] yVals = y.data().asFloat();

        // Compute end-points for all four lines (at xMin/xMax)
        // Closed-form
        float yMin_best = w_best.getFloat(0) + w_best.getFloat(1) * xMin;
        float yMax_best = w_best.getFloat(0) + w_best.getFloat(1) * xMax;
        XYSeries lineBest = new XYSeries("Closed-form");
        lineBest.add(xMin, yMin_best);
        lineBest.add(xMax, yMax_best);

        // Batch GD
        float yMin_batch = w_batch.getFloat(0) + w_batch.getFloat(1) * xMin;
        float yMax_batch = w_batch.getFloat(0) + w_batch.getFloat(1) * xMax;
        XYSeries lineBatch = new XYSeries("Batch GD");
        lineBatch.add(xMin, yMin_batch);
        lineBatch.add(xMax, yMax_batch);

        // Stochastic GD
        float yMin_sgd = w_sgd.getFloat(0) + w_sgd.getFloat(1) * xMin;
        float yMax_sgd = w_sgd.getFloat(0) + w_sgd.getFloat(1) * xMax;
        XYSeries lineSGD = new XYSeries("Stochastic GD");
        lineSGD.add(xMin, yMin_sgd);
        lineSGD.add(xMax, yMax_sgd);

        // Mini-Batch GD
        float yMin_mb = w_mb.getFloat(0) + w_mb.getFloat(1) * xMin;
        float yMax_mb = w_mb.getFloat(0) + w_mb.getFloat(1) * xMax;
        XYSeries lineMB = new XYSeries("Mini-Batch GD");
        lineMB.add(xMin, yMin_mb);
        lineMB.add(xMax, yMax_mb);

        XYSeries scatterSeries = new XYSeries("Train Data");
        for (int i = 0; i < xVals.length; i++) scatterSeries.add(xVals[i], yVals[i]);

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(scatterSeries);
        dataset.addSeries(lineBest);
        dataset.addSeries(lineBatch);
        dataset.addSeries(lineSGD);
        dataset.addSeries(lineMB);

        JFreeChart chart = ChartFactory.createScatterPlot(
                "ND4J Linear Regression: Closed-Form, Batch GD, SGD, Mini-Batch GD",
                "X", "y", dataset,
                PlotOrientation.VERTICAL, true, true, false
        );

        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesLinesVisible(0, false); // scatter only
        renderer.setSeriesShapesVisible(0, true);

        // Closed-form: red, solid
        renderer.setSeriesLinesVisible(1, true); renderer.setSeriesShapesVisible(1, false);
        renderer.setSeriesPaint(1, Color.RED);

        // Batch GD: blue, dashed
        renderer.setSeriesLinesVisible(2, true); renderer.setSeriesShapesVisible(2, false);
        renderer.setSeriesPaint(2, Color.BLUE);
        renderer.setSeriesStroke(2, new BasicStroke(2.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{5.0f, 5.0f}, 0));

        // SGD: green, dotted
        renderer.setSeriesLinesVisible(3, true); renderer.setSeriesShapesVisible(3, false);
        renderer.setSeriesPaint(3, Color.GREEN.darker());
        renderer.setSeriesStroke(3, new BasicStroke(2.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{2.0f, 6.0f}, 0));

        // Mini-Batch GD: magenta, dash-dot
        renderer.setSeriesLinesVisible(4, true); renderer.setSeriesShapesVisible(4, false);
        renderer.setSeriesPaint(4, Color.MAGENTA.darker());
        renderer.setSeriesStroke(4, new BasicStroke(2.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{8.0f, 3.0f, 2.0f, 3.0f}, 0));


        plot.setRenderer(renderer);

        JFrame frame = new JFrame("ND4J Linear Regression Fits");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.setSize(800, 550);
        frame.setVisible(true);

        // === Plot weight evolution ===
        XYSeriesCollection wsCollection = new XYSeriesCollection();
        XYSeries ws_b0 = new XYSeries("BatchGD intercept");
        XYSeries ws_b1 = new XYSeries("BatchGD slope");
        for (int i = 0; i < nEpochsBatch; i++) { ws_b0.add(i, w0_batch[i]); ws_b1.add(i, w1_batch[i]);}
        wsCollection.addSeries(ws_b0); wsCollection.addSeries(ws_b1);

        XYSeries ws_s0 = new XYSeries("StochGD intercept");
        XYSeries ws_s1 = new XYSeries("StochGD slope");
        for (int i = 0; i < nEpochsSGD; i++) { ws_s0.add(i, w0_sgd[i]); ws_s1.add(i, w1_sgd[i]);}
        wsCollection.addSeries(ws_s0); wsCollection.addSeries(ws_s1);

        XYSeries ws_m0 = new XYSeries("MiniBatchGD intercept");
        XYSeries ws_m1 = new XYSeries("MiniBatchGD slope");
        for (int i = 0; i < nEpochsMB; i++) { ws_m0.add(i, w0_mb[i]); ws_m1.add(i, w1_mb[i]);}
        wsCollection.addSeries(ws_m0); wsCollection.addSeries(ws_m1);

        JFreeChart wsChart = ChartFactory.createXYLineChart(
                "Evolution of Weights Per Epoch (Batch, Stochastic, Mini-Batch)",
                "Epoch", "Weight Value", wsCollection,
                PlotOrientation.VERTICAL, true, true, false
        );
        XYPlot wsPlot = wsChart.getXYPlot();
        XYLineAndShapeRenderer wsRenderer = new XYLineAndShapeRenderer(true, false);
        wsPlot.setRenderer(wsRenderer);

        JFrame wsFrame = new JFrame("Weight Evolution");
        wsFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        wsFrame.add(new ChartPanel(wsChart));
        wsFrame.setSize(800, 550);
        wsFrame.setVisible(true);

        // Console output
        System.out.println("Closed-form weights:   " + w_best);
        System.out.println("Batch GD weights:      " + w_batch);
        System.out.println("Stochastic GD weights: " + w_sgd);
        System.out.println("Mini-Batch GD weights: " + w_mb);
    }
}