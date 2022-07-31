/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.bana274.controllers;

import com.bana274.Main;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.ResourceBundle;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Paint;
import javafx.stage.WindowEvent;
import javafx.util.Pair;
import javax.imageio.ImageIO;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kordamp.ikonli.javafx.FontIcon;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author brianj
 */
public class MainController {

    private final ResourceBundle mainBundle = ResourceBundle.getBundle("com.bana274.main");

    @FXML
    private AnchorPane viewContainerAnchorPane;

    @FXML
    private Button classifyButton;
    @FXML
    private Button createModelButton;

    @FXML
    void initialize() {
        initActions();
    }

    private void initActions() {
        classifyButton.setOnAction(evt -> {
            try {
                classify(null);
            } catch (IOException ex) {
                Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
            }
        });
        createModelButton.setOnAction(this::buildCNNModel);
    }

    private Pair<DataSetIterator, DataSetIterator> createDataIterators() {

        String dataPath = "/home/brianj/Pictures/images";

        int batchSize = 1000;

        int height = 32;
        int width = 32;
        int channels = 3;
        int numInput = height * width;

        int numLabels = 2;

        File parentDir = new File(dataPath);
        FileSplit filesInDir = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(), labelMaker, 47000);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        // Assumes images are separated into different folders, where each folder is a class.
        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, 3, labelMaker);
        try {
            trainRecordReader.initialize(trainData);
        } catch (IOException ex) {
            Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
        }

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, numLabels);
        // pixel values from 0-255 to 0-1 (min-max scaling)
        DataNormalization imageScaler = new ImagePreProcessingScaler(0, 1);
        imageScaler.fit(trainIter);
        trainIter.setPreProcessor(imageScaler);

        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, 3, labelMaker);
        try {
            testRecordReader.initialize(testData);
        } catch (IOException ex) {
            Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
        }
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, numLabels);
        DataNormalization imageScaler2 = new ImagePreProcessingScaler(0, 1);
        imageScaler2.fit(testIter);
        testIter.setPreProcessor(imageScaler2);

        return new Pair<DataSetIterator, DataSetIterator>(trainIter, testIter);
    }

    private void buildModel(ActionEvent evt) {

        String dataPath = "/home/brianj/Pictures/images";

        int seed = 123;
        int batchSize = 1000;
        int numEpochs = 1;

        int height = 100;
        int width = 80;
        int channels = 3;
        int numInput = height * width;

        int numLabels = 2;

        Pair<DataSetIterator, DataSetIterator> iterators = createDataIterators();

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .activation(Activation.SIGMOID)
                        .nIn(numInput)
                        .nOut(1000)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID)
                        .nIn(1000)
                        .nOut(2)
                        .build())
                .setInputType(InputType.convolutional(height, width, 3))
                .build();

        System.out.println(configuration.toJson());

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(iterators.getKey(), numEpochs);

        Evaluation evaluation = model.evaluate(iterators.getValue());
        System.out.println(evaluation.stats());

    }

    private void buildCNNModel(ActionEvent evt) {
        final int HEIGHT = 32;
        final int WIDTH = 32;
        final int CHANNELS = 3;
        final int N_OUTCOMES = 2;
        final int N_EPOCHS = 11;

        Pair<DataSetIterator, DataSetIterator> iterators = createDataIterators();

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.0005) // ridge regression value
                .updater(new Nesterovs(0.006, 0.9))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(N_OUTCOMES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS)) // InputType.convolutional for normal image
                .build();

        System.out.println(configuration.toJson());

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(iterators.getKey(), N_EPOCHS);

        Evaluation evaluation = model.evaluate(iterators.getValue());
        System.out.println(evaluation.stats());
        try {
            // Save model
            model.save(new File("gender_cnn_model.zip"));
        } catch (IOException ex) {
            Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static INDArray classify(BufferedImage image) throws IOException {

        final int HEIGHT = 32;
        final int WIDTH = 32;
        final int CHANNELS = 3;
        
        image = ImageIO.read(new File("/home/brianj/Pictures/images/female/131437.jpg.jpg"));

        String modelFile = "gender_cnn_model.zip";
        MultiLayerNetwork classifier = MultiLayerNetwork.load(new File(modelFile), false);
       
        ImageLoader loader = new ImageLoader(WIDTH, HEIGHT, CHANNELS);        
        INDArray input = loader.asMatrix(image).reshape(1, 3, 32, 32);
        INDArray output = classifier.output(input);

        System.out.println(output);
        return output;
    }

    public EventHandler<WindowEvent> getWindowCloseEventHandler() {
        return (WindowEvent event) -> {
            closeRequestHandler();
            event.consume();
        };
    }

    private void closeRequestHandler() {
        Alert alertDialog = new Alert(Alert.AlertType.CONFIRMATION);
        FontIcon fontIcon = new FontIcon("fas-sign-out-alt");
        fontIcon.setIconSize(56);
        fontIcon.setIconColor(Paint.valueOf("#e05600"));
        alertDialog.setGraphic(fontIcon);
        alertDialog.initOwner(viewContainerAnchorPane.getScene().getWindow());
        alertDialog.getDialogPane().getStylesheets().add(Main.class.getResource("application.css").toExternalForm());
        alertDialog.setTitle(mainBundle.getString("exit-dialog-title-text"));
        alertDialog.setHeaderText(mainBundle.getString("exit-dialog-header-text"));
        //alertDialog.setContentText(bundle.getString("exit-dialog-content-text"));
        alertDialog.showAndWait()
                .filter(response -> response == ButtonType.OK)
                .ifPresent(response -> {
                    Platform.exit();
                });
    }

}
