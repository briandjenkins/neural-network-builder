/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.bana274.controllers;

import com.bana274.Main;
import static com.bana274.utilities.AppUtils.ofMimeMultipartData;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Instant;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.ResourceBundle;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;
import java.util.function.UnaryOperator;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.animation.FadeTransition;
import javafx.animation.SequentialTransition;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.binding.ObjectBinding;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.control.TextFormatter;
import javafx.scene.control.TextFormatter.Change;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Paint;
import javafx.stage.FileChooser;
import javafx.stage.WindowEvent;
import javafx.util.Duration;
import javafx.util.Pair;
import javafx.util.converter.DoubleStringConverter;
import javafx.util.converter.IntegerStringConverter;
import javax.imageio.ImageIO;
import org.apache.commons.lang.time.DurationFormatUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.deeplearning4j.ui.api.UIServer;
import org.kordamp.ikonli.javafx.FontIcon;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Version: 1.1.0 Last Update: 2022-08-19
 *
 *
 * @author brianj
 */
public class MainController {

    public static final int ELAPSED_TIME_INTERVAL = 1000;

    private final ResourceBundle mainBundle = ResourceBundle.getBundle("com.bana274.main");
    private final FileChooser fileChooser = new FileChooser();
    private Timer timer;
    private Instant start;
    private UIServer uiServer;
    private StatsStorage statsStorage;

    private BooleanProperty finishedIconVisible = new SimpleBooleanProperty(false);
    private DoubleProperty accuracyText = new SimpleDoubleProperty();
    private ObjectBinding<Node> frontNode;
    private StringProperty classificationText = new SimpleStringProperty();

    @FXML
    private AnchorPane viewContainerAnchorPane;
    @FXML
    private AnchorPane overlayAnchorPane;
    @FXML
    private AnchorPane classifyingOverlayAnchorPane;
    @FXML
    private AnchorPane classifiedOverlayAnchorPane;
    @FXML
    private StackPane mainStackPane;
    @FXML
    private Label elapsedTimeLabel;
    @FXML
    private Label accuracyLabel;
    @FXML
    private Label classificationLabel;
    @FXML
    private TextField epochsTextField;
    @FXML
    private TextField learningRateTextField;
    @FXML
    private TextField minibatchSizeTextField;
    @FXML
    private Button classifyButton;
    @FXML
    private Button createModelButton;
    @FXML
    private Button closeOverlayButton;
    @FXML
    private Button closeClassifierOverlayButton;
    @FXML
    private Button closeClassifiedOverlayButton;
    @FXML
    private ImageView mainBackgroundImageView;
    @FXML
    private ImageView classifiedGenderImageView;
    @FXML
    private FontIcon elapsedTimeFontIcon;

    @FXML
    void initialize() {
        initControls();
        initActions();
        initBindings();
    }

    /**
     * Initialize form controls with default values.
     */
    private void initControls() {
        // Move overlay off-screen.
        installAnimation(mainStackPane);

        epochsTextField.setTextFormatter(new TextFormatter<Integer>(new IntegerStringConverter(), 0, customIntegerFilter()));
        epochsTextField.setText("100");
        learningRateTextField.setTextFormatter(new TextFormatter<Double>(new DoubleStringConverter(), 0.0, customDoubleFilter()));
        learningRateTextField.setText("0.1");
        minibatchSizeTextField.setTextFormatter(new TextFormatter<Integer>(new IntegerStringConverter(), 0, customIntegerFilter()));
        minibatchSizeTextField.setText("1000");
        mainBackgroundImageView.setImage(new Image(getClass().getResource("/com/bana274/background-image.png").toExternalForm()));
    }

    private void initActions() {
        classifyButton.setOnAction(this::selectImageAndClassify);
        createModelButton.setOnAction(this::buildCNNModel);
        closeOverlayButton.setOnAction(this::closeTrainingOverlay);
        closeClassifierOverlayButton.setOnAction(this::closeClassifyingOverlay);
        closeClassifiedOverlayButton.setOnAction(this::closeClassifiedOverlay);
    }

    private void initBindings() {
        elapsedTimeFontIcon.visibleProperty().bind(finishedIconVisible);
        accuracyLabel.textProperty().bind(Bindings.format("%.2f%%", accuracyText));
        classificationLabel.textProperty().bind(classificationText);
    }

    private void installAnimation(StackPane root) {
        frontNode = Bindings.valueAt(root.getChildren(), Bindings.size(root.getChildren()).subtract(1));
        frontNode.addListener((obs, oldNode, newNode) -> {
            SequentialTransition fadeOutIn = new SequentialTransition();
            if (oldNode != null) {
                FadeTransition fadeOut = new FadeTransition(Duration.millis(1000), oldNode);
                fadeOut.setToValue(0);
                fadeOutIn.getChildren().add(fadeOut);
                System.out.println("Fade out");

            }
            if (newNode != null) {
                FadeTransition fadeIn = new FadeTransition(Duration.millis(1000), newNode);
                fadeIn.setFromValue(0);
                fadeIn.setToValue(1);
                fadeOutIn.getChildren().add(fadeIn);
                System.out.println("Fade in");
            }
            fadeOutIn.play();
        });
    }

    private UnaryOperator<Change> customIntegerFilter() {
        return change -> {
            String newText = change.getControlNewText();
            // if proposed change results in a valid value, return change as-is:
            if (newText.matches("-?([1-9][0-9]*)?")) {
                return change;
            } else if ("-".equals(change.getText())) {

                // if user types or pastes a "-" in middle of current text,
                // toggle sign of value:
                if (change.getControlText().startsWith("-")) {
                    // if we currently start with a "-", remove first character:
                    change.setText("");
                    change.setRange(0, 1);
                    // since we're deleting a character instead of adding one,
                    // the caret position needs to move back one, instead of 
                    // moving forward one, so we modify the proposed change to
                    // move the caret two places earlier than the proposed change:
                    change.setCaretPosition(change.getCaretPosition() - 2);
                    change.setAnchor(change.getAnchor() - 2);
                } else {
                    // otherwise just insert at the beginning of the text:
                    change.setRange(0, 0);
                }
                return change;
            }
            // invalid change, veto it by returning null:
            return null;
        };
    }

    private UnaryOperator<Change> customDoubleFilter() {
        return change -> {
            String newText = change.getControlNewText();
            // if proposed change results in a valid value, return change as-is:
            if (newText.matches("-?([0-9]*\\.?[0-9]*)?")) {
                return change;
            } else if ("-".equals(change.getText())) {

                // if user types or pastes a "-" in middle of current text,
                // toggle sign of value:
                if (change.getControlText().startsWith("-")) {
                    // if we currently start with a "-", remove first character:
                    change.setText("");
                    change.setRange(0, 1);
                    // since we're deleting a character instead of adding one,
                    // the caret position needs to move back one, instead of 
                    // moving forward one, so we modify the proposed change to
                    // move the caret two places earlier than the proposed change:
                    change.setCaretPosition(change.getCaretPosition() - 2);
                    change.setAnchor(change.getAnchor() - 2);
                } else {
                    // otherwise just insert at the beginning of the text:
                    change.setRange(0, 0);
                }
                return change;
            }
            // invalid change, veto it by returning null:
            return null;
        };
    }

    private void startClock() {
        start = Instant.now();
        timer = new Timer();
        updateClock();
        TimerTask task = new TimerTask() {
            @Override
            public void run() {
                //The task you want to do
                Platform.runLater(() -> updateClock());
            }
        };
        timer.scheduleAtFixedRate(task, 0, ELAPSED_TIME_INTERVAL);
    }

    private void updateClock() {
        Instant finish = Instant.now();
        long timeElapsed = java.time.Duration.between(start, finish).toMillis();
        String elapsedTime = DurationFormatUtils.formatDuration(timeElapsed, "HH:mm:ss");
        elapsedTimeLabel.setText(elapsedTime);
    }

    private void stopClock() {
        timer.cancel();
    }

    private void closeTrainingOverlay(ActionEvent evt) {
        overlayAnchorPane.toBack();
    }

    private void closeClassifyingOverlay(ActionEvent evt) {
        classifyingOverlayAnchorPane.toBack();
    }

    private void closeClassifiedOverlay(ActionEvent evt) {
        classifiedOverlayAnchorPane.toBack();
    }

    private void selectImageAndClassify(ActionEvent evt) {

        File selectedFile = fileChooser.showOpenDialog(mainStackPane.getScene().getWindow());
        if (selectedFile != null) {
            try {
                classificationText.set("");
                classifyingOverlayAnchorPane.toFront();
                postImageToClassifier(selectedFile);
            } catch (IOException ex) {
                Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
            } catch (URISyntaxException ex) {
                Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
            } catch (InterruptedException ex) {
                Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
            } catch (TimeoutException ex) {
                Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    /**
     * CNN Model Related Methods
     *
     */
    /**
     * Configure and train the model.
     *
     * @param evt
     */
    private void buildCNNModel(ActionEvent evt) {
        overlayAnchorPane.toFront();
        // TODO: Create "running" state property.
        finishedIconVisible.set(false);
        startClock();

        if (uiServer != null) {
            uiServer.detach(statsStorage);
        }

        final int HEIGHT = 32;
        final int WIDTH = 32;
        final int CHANNELS = 3;
        final int N_OUTCOMES = 2;
        final int N_EPOCHS = Integer.parseInt(epochsTextField.getText());//11

        Task<org.nd4j.evaluation.classification.Evaluation> task = new Task<org.nd4j.evaluation.classification.Evaluation>() {
            @Override
            protected org.nd4j.evaluation.classification.Evaluation call() throws Exception {

                Pair<DataSetIterator, DataSetIterator> iterators = createDataIterators();

                MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .l2(0.0005) // ridge regression value
                        .updater(new Nesterovs(Double.parseDouble(learningRateTextField.getText()), 0.9)) //0.006              
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

                /**
                 * Visualization
                 *
                 * URL: http://localhost:9000/train/overview
                 */
                //Initialize the user interface backend
//                UIServer uiServer = UIServer.getInstance();
                //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
                //StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
//                statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
//                int listenerFrequency = 1;
//                model.setListeners(new StatsListener(statsStorage, listenerFrequency));
                //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//                uiServer.attach(statsStorage);
                model.fit(iterators.getKey(), N_EPOCHS);

                org.nd4j.evaluation.classification.Evaluation evaluation = model.evaluate(iterators.getValue());
                try {
                    // Save model
                    model.save(new File("gender_cnn_model.zip"));
                } catch (IOException ex) {
                    Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
                }
                return evaluation;
            }
        };
        task.setOnSucceeded(e -> {
            org.nd4j.evaluation.classification.Evaluation evaluation = task.getValue();
            accuracyText.set(evaluation.accuracy());
            System.out.println(evaluation.stats());
            System.out.println(evaluation.confusionToString());
            stopClock();
            finishedIconVisible.set(true);
        });
        task.setOnFailed(event -> task.getException().printStackTrace());
        Thread th = new Thread(task);
        th.setDaemon(true);
        th.start();

    }

    /**
     * For data preprocessing.
     *
     * Last Update: 2022-08-18
     *
     * @return
     */
    private Pair<DataSetIterator, DataSetIterator> createDataIterators() {

        String dataPath = "/home/brianj/Pictures/images";

        // The typically mini-batch sizes are 64, 128, 256 or 512.
        //int batchSize = 1000;
        int batchSize = Integer.parseInt(minibatchSizeTextField.getText());
        int height = 32;
        int width = 32;

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
        DataNormalization trainImageScaler = new ImagePreProcessingScaler(0, 1);
        trainImageScaler.fit(trainIter);
        trainIter.setPreProcessor(trainImageScaler);

        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, 3, labelMaker);
        try {
            testRecordReader.initialize(testData);
        } catch (IOException ex) {
            Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
        }
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, numLabels);
        DataNormalization testImageScaler = new ImagePreProcessingScaler(0, 1);
        testImageScaler.fit(testIter);
        testIter.setPreProcessor(testImageScaler);

        return new Pair<DataSetIterator, DataSetIterator>(trainIter, testIter);
    }

    /**
     * Used to test how well the model generalizes against new samples.
     *
     * @param image
     * @return
     * @throws IOException
     */
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

    private void postImageToClassifier(File file) throws IOException, URISyntaxException, InterruptedException, ExecutionException, TimeoutException {

        // Random 256 length string is used as multipart boundary
        String boundary = new BigInteger(256, new Random()).toString();
        Map<Object, Object> data = new LinkedHashMap<>();
        data.put("image", file.toPath());
        HttpRequest request = HttpRequest.newBuilder()
                .uri(new URI("http://gender-classifier-rest-service.herokuapp.com/upload/image"))
                .headers("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(ofMimeMultipartData(data, boundary))
                .build();
        CompletableFuture<HttpResponse<String>> response = HttpClient.newBuilder()
                .build()
                .sendAsync(request, HttpResponse.BodyHandlers.ofString());
        response.thenApply(HttpResponse::body).thenAccept(r -> {
            try {
                HashMap<String, Object> result = new ObjectMapper().readValue(r, HashMap.class);
                Platform.runLater(() -> {
                    classificationText.set(result.get("gender").toString());
                    if (result.get("gender").toString().equals("male")) {
                        classifiedGenderImageView.setImage(new Image(getClass().getResource("/com/bana274/male.png").toExternalForm()));
                    } else {
                        classifiedGenderImageView.setImage(new Image(getClass().getResource("/com/bana274/female.png").toExternalForm()));
                    }
                    
                    classifiedOverlayAnchorPane.toFront();
                });
            } catch (JsonProcessingException ex) {
                Logger.getLogger(MainController.class.getName()).log(Level.SEVERE, null, ex);
            }
        });
    }

    /**
     * Application related methods.
     *
     * @return
     */
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
                    if (timer != null) {
                        timer.cancel();
                    }
                    Platform.exit();
                    System.exit(0);
                });
    }

}
