/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.bana274;

import com.bana274.controllers.MainController;
import java.io.IOException;
import java.util.ResourceBundle;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.stage.Stage;

/**
 *
 * @author brianj
 */
public class Main extends Application {
    
    private final ResourceBundle bundle = ResourceBundle.getBundle("com.bana274.main");
    private final String APP_ICON = "favicon-32x32.png";

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws Exception {
        initControls(stage);
    }

    private void initControls(Stage stage) throws IOException {
        FXMLLoader loader = new FXMLLoader();
        loader.setLocation(getClass().getResource("main.fxml"));
        loader.setResources(bundle);
        Parent root = (Parent) loader.load();
        MainController controller = loader.getController();
        stage.setTitle(bundle.getString("window-title"));
        stage.setScene(new Scene(root));
        stage.setMaximized(true);
        stage.sizeToScene();
        stage.getIcons().add(new Image(Main.class.getResource(APP_ICON).toExternalForm()));
        stage.setOnCloseRequest(controller.getWindowCloseEventHandler());
        stage.show();
    }

}
