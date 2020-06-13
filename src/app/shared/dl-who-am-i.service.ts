import { Injectable, OnInit } from '@angular/core';


import * as tfl from '@tensorflow/tfjs-layers';
import * as tfc from '@tensorflow/tfjs-core';

// Add the WASM backend to the global backend registry.
import * as tf from '@tensorflow/tfjs';
import * as tfbackend from '@tensorflow/tfjs-backend-webgl';


import { categoricalCrossentropy } from '@tensorflow/tfjs-layers/dist/exports_metrics';

import { ToastController } from '@ionic/angular';
import { File } from '@ionic-native/file/ngx';
import { Platform } from '@ionic/angular';
import { IPersonRep, ITrainingMetrics } from '../core/interfaces';
import { HttpClient } from '@angular/common/http';

const TRAIN_FOLDER_PATH = 'www/assets/data/train';
const TEST_FOLDER_PATH = 'www/assets/data/test';

@Injectable({
  providedIn: 'root'
})
export class DlWhoAmIService {

  ProgressBarValue: number;
  files = [];
  PersonRep: Array<IPersonRep> = new Array<IPersonRep>();
  AuxPersonRep: IPersonRep;
  public traningMetrics: ITrainingMetrics[] = []; //instance of TrainingMetrics

  public mobilenetModified: tfl.LayersModel
  public x_train: tf.Tensor;
  public y_train: tf.Tensor;

  public images: tf.Tensor[] = [];
  public targets: tf.Tensor[] = [];

  constructor(
    public toastController: ToastController,
    private file: File,
    private plt: Platform,
    private http: HttpClient
  ) { }

  /**
   * Shows info of the process
   * @param arg0 : message
   * @param arg1 
   */
  async presentToast(arg0: string, arg1: string) {
    const toast = await this.toastController.create({
      message: arg0+"\n  "+arg1 ,
      duration: 2000
    });
    toast.present();
  }
  /**
   * Initialization, set backend to webgl 
   * launches the training process on startup of the application
   */
  async main() {
    this.images = []
    this.targets = []
    this.PersonRep = []

    // Set the backend to (webgl) and wait for the module to be ready.
    tf.setBackend('webgl').then(() =>
      this.train()

    );// Correction for: No backend found for registry

  }

  /**
   * calls generateData() to prepare the training dataset
   * calls getModfiedMobilenet() to prepare the model for training
   * calls fineTuneModifiedModel() to finetune the model
   */
  async train() {
    console.log("Training")
    this.generateData("Training");

    this.ProgressBarValue = 35;
    this.presentToast("Images are loaded into the memory as tensor!", "Ready");

    this.mobilenetModified = await this.getModifiedMobilenet();
    this.ProgressBarValue = 50;
    this.presentToast("Modefiled Mobilenet AI Model is loaded!", "Steady");

    // setTimeout(() => {
    this.fineTuneModifiedModel(this.mobilenetModified);
    // }, 2000)

    this.presentToast("Model training is completed !", "Go");
    this.ProgressBarValue = 100;
  }
  /**
    * calls generateData() to prepare the testing dataset
    * makes prediction
    */
  async test() {
    console.log("Testing")
    this.generateData("Testing");

    this.ProgressBarValue = 35;
    this.presentToast("Images are loaded into the memory as tensor !", "Close");

    this.presentToast("Model testing is completed !", "Close");
    this.ProgressBarValue = 100;
    return this.traningMetrics;
  }
  /**
  * Get the highest confidence prediction from our model
  * @param predictim 
  */
  async predict(predictim) {
    let result: any = await this.mobilenetModified.predict(predictim);
    console.log(result)
    console.log(result);
    console.log("result.argMax()  :" + result.argMax());
    const winner = this.PersonRep[result.argMax().dataSync()[0]];

    // Display the winner
    console.log(winner);
    this.presentToast(`
                      Class: ${winner.Label}
                      
                      `, "");
  }
  
  async predict_test(predictim) {
    let result: any = await this.mobilenetModified.predict(predictim);
    console.log(result)
    console.log(result);
    console.log("result.argMax()  :" +  result.argMax());

    result.forEach(element => {
      let winner :any = this.PersonRep[element.dataSync()[0]];

      // Display the winner
      console.log(winner);
      this.traningMetrics.push(winner)
    });
  }
  
  //-------------------------------------------------------------
  //region Picture treatment methods
  //
  // normalizeImageToTensor
  // capture
  //-------------------------------------------------------------
  /**
   * converts images into  tensor
   * takes Image In HTMLImageElement as argument
   * @param picture 
   */
  normalizeImageToTensor(picture): tfc.Tensor<tfc.Rank> {
    // Reads the image as a Tensor from the <image> element.
    let trainImage = tfc.browser.fromPixels(picture);
    // Normalize the image between -1 and 1. The image comes in between 0-255,
    // so we divide by 127 and subtract 1.
    return trainImage.toFloat().div(tfc.scalar(127)).sub(tfc.scalar(1));
  }
  /**
   * Gets image in HTMLImageElement, convert it into tensor
   * gets prediction
   * @param picture 
   */
  capture(picture) {
    // Normalize the image between -1 and 1. The image comes in between 0-255,
    // so we divide by 127 and subtract 1.
    let tpredictim = this.normalizeImageToTensor(picture)
    let exp_predict_image = tf.expandDims(tpredictim, 0)
    // let predict_image: any = this.mobilenetModified.predict(exp_predict_image) as tfc.Tensor<tfc.Rank>
    this.predict(exp_predict_image)

  }

  //-------------------------------------------------------------
  //region Data treatment methods
  //
  // generateData
  // loadFilesTraingin
  // loadFilesTest
  //-------------------------------------------------------------
  /**
   * this function generate input and target tensors for the training
   * input images tensor is produced from 224x224x3 image in HTMLImageElement
   * target tensor shape1 is produced from the class definition
   */
  generateData(imageFolderName) {
    this.plt.ready().then(() => {

      let path = this.file.applicationDirectory;
      if (imageFolderName == "Training") {
        this.file.checkDir(path, TRAIN_FOLDER_PATH).then(
          () => {
            this.loadFilesTraining(this.file);
          },
          err => {
            this.file.createDir(path, TRAIN_FOLDER_PATH, false);
          }
        );
      } else {
        // Testing
        this.file.checkDir(path, TEST_FOLDER_PATH).then(
          () => {
            this.loadFilesTesting(this.file);
          },
          err => {
            this.file.createDir(path, TEST_FOLDER_PATH, false);
          }
        );
      }
    });
  }

  /**
   * TODO: Implentation for the web version
   * Gets training images from URL
   * @param picture 
   */
  // generateData() {
  //   this.http.get('http://localhost:8000/assets/data/train')
  //   .subscribe(
  //     data => console.log('success', data),
  //     error => console.log('oops', error)
  //   );
  // }
  /**
   * Generate training dataset from data/train folder in assets
   */
  private directoryPaths: Array<string> = Array<string>();
  loadFilesTraining(file) {

    file.listDir(this.file.applicationDirectory, TRAIN_FOLDER_PATH).then(
      result => {
        // Folder names in training directory are our celebrities
        result.forEach((file, index) => {
          if (file.isDirectory == true && file.name != '.' && file.name != '..') {
            // Code if its a folder
            let path = this.file.applicationDirectory + TRAIN_FOLDER_PATH + '/' + file.name// File path
            this.directoryPaths.push(TRAIN_FOLDER_PATH + '/' + file.name)
            this.AuxPersonRep = { ImageSrc: path, ImageName: file.name, Class: index, Label: file.name }
            this.PersonRep.push(this.AuxPersonRep);
          }
        },
          err => console.log('error loading files: ', err)
        );
        // After we have the people name we need to recover photos
        this.directoryPaths.forEach((f: string, index) => {
          let p: string = f
          file.listDir(this.file.applicationDirectory, p).then(
            result => {
              result.forEach((imageData, idx) => {
                let image = new Image();   // Create new img element
                image.src = imageData;
                let canvas = document.createElement('canvas');
                let context = canvas.getContext('2d');
                // Is need to set same dimensions used in Neuronal Network
                image.width = 224;
                image.height = 224;
                image.crossOrigin = "Anonymous";

                let imageTensor = this.normalizeImageToTensor(image)
                this.images.push(imageTensor)

                let targetTensor = tfc.tensor1d([this.PersonRep[index].Class]);
                this.targets.push(targetTensor)
              });
            });

        });
      })
  }
  /**
     * Generate testing dataset from data/test folder in assets
     */
  loadFilesTesting(file) {

    file.listDir(this.file.applicationDirectory, TEST_FOLDER_PATH).then(
      result => {
        result.forEach((imageData, idx) => {
          let image = new Image();   // Create new img element
          image.src = imageData;
          let canvas = document.createElement('canvas');
          let context = canvas.getContext('2d');
          // Is need to set same dimensions used in Neuronal Network
          image.width = 224;
          image.height = 224;
          image.crossOrigin = "Anonymous";

          // Makes predicion
          // Normalize the image between -1 and 1. The image comes in between 0-255,
          // so we divide by 127 and subtract 1.
          let tpredictim = this.normalizeImageToTensor(image)
          let exp_predict_image = tf.expandDims(tpredictim, 0)
          this.predict_test(exp_predict_image)

        });
      });
  }
  //-------------------------------------------------------------
  //region Deep Learning methods
  //
  // getModifiedMobilenet
  // freezeModelLayers
  // fineTuneModifiedModel
  //-------------------------------------------------------------
  /**
   * modifies the pre-trained mobilenet 
   * cells, freezes layers to train only the last couple of layers
   */
  async getModifiedMobilenet() {
    const trainableLayers = ['denseModified', 'conv_pw_13_bn', 'conv_pw_13', 'conv_dw_13_bn', 'conv_dw_13'];
    const mobilenet = await tfl.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    console.log('Mobilenet model is loaded')

    const x = mobilenet.getLayer('global_average_pooling2d_1');
    // units => number of classes for output prediction ___
    const predictions = <tfl.SymbolicTensor>tfl.layers.dense(
      { units: 1, activation: 'softmax', name: 'denseModified' }
    ).apply(x.output);
    let mobilenetModified = tfl.model({ inputs: mobilenet.input, outputs: predictions, name: 'modelModified' });
    console.log('Mobilenet model is modified')

    mobilenetModified = this.freezeModelLayers(trainableLayers, mobilenetModified)
    console.log('ModifiedMobilenet model layers are freezed')

    mobilenetModified.compile({ loss: categoricalCrossentropy, optimizer: tf.train.adam(1e-3), metrics: ['accuracy', 'crossentropy'] });

    return mobilenetModified
  }
  /**
   * freezes mobilenet layers to make them untrainable
   * just keeps final layers trainable with argument trainableLayers
   * @param trainableLayers 
   * @param mobilenetModified 
   */
  freezeModelLayers(trainableLayers, mobilenetModified) {
    for (const layer of mobilenetModified.layers) {
      layer.trainable = false;
      for (const tobeTrained of trainableLayers) {
        if (layer.name.indexOf(tobeTrained) === 0) {
          layer.trainable = true;
          break;
        }
      }
    }
    return mobilenetModified;
  }

  /**
   * finetunes the modified mobilenet model in 5 training batches
   * takes model, images and targets as arguments
   * @param model 
   * @param images 
   * @param targets 
   */
  async fineTuneModifiedModel(model) {
    function onBatchEnd(batch, logs) {
      console.log('Accuracy', logs.acc);
      console.log('CrossEntropy', logs.ce);
      console.log('All', logs);
    }
    console.log('Finetuning the model...');

    // Stacks a list of rank-R tf.Tensors into one rank-(R+1) tf.Tensor.
    let X = this.images as tf.Tensor[]
    this.x_train = tf.stack(X)
    let y: any = this.targets as tf.Tensor[]
    this.y_train = tf.stack(y)

    await model.fit(this.x_train, this.y_train,
      {
        epochs: 5,
        batchSize: 24,
        validationSplit: 0.2,
        callbacks: { onBatchEnd }

      }).then(info => {
        console.log
        console.log('Final accuracy', info.history.acc);
        console.log('Cross entropy', info.ce);
        console.log('All', info);
        console.log('All', info.history['acc'][0]);

        for (let k = 0; k < 3; k++) {
          this.traningMetrics.push({ acc: 0, ce: 0, loss: 0 });

          this.traningMetrics[k].acc = info.history['acc'][k];
          this.traningMetrics[k].ce = info.history['ce'][k];
          this.traningMetrics[k].loss = info.history['loss'][k];
        }
        //this.x_train.dispose();
        //this.y_train.dispose();
        // model.dispose();
      });;
  }
}
