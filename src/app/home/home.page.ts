import { Component, OnInit } from '@angular/core';

import { DlWhoAmIService } from '../shared/dl-who-am-i.service';

import { Camera, CameraOptions } from '@ionic-native/camera/ngx';
import { Platform } from  '@ionic/angular';
import { DomSanitizer} from '@angular/platform-browser'
import { ITrainingMetrics } from '../core/interfaces';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage implements OnInit{

  public traningMetrics: ITrainingMetrics[] = []; //instance of TrainingMetrics
  public isTest:boolean = false;
  public isTrained:boolean = false;
  constructor( private camera: Camera, private dlService: DlWhoAmIService, public platform:Platform, public sanitizer: DomSanitizer) {  }

  ngOnInit(){
    console.log("Starting Facial Recognition system")
    this.isTrained = false
    this.dlService.main()  
    this.isTrained = true 
  }

   public options: CameraOptions = {
    quality: 100,
    destinationType: this.camera.DestinationType.FILE_URI,
    encodingType: this.camera.EncodingType.JPEG,
    mediaType: this.camera.MediaType.PICTURE
  }

   takePicture() {  
    this.isTest = false;  
    this.camera.getPicture(this.options).then((imageData) => {
      // imageData is either a base64 encoded string or a file URI
      // If it's base64 (DATA_URL):
      //let base64Image = 'data:image/jpeg;base64,' + imageData;
      let image = new Image();   // Create new img element
      let safeurl = this.sanitizer.bypassSecurityTrustUrl(imageData);

      image.src = safeurl as string;
      let canvas = document.createElement('canvas');
      let  context = canvas.getContext('2d');
      // Need to set dimensions used in Neural Network
      image.width = 224;
      image.height = 224;
      image.crossOrigin = "Anonymous";
      
      this.dlService.capture(image)


     }, (err) => {
      // Handle error
     });
  }

  async test() {
    this.isTest = true;
    this.dlService.test().then(result=>this.traningMetrics  = result   )
  }

}
