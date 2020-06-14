
## Who am I?

Aplicación que muestra el parecido de la persona en la foto.
Se ha desarrollado una pequeña aplicación para poner a prueba el uso de TensorFlow JavaScript.
La red reunonal utilizada ha sido mobileNet.Se ha implementado un fine tuning para adaptar esta red a la detección de nuestros personajes personalizados. La red se ha entrenado con 3 fotos de [3 personajes](./dl-upna-who-am-i.html/src/assets/data); Bill Gates, Larry Page y Mark Zuckerberg. Los resultados de la red neuronal no son buenos, se obtiene una precisión del 14%.

    //Una vez clonado el proyecto:
    git clone https://github.com/afrago/dl-upna-who-am-i
    //Se obtienen las dependencias: 
    npm install
    //Se puede lanzar la vesión android con: 
    ionic cordova run android
    // o la versión web: 
    // ionic cordova run browser
    

La conclusión con esta versión del producto final es que en este tipo de aplicaciones es más recomentable trasladar la realización de los cálculos a un servidor. El uso de versiones JS para el navegador solo sería una opción en caso de no tener conexión a Internet, lo cual no es lo más habitual cuando utilizamos dispositivos móviles.

Por otro lado, en proximas versiones se analizará la realización del proyecto con [ReactNative](https://blog.tensorflow.org/2020/02/tensorflowjs-for-react-native-is-here.html) ya que tensorflow.js dispone de una versión específica para este entorno.

