/* 
 *  ===============================================
  Example sketch using the Intel CurieIMU library and the General Vision CurieNeurons library 
  for Intel(R) Curie(TM) devices

  Motion is converted into a simple feature vector as follows:
  [ax'1, ay'1, az'1, gx'1,gy'1, gz'1, ax'2, ay'2, az'2, gx'2, gy'2, gz'2, ...] over a number of time samples
  Note that the values a' and g' are normalized and expand between the running min and max of the a and g signals.

  After calibration is made,
  Use the serial monitor to edit the category of a motion, 
    (ex= 1 for vertical, 2 for horizontal, 0 for stillness or anything else),
  Start moving the Curie in an expected direction,
  and when you press Enter the last feature vector is learned.
  
  Note that this "snapshot" approach is simplistic and you may have to teach several times
  a given motion so the neurons store models with different amplitudes, acceleration, etc.
  Ideally we want to learn consecutives vectors for a few seconds.
  
  ===============================================
*/
#include "CurieIMU.h"
#include <CurieBLE.h>
#include <CurieNeurons.h>
#include <Adafruit_NeoPixel.h>

//RGB
// Which pin on the Arduino is connected to the NeoPixels?
// On a Trinket or Gemma we suggest changing this to 1
#define PIN            5
// How many NeoPixels are attached to the Arduino?
#define NUMPIXELS      1
// When we setup the NeoPixel library, we tell it how many pixels, and which pin to use to send signals.
// Note that for older NeoPixel strips you might need to change the third parameter--see the strandtest
// example for more information on possible values.
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);
int delayval = 50; // delay for half a second
//
const int ledPin = 13; // set ledPin to use on-board LED
const int ledPin_wikit = 14; // set ledPin to use on-board LED
int neuron_count = 0;
//WIKIT FUSIONWIRE NETWORK
#define MP6050_DATA_TYPE 0x80
#define TEMP_DATA_TYPE  0x82
#define KEY_DATA_TYPE 0x83
#define LIGHT_DATA_TYPE 0x84
//WIKIT bit position
#define WIKIT_HEADER 0x0
#define WIKIT_TEMP 0x1
#define WIKIT_KEY 0x3
#define WIKIT_LIGHT 0x5
//#define WIKIT_MOTION_NEURON 0xB
#define WIKIT_MOTION_NEURON 0x8
volatile bool NO_FUSIONWIRE_MP6050 = true;

volatile bool WIKIT_MP6050_UPDATED = false;
volatile bool WIKIT_LIGHT_UPDATED = false;
volatile bool WIKIT_TEMP_UPDATED = false;
volatile bool WIKIT_KEY_UPDATED = false;

volatile bool WIKIT_LIGHT_REQUIRED = false;
volatile bool WIKIT_TEMP_REQUIRED = false;
volatile bool WIKIT_KEY_REQUIRED = false;

volatile bool WIKIT_LED_ON_REQUIRED = false;
volatile bool WIKIT_LED_OFF_REQUIRED = false;

volatile bool WIKIT_RGB_R_REQUIRED = false;
volatile bool WIKIT_RGB_G_REQUIRED = false;
volatile bool WIKIT_RGB_B_REQUIRED = false;

volatile bool WIKIT_MOTION_NEURON_CAT_REQUIRED = false;
volatile bool WIKIT_MOTION_NEURON_SEND_CAT_REQUIRED = false;

volatile bool WIKIT_MOTION_NEURON_MODE1_LEARN_REQUIRED = false;
volatile bool WIKIT_MOTION_NEURON_MODE2_LEARN_REQUIRED = false;
volatile bool WIKIT_MOTION_NEURON_MODE3_LEARN_REQUIRED = false;

volatile bool WIKIT_MODE1_LEARNING_FINISHED = false;
volatile bool WIKIT_MODE2_LEARNING_FINISHED = true;

volatile bool WIKIT_LEARNING_FINISHED = false;
volatile bool WIKIT_NEURON_RESTARTED = false;



unsigned char wikit_light;
unsigned char wikit_temp;
unsigned char wikit_key;

unsigned char wikit_send_value[9];
unsigned char wikit_latest_recognize;

////BLE
BLEPeripheral blePeripheral; // create peripheral instance
BLEService wikitService("00001523-1212-EFDE-1523-785FEABCD123"); // create service
BLECharacteristic wikitDataChar("00001526-1212-EFDE-1523-785FEABCD123",  BLEWrite | BLERead | BLENotify,16);
BLECharacteristic wikitLedSettingChar("00001528-1212-EFDE-1523-785FEABCD123", BLERead | BLEWrite | BLENotify,16);
BLECharacteristic wikitMultilinkPeripheralChar("00001531-1212-EFDE-1523-785FEABCD123", BLERead | BLEWrite | BLENotify,16);


//NEURON
int ax, ay, az;         // accelerometer values
int gx, gy, gz;         // gyrometer values
int calibrateOffsets = 1; // int to determine whether calibration takes place or not
CurieNeurons hNN;

int catL=0; // category to learn
int prevcat=0; // previously recognized category
int dist, cat, nid, nsr, ncount; // response from the neurons


//
// Variables used for the calculation of the feature vector
//
#define sampleNbr 10  // number of samples to assemble a vector
#define signalNbr  6  // ax,ay,az,gx,gy,gz
int raw_vector[sampleNbr*signalNbr]; // vector accumulating the raw sensor data
byte vector[sampleNbr*signalNbr]; // vector holding the pattern to learn or recognize
int mina=0xFFFF, maxa=0, ming=0xFFFF, maxg=0, da, dg;


void setup() 
{
  //MQBAO Serial.begin(9600); // initialize Serial communication
  // initialize device
  pixels.begin(); // This initializes the NeoPixel library.
  //MQBAO Serial.println("Initializing IMU device...");
  CurieIMU.begin();
  
  
  // mqbao calibration happens in sensors automatically. indicated by LED on wikit zero wire
  // use the code below to calibrate accel/gyro offset values
  if (calibrateOffsets == 1) 
  {    
    //MQBAO Serial.println("About to calibrate. Make sure your board is stable and upright");
    delay(5000);
    //MQBAO Serial.print("Starting Gyroscope calibration and enabling offset compensation...");
    CurieIMU.autoCalibrateGyroOffset();
    //MQBAO Serial.println(" Done");
    //MQBAO Serial.print("Starting Acceleration calibration and enabling offset compensation...");
    CurieIMU.autoCalibrateAccelerometerOffset(X_AXIS, 0);
    CurieIMU.autoCalibrateAccelerometerOffset(Y_AXIS, 0);
    CurieIMU.autoCalibrateAccelerometerOffset(Z_AXIS, 1);
    //MQBAO Serial.println(" Done");
  }
  
  
  // Initialize the neurons and set a conservative Max Influence Field
  hNN.Init();
  hNN.Forget(1000); //set a conservative  Max Influence Field prior to learning
  //int value=hNN.MAXIF(); // read the MAXIF back to verify proper SPI communication
  //Serial.print("\nMaxif register=");Serial.print(value);
  //while (!Serial);    // wait for the serial port to open
 // set the local name peripheral advertises
 //BLE_START
  blePeripheral.setLocalName("CurieNeuron");
  blePeripheral.setDeviceName("CurieNeuron");
  // set the UUID for the service this peripheral advertises
  blePeripheral.setAdvertisedServiceUuid(wikitService.uuid());
  //blePeripheral.setAdvertisedServiceUuid("0901");
  // add service and characteristic
  blePeripheral.addAttribute(wikitService);
  blePeripheral.addAttribute(wikitDataChar);
  blePeripheral.addAttribute(wikitLedSettingChar);
  blePeripheral.addAttribute(wikitMultilinkPeripheralChar);
  // assign event handlers for connected, disconnected to peripheral
  blePeripheral.setEventHandler(BLEConnected, blePeripheralConnectHandler);
  blePeripheral.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);

  // assign event handlers for characteristic
  wikitDataChar.setEventHandler(BLEWritten, wikitDataCharacteristicWritten);
  wikitLedSettingChar.setEventHandler(BLEWritten,wikitLedSettingCharcteristicWritten);
  wikitMultilinkPeripheralChar.setEventHandler(BLEWritten,wikitMultilinkPeripheralCharcteristicWritten); 
  //BLE END
  // advertise the service
  digitalWrite(ledPin,LOW);
  digitalWrite(ledPin_wikit,LOW);
  
  blePeripheral.begin();
  //MQBAO Serial.println(("Bluetooth device active, waiting for connections..."));
  
}

void loop() 
{   
    blePeripheral.poll();
    //delay(10);
    
    // Learn if push button depressed and report if a new neuron is committed
    //if (WIKIT_LIGHT_REQUIRED){
    //while (!WIKIT_LIGHT_UPDATED){};
    if(WIKIT_LIGHT_REQUIRED){   //mqbao FIXME in app side later
      wikit_send_value[WIKIT_HEADER] = 0x55;
      //wikit_send_value[5] = 0x55;
      wikit_send_value[WIKIT_LIGHT] = wikit_light;
      wikit_send_value[WIKIT_TEMP] = wikit_temp;
      wikit_send_value[WIKIT_KEY] = wikit_key;
      //wikit_send_value[3] = 0x22;
      wikitDataChar.setValue(wikit_send_value,sizeof(wikit_send_value));
      WIKIT_LIGHT_UPDATED = false;
      //Serial.print("\nLight data sended");
    
      WIKIT_LIGHT_REQUIRED = false;
      delay(66);
    }

     if(WIKIT_TEMP_REQUIRED){ //mqbao FIXME in app side later

      wikit_send_value[WIKIT_TEMP] = wikit_temp;
        
      wikit_send_value[WIKIT_HEADER] = 0x55;
      //wikit_send_value[5] = 0x55;
      wikit_send_value[WIKIT_LIGHT] = wikit_light;
      wikit_send_value[WIKIT_TEMP] = wikit_temp;
      wikit_send_value[WIKIT_KEY] = wikit_key;
      //wikit_send_value[3] = 0x22;
      wikitDataChar.setValue(wikit_send_value,sizeof(wikit_send_value));
      WIKIT_TEMP_UPDATED = false;
      //Serial.print("\nLight data sended");
    
      WIKIT_TEMP_REQUIRED = false;
      delay(77);
    }

    if(WIKIT_KEY_REQUIRED){ //mqbao FIXME in app side later
      wikit_send_value[WIKIT_HEADER] = 0x55;
      //wikit_send_value[5] = 0x55;
      wikit_send_value[WIKIT_LIGHT] = wikit_light;
      wikit_send_value[WIKIT_TEMP] = wikit_temp;
      wikit_send_value[WIKIT_KEY] = wikit_key;
      //wikit_send_value[3] = 0x22;
      wikitDataChar.setValue(wikit_send_value,sizeof(wikit_send_value));
      WIKIT_KEY_UPDATED = false;  
      WIKIT_KEY_REQUIRED = false;
      delay(88);
    }
    //}
      
    if (WIKIT_LED_ON_REQUIRED){
      digitalWrite(ledPin_wikit,HIGH);
      WIKIT_LED_ON_REQUIRED = false;
    }
  
    if (WIKIT_LED_OFF_REQUIRED){
      digitalWrite(ledPin_wikit,LOW);
      WIKIT_LED_OFF_REQUIRED = false;
     }  
     
     if (WIKIT_RGB_G_REQUIRED){
      pixels.setPixelColor(0, pixels.Color(0,30,0)); // Moderately bright green color.
      pixels.show(); // This sends the updated pixel color to the hardware.
      delay(delayval);
      WIKIT_RGB_G_REQUIRED = false;
    }

    if (WIKIT_RGB_R_REQUIRED){
      pixels.setPixelColor(0, pixels.Color(30,0,0)); // Moderately bright green color.
      pixels.show(); // This sends the updated pixel color to the hardware.
      delay(delayval);
      WIKIT_RGB_R_REQUIRED = false;
    }

   if (WIKIT_RGB_B_REQUIRED){
      pixels.setPixelColor(0, pixels.Color(0,0,30)); // Moderately bright green color.
      pixels.show(); // This sends the updated pixel color to the hardware.
      delay(delayval);
      WIKIT_RGB_B_REQUIRED = false;
    }   
    
    if (WIKIT_MOTION_NEURON_SEND_CAT_REQUIRED){
         wikit_send_value[WIKIT_HEADER] = 0x55;
         wikit_send_value[WIKIT_MOTION_NEURON] = wikit_latest_recognize;  //means ERROR
         wikitDataChar.setValue(wikit_send_value,sizeof(wikit_send_value));     
         delay(51);
    }
    
    /////NEURON LEARNING/CATOGRORIZING DEFAULT IS OFF 
    if (!WIKIT_MODE1_LEARNING_FINISHED || !WIKIT_MODE2_LEARNING_FINISHED){
    if (WIKIT_MOTION_NEURON_MODE1_LEARN_REQUIRED) 
    {
       //MQBAO Serial.print("\nLearning motion category mode1"); //Serial.print(catL);
        // learn 5 consecutive sample vectors
       for (int j=0; j<2; j++){
        for (int i=0; i<5; i++)
        {
          getVector(false); // the vector array is a global
          //Serial.print("\nVector = ");
          //for (int i=0; i<signalNbr*sampleNbr; i++) {Serial.print(vector[i]);Serial.print("\t");}
          ncount=hNN.Learn(vector, sampleNbr*signalNbr, 1);
         wikit_send_value[WIKIT_HEADER] = 0x55;
         wikit_send_value[WIKIT_MOTION_NEURON] = ((neuron_count+1)*20);  //means success
         wikitDataChar.setValue(wikit_send_value,sizeof(wikit_send_value));
         neuron_count++;
        }
        //MQBAO Serial.print("\tNeurons="); Serial.print(ncount);  
        }
        WIKIT_MOTION_NEURON_MODE1_LEARN_REQUIRED = false;
        //mqbao FIXME need to send back data here     
         wikit_send_value[WIKIT_HEADER] = 0x55;
         wikit_send_value[WIKIT_MOTION_NEURON] = 255;  //means success
         wikitDataChar.setValue(wikit_send_value,sizeof(wikit_send_value));
         WIKIT_MODE1_LEARNING_FINISHED = true;
         neuron_count = 0;
    }
    
    else if (WIKIT_MOTION_NEURON_MODE2_LEARN_REQUIRED) 
    {
        //MQBAO Serial.print("\nLearning motion category mode1"); 
        // learn 5 consecutive sample vectors
        for (int i=0; i<5; i++)
        {
          getVector(true); // the vector array is a global
          //MQBAO Serial.print("\nVector = ");
          //MQBAO for (int i=0; i<signalNbr*sampleNbr; i++) {Serial.print(vector[i]);Serial.print("\t");}
          ncount=hNN.Learn(vector, sampleNbr*signalNbr, 2);
        }
       //MQBAO Serial.print("\tNeurons="); Serial.print(ncount);  
        WIKIT_MOTION_NEURON_MODE2_LEARN_REQUIRED = false;
        //mqbao FIXME need to send back data here     
         wikit_send_value[WIKIT_HEADER] = 0x55;
         wikit_send_value[WIKIT_MOTION_NEURON] = 55;  //means success
         wikitDataChar.setValue(wikit_send_value,sizeof(wikit_send_value));
         WIKIT_MODE2_LEARNING_FINISHED = true;
    }
    WIKIT_LEARNING_FINISHED = WIKIT_MODE1_LEARNING_FINISHED & WIKIT_MODE2_LEARNING_FINISHED;
    }
    else  //NEURON CATOGROTY
    {
      if (WIKIT_MOTION_NEURON_CAT_REQUIRED){
      // Recognize
      getVector(false); // the vector array is a global
      hNN.Classify(vector, sampleNbr*signalNbr,&dist, &cat, &nid);
     // if (cat!=prevcat)
      {
        if (cat!=0x7FFF)
        {
          //Serial.print("\nMotion category #"); Serial.print(cat);
         //mqbao FIXME need to send back data here   
         if (NO_FUSIONWIRE_MP6050){
           //delay(1000);  
         }
         //for (int i = 0; i < 10; i ++){
         wikit_send_value[WIKIT_HEADER] = 0x55;
         wikit_send_value[WIKIT_MOTION_NEURON] = cat;  //means success
         wikit_latest_recognize = cat;
         wikitDataChar.setValue(wikit_send_value,sizeof(wikit_send_value));
        // }
        }
        else
        { 
          //Serial.print("\nMotion unknown"); 
          //mqbao FIXME need to send back data here  
         if (NO_FUSIONWIRE_MP6050){
           //delay(1000);    
         }
        // for (int i = 0; i < 10; i ++){
         wikit_send_value[WIKIT_HEADER] = 0x55;
         wikit_send_value[WIKIT_MOTION_NEURON] = 33;  //means ERROR
         wikit_latest_recognize = 33;
         wikitDataChar.setValue(wikit_send_value,sizeof(wikit_send_value));     
        // }
        }
        prevcat=cat;
      }  
      WIKIT_MOTION_NEURON_CAT_REQUIRED = false;
      }
      
    }
    /////NEURON LEARNING/CATOGRORIZING DEFAULT IS OFF
   
    
}  

void getVector(bool mode)
{
  // the reset of the min and max values is optional depending if you want to
  // use a running min and max from the launch of the script or not
  mina=0xFFFF, maxa=0, ming=0xFFFF, maxg=0, da, dg;
  
  for (int sampleId=0; sampleId<sampleNbr; sampleId++)
  {
    //Build the vector over sampleNbr and broadcast to the neurons
    if (NO_FUSIONWIRE_MP6050){
    CurieIMU.readMotionSensor(ax, ay, az, gx, gy, gz);
    }else{
    if (mode){
    while(WIKIT_MP6050_UPDATED == false);
    //Serial.print("\n mqbao get mp6050 data success");
    WIKIT_MP6050_UPDATED = false; // waiting for another update
    }else{
    delay(2);
    }
    }
    
    // update the running min/max for the a signals
    if (ax>maxa) maxa=ax; else if (ax<mina) mina=ax;
    if (ay>maxa) maxa=ay; else if (ay<mina) mina=ay;
    if (az>maxa) maxa=az; else if (az<mina) mina=az;    
    da= maxa-mina;
    
    // update the running min/max for the g signals
    if (gx>maxg) maxg=gx; else if (gx<ming) ming=gx;
    if (gy>maxg) maxg=gy; else if (gy<ming) ming=gy;
    if (gz>maxg) maxg=gz; else if (gz<ming) ming=gz;   
    dg= maxg-ming;

    // accumulate the sensor data
    raw_vector[sampleId*signalNbr]= ax;
    raw_vector[(sampleId*signalNbr)+1]= ay;
    raw_vector[(sampleId*signalNbr)+2]= az;
    raw_vector[(sampleId*signalNbr)+3]= gx;
    raw_vector[(sampleId*signalNbr)+4]= gy;
    raw_vector[(sampleId*signalNbr)+5]= gz;
    //delay(10);
  }
  
  // normalize vector
  for(int sampleId=0; sampleId < sampleNbr; sampleId++)
  {
    for(int i=0; i<3; i++)
    {
      vector[sampleId*signalNbr+i]  = (((raw_vector[sampleId*signalNbr+i] - mina) * 255)/da) & 0x00FF;
      vector[sampleId*signalNbr+3+i]  = (((raw_vector[sampleId*signalNbr+3+i] - ming) * 255)/dg) & 0x00FF;
    }
  }
}

void blePeripheralConnectHandler(BLECentral& central) {
  // central connected event handler
  //MQBAO Serial.print("Connected event, central: ");
  //MQBAO Serial.println(central.address());  
}

void blePeripheralDisconnectHandler(BLECentral& central) {
  // central disconnected event handler
  //MQBAO Serial.print("Disconnected event, central: ");
  //MQBAO Serial.println(central.address());
  digitalWrite(ledPin,LOW);
  digitalWrite(ledPin_wikit,LOW);
  NO_FUSIONWIRE_MP6050 = true;
  blePeripheral.begin();
}

void wikitDataCharacteristicWritten(BLECentral& central, BLECharacteristic& characteristic) {
  // central wrote new value to characteristic, update LED
  //MQBAO Serial.print("\nDataCharacteristic event, written: ");
  //digitalWrite(ledPin,HIGH);
  //MQBAO Serial.print(wikitDataChar.value()[0]);
  if (wikitDataChar.value()[0] == 0x0A){  
    if (!WIKIT_LEARNING_FINISHED){
      //Serial.print(" WIKIT_MOTION_NEURON_MODE1_LEARN_REQUIRED");
      WIKIT_MOTION_NEURON_MODE1_LEARN_REQUIRED = true;
    }
    else{
      WIKIT_MOTION_NEURON_CAT_REQUIRED = true;
      WIKIT_MOTION_NEURON_SEND_CAT_REQUIRED = true;
  }
  }else if (wikitDataChar.value()[0] == 0x0B){
    if (!WIKIT_LEARNING_FINISHED){
      //MQBAO Serial.print(" WIKIT_MOTION_NEURON_MODE2_LEARN_REQUIRED");
      WIKIT_MOTION_NEURON_MODE2_LEARN_REQUIRED = true;
    }
    else{
      WIKIT_MOTION_NEURON_SEND_CAT_REQUIRED = true;
      WIKIT_MOTION_NEURON_CAT_REQUIRED = true;
      
  }
  }else if (wikitDataChar.value()[0] == 0x3F){
       WIKIT_NEURON_RESTARTED = true;
       WIKIT_MOTION_NEURON_MODE1_LEARN_REQUIRED = false;
       //WIKIT_MOTION_NEURON_MODE2_LEARN_REQUIRED = false;
       WIKIT_MOTION_NEURON_CAT_REQUIRED = false;
       WIKIT_MOTION_NEURON_SEND_CAT_REQUIRED = false; //mqbao 0721 added
       WIKIT_MODE1_LEARNING_FINISHED = false;
      //MQBAO Serial.println("mqbao delete");
       //WIKIT_MODE2_LEARNING_FINISHED = false; //otherwise the case will hang up
  } else if (wikitDataChar.value()[0] == 0x03){
    WIKIT_LIGHT_REQUIRED = true;
  } else if (wikitDataChar.value()[0] == 0x38){
    WIKIT_LIGHT_REQUIRED = false;
  } else if (wikitDataChar.value()[0] == 0x01){
    WIKIT_TEMP_REQUIRED = true;
  } else if (wikitDataChar.value()[0] == 0x36){
    WIKIT_TEMP_REQUIRED = false;
  } else if (wikitDataChar.value()[0] == 0x08){
    WIKIT_KEY_REQUIRED = true;
  } else if (wikitDataChar.value()[0] == 0x3D){
    WIKIT_KEY_REQUIRED = false;
  }
}

void wikitLedSettingCharcteristicWritten(BLECentral& central, BLECharacteristic& characteristic) {
  Serial.print("\nled setting characteristics event, written: ");
    if (wikitLedSettingChar.value()[0] == 1){
    WIKIT_LED_ON_REQUIRED = true;
    WIKIT_LED_OFF_REQUIRED = false;
    //MQBAO Serial.print("mqbao led data on\n");
  }
  else if (wikitLedSettingChar.value()[0] == 2){
    WIKIT_LED_ON_REQUIRED = false;
    WIKIT_LED_OFF_REQUIRED = true;
    //MQBAO Serial.print("mqbao led data off\n");
  }
 else if (wikitLedSettingChar.value()[0] == 0x17){
    WIKIT_RGB_G_REQUIRED = true;
    WIKIT_RGB_R_REQUIRED = false;
    WIKIT_RGB_B_REQUIRED = false;
  }else if (wikitLedSettingChar.value()[0] == 6){
    WIKIT_RGB_G_REQUIRED = false;
    WIKIT_RGB_R_REQUIRED = true;
    WIKIT_RGB_B_REQUIRED = false;
  }else if (wikitLedSettingChar.value()[0] == 8){
    WIKIT_RGB_G_REQUIRED = false;
    WIKIT_RGB_R_REQUIRED = false;
    WIKIT_RGB_B_REQUIRED = true;
  }else if (wikitLedSettingChar.value()[0] == 0x3b){
    WIKIT_RGB_G_REQUIRED = false;
    WIKIT_RGB_R_REQUIRED = false;
    WIKIT_RGB_B_REQUIRED = false;
  }
}

void wikitMultilinkPeripheralCharcteristicWritten(BLECentral& central, BLECharacteristic& characteristic) {
  //Serial.println(wikitMultilinkPeripheralChar.value()[0]);
  digitalWrite(ledPin,HIGH);
  /*
  Serial.print("wikitMultilinkPeripheralCharcteristicWritten event, written: ");
  Serial.println(wikitMultilinkPeripheralChar.value()[0]);
  Serial.println(wikitMultilinkPeripheralChar.value()[1]);
  Serial.println(wikitMultilinkPeripheralChar.value()[2]);
  Serial.println(wikitMultilinkPeripheralChar.value()[3]);
  Serial.println(wikitMultilinkPeripheralChar.value()[4]);
  Serial.println(wikitMultilinkPeripheralChar.value()[5]);
  Serial.println(wikitMultilinkPeripheralChar.value()[6]);
  Serial.println(wikitMultilinkPeripheralChar.value()[7]);
 */
  if (wikitMultilinkPeripheralChar.value()[1] == MP6050_DATA_TYPE){//MP6050
    NO_FUSIONWIRE_MP6050 = false;
    ax = (wikitMultilinkPeripheralChar.value()[3] << 8) | wikitMultilinkPeripheralChar.value()[4];
    ay = (wikitMultilinkPeripheralChar.value()[5] << 8) | wikitMultilinkPeripheralChar.value()[6];
    az = (wikitMultilinkPeripheralChar.value()[7] << 8) | wikitMultilinkPeripheralChar.value()[8];
    gx = (wikitMultilinkPeripheralChar.value()[9] << 8) | wikitMultilinkPeripheralChar.value()[10];
    gy = (wikitMultilinkPeripheralChar.value()[11] << 8) | wikitMultilinkPeripheralChar.value()[12];
    gz = (wikitMultilinkPeripheralChar.value()[13] << 8) | wikitMultilinkPeripheralChar.value()[14]; 
    //Serial.println(ax);
    /*
    Serial.println(ax);
    Serial.println(ay);
    Serial.println(az);
    Serial.println(gx);
    Serial.println(gy);
    Serial.println(gz);
    */
    WIKIT_MP6050_UPDATED = true;
  }else if (wikitMultilinkPeripheralChar.value()[1] == LIGHT_DATA_TYPE) {//LIGHT
    wikit_light = wikitMultilinkPeripheralChar.value()[3];
    WIKIT_LIGHT_UPDATED = true;
  }else if (wikitMultilinkPeripheralChar.value()[1] == TEMP_DATA_TYPE) {//TEMP
    wikit_temp = wikitMultilinkPeripheralChar.value()[3];
    WIKIT_TEMP_UPDATED = true;
  }else if (wikitMultilinkPeripheralChar.value()[1] == KEY_DATA_TYPE) {//KEY
    wikit_key = wikitMultilinkPeripheralChar.value()[3];
    WIKIT_KEY_UPDATED = true;
  }
  
}
