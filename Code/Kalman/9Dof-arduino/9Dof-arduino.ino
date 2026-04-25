#include <Arduino_BHY2.h>
#include <ArduinoBLE.h>
#include <Nicla_System.h>

namespace {

constexpr unsigned long kSampleIntervalMs = 7;
constexpr float kSampleRateHz = 1000.0f / kSampleIntervalMs;
constexpr uint32_t kSensorLatencyMs = 1;

constexpr uint16_t kAccelRangeG = 4;
constexpr uint16_t kGyroRangeDps = 500;

// Keep the BLE contract identical to the Xiao sketch so the Python tooling does
// not need to change.
constexpr char kDeviceName[] = "DPOINT";
constexpr char kServiceUuid[] = "19B10010-E8F2-537E-4F6C-D104768A1214";
constexpr char kImuCharacteristicUuid[] = "19B10013-E8F2-537E-4F6C-D104768A1214";

// Nicla-specific note:
// The original Xiao sketch powered the force sensor from a GPIO and sampled it
// with the nRF SAADC in differential mode. That path does not exist on Nicla.
// This rewrite keeps the pressure field in the BLE packet and reads it through
// the standard ADC path instead. If your sensor is externally powered, leave
// kPressureSensorPowerPin as -1. Otherwise, set it to the GPIO you use to feed
// the sensor and keep kPressureSensorPin wired to the analog output.
constexpr int kPressureSensorPin = A0;
constexpr int kPressureSensorPowerPin = -1;

struct IMUDataPacket {
  int16_t accel[3];
  int16_t gyro[3];
  int16_t mag[3];
  uint16_t pressure;
  uint32_t timestamp_ms;
};

#ifndef SENSOR_ID_ACC_PASS
#define SENSOR_ID_ACC_PASS SENSOR_ID_ACC
#endif

#ifndef SENSOR_ID_GYRO_PASS
#define SENSOR_ID_GYRO_PASS SENSOR_ID_GYRO
#endif

#ifndef SENSOR_ID_MAG_PASS
#define SENSOR_ID_MAG_PASS SENSOR_ID_MAG
#endif

BLEService stylusService(kServiceUuid);
BLECharacteristic imuCharacteristic(
  kImuCharacteristicUuid,
  BLERead | BLENotify,
  sizeof(IMUDataPacket)
);

// Pass-through sensors are the closest match to the old sketch's direct raw
// register reads from the LSM6DS3.
SensorXYZ accelerometer(SENSOR_ID_ACC_PASS);
SensorXYZ gyroscope(SENSOR_ID_GYRO_PASS);
SensorXYZ magnetometer(SENSOR_ID_MAG_PASS);

unsigned long nextSampleAtMs = 0;

void fatalBlink() {
  while (true) {
    nicla::leds.setColor(red);
    delay(150);
    nicla::leds.setColor(off);
    delay(150);
  }
}

void configurePressureSensor() {
  analogReadResolution(12);

  if (kPressureSensorPowerPin >= 0) {
    pinMode(kPressureSensorPowerPin, OUTPUT);
    digitalWrite(kPressureSensorPowerPin, HIGH);
    delay(5);
  }
}

uint16_t readPressure() {
  const int raw = analogRead(kPressureSensorPin);
  const long scaled = map(raw, 0, 4095, 0, 65535);
  return static_cast<uint16_t>(constrain(scaled, 0L, 65535L));
}

void configureImu() {
#ifdef NICLA_STANDALONE
  BHY2.begin(NICLA_STANDALONE);
#else
  BHY2.begin();
#endif

  accelerometer.begin(kSampleRateHz, kSensorLatencyMs);
  gyroscope.begin(kSampleRateHz, kSensorLatencyMs);
  magnetometer.begin(kSampleRateHz, kSensorLatencyMs);

  accelerometer.setRange(kAccelRangeG);
  gyroscope.setRange(kGyroRangeDps);
}

void configureBle() {
  if (!BLE.begin()) {
    Serial.println("BLE.begin() failed");
    fatalBlink();
  }

  BLE.setLocalName(kDeviceName);
  BLE.setDeviceName(kDeviceName);
  BLE.setAdvertisedService(stylusService);

  stylusService.addCharacteristic(imuCharacteristic);
  BLE.addService(stylusService);

  const IMUDataPacket initialPacket = {};
  imuCharacteristic.writeValue(
    reinterpret_cast<const uint8_t*>(&initialPacket),
    sizeof(initialPacket)
  );

  BLE.advertise();
}

void sendPacketIfReady() {
  const unsigned long now = millis();
  if (static_cast<long>(now - nextSampleAtMs) < 0) {
    return;
  }
  nextSampleAtMs = now + kSampleIntervalMs;

  BHY2.update();

  IMUDataPacket packet = {};
  packet.accel[0] = accelerometer.x();
  packet.accel[1] = accelerometer.y();
  packet.accel[2] = accelerometer.z();
  packet.gyro[0] = gyroscope.x();
  packet.gyro[1] = gyroscope.y();
  packet.gyro[2] = gyroscope.z();
  packet.mag[0] = magnetometer.x();
  packet.mag[1] = magnetometer.y();
  packet.mag[2] = magnetometer.z();
  packet.pressure = readPressure();
  packet.timestamp_ms = millis();

  imuCharacteristic.writeValue(
    reinterpret_cast<const uint8_t*>(&packet),
    sizeof(packet)
  );
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(500);

  nicla::begin();
  nicla::leds.begin();
  nicla::leds.setColor(blue);

  configurePressureSensor();
  configureImu();
  configureBle();

  nextSampleAtMs = millis();
  Serial.println("Nicla stylus streamer ready");
}

void loop() {
  BLE.poll();
  BHY2.update();

  if (BLE.connected()) {
    nicla::leds.setColor(green);
    sendPacketIfReady();
  } else {
    nicla::leds.setColor(blue);
  }
}
