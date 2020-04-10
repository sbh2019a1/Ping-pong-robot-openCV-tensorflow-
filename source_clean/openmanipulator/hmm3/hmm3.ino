#include <open_manipulator_libs.h>

OpenManipulator open_manipulator;
String *cmd;
String global_cmd[10];
Eigen::Vector3d goal_position;

double control_time = 0.010;
double present_time = 0.0;
double previous_time = 0.0;

void getData()
{
  String data = Serial.readStringUntil('\n');
  data.trim();
  split(data, ',', global_cmd);
  goal_position = Eigen::Vector3d((double)global_cmd[0].toFloat(), (double)global_cmd[1].toFloat(), (double)global_cmd[2].toFloat());
  Serial.print(goal_position.x());
  Serial.print(", ");
  Serial.print(goal_position.y());
  Serial.print(", ");
  Serial.println(goal_position.z());
//  open_manipulator.processOpenManipulator(millis()/1000.0);
}

void split(String data, char separator, String* temp)
{
  int cnt = 0;
  int get_index = 0;

  String copy = data;
  
  while(true)
  {
    get_index = copy.indexOf(separator);

    if(-1 != get_index)
    {
      temp[cnt] = copy.substring(0, get_index);
      copy = copy.substring(get_index + 1);
    }
    else
    {
      temp[cnt] = copy.substring(0, copy.length());
      break;
    }
    ++cnt;
  }
}

void setup() 
{
  Serial.begin(57600);  
  Serial.available();

  goal_position = Eigen::Vector3d(0.1, 0.0, 0.1);
  open_manipulator.initOpenManipulator(true);
  delay(100);
  open_manipulator.processOpenManipulator(millis()/1000.0);
}

void loop() 
{
  present_time = millis()/1000.0;
  if(present_time-previous_time >= control_time)
  {
    open_manipulator.makeJointTrajectory("gripper", goal_position, 0.1, open_manipulator.getAllActiveJointValue());
    open_manipulator.processOpenManipulator(millis()/1000.0);
    previous_time = millis()/1000.0;
  }
}

void serialEvent()
{
  while(Serial.available())
  {
    getData();
  }
}
