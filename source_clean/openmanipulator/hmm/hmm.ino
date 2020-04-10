#include <open_manipulator_libs.h>

OpenManipulator open_manipulator;
//std::vector<double> goal_position;
String *cmd;
String global_cmd[10];
Eigen::Vector3d goal_position;

void readAngle(JointWaypoint joint_states_vector)
{
  Serial.print("angle");
  for (int i = 0; i < (int)joint_states_vector.size(); i++)
  {
    Serial.print(",");
    Serial.print(joint_states_vector.at(i).position * 180/PI, 3);
  }
  Serial.print("\n");
}

void readPosition(KinematicPose joint_states_vector)
{
  Serial.print("position");
  Serial.print(", ");
  Serial.print(joint_states_vector.position.x());
  Serial.print(", ");
  Serial.print(joint_states_vector.position.y());
  Serial.print(", ");
  Serial.print(joint_states_vector.position.z());
  Serial.print("\n");
}

void getData()
{
  String data = Serial.readStringUntil('\n');
  data.trim();
  split(data, ',', global_cmd);
//  goal_position[0] = (double)global_cmd[0].toFloat() * PI/180.0;
//  goal_position[1] = (double)global_cmd[1].toFloat() * PI/180.0;
//  goal_position[2] = (double)global_cmd[2].toFloat() * PI/180.0;
//  goal_position[3] = (double)global_cmd[3].toFloat() * PI/180.0;
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
  
  open_manipulator.initOpenManipulator(true);
  open_manipulator.processOpenManipulator(millis()/1000.0);
  goal_position = Eigen::Vector3d(0.1, 0, 0.25);
  
//  goal_position.push_back(double(0.0));
//  goal_position.push_back(double(-60.0  * PI/180.0));
//  goal_position.push_back(double(20.0 * PI/180.0));
//  goal_position.push_back(double(40.0 * PI/180.0));
//  open_manipulator.disableAllActuator();
}

void loop() 
{ 
//  readAngle(open_manipulator.getAllActiveJointValue());
//  open_manipulator.makeJointTrajectory(goal_position, 1.0); // FIX TIME PARAM

  readPosition(open_manipulator.getKinematicPose("gripper"));
//  open_manipulator.makeJointTrajectory("gripper", goal_position, 1.0, open_manipulator.getAllActiveJointValue());
//  open_manipulator.processOpenManipulator(millis()/1000.0);
}

//void serialEvent()
//{
//  while(Serial.available())
//  {
//    getData();
//  }
//}
