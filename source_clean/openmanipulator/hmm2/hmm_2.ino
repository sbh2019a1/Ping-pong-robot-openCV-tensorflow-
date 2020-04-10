#include <open_manipulator_libs.h>

robotis_manipulator::RobotisManipulator manipulator;
robotis_manipulator::JointActuator *joint;
robotis_manipulator::ToolActuator *tool;
robotis_manipulator::Kinematics *kinematics_;
robotis_manipulator::Trajectory trajectory_;
bool moving_state_;
bool step_moving_state_ = 0;

std::vector<JointValue> joint_V;
Eigen::Vector3d position_meter;
  
void setup() 
{
  Serial.begin(57600);  
  Serial.available();

  manipulator.addWorld("world", "joint1");
  manipulator.addJoint("joint1", "world", "joint2", math::vector3(0.012, 0.0, 0.017), math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), Z_AXIS, 11, M_PI, -M_PI);
  manipulator.addJoint("joint2", "joint1", "joint3", math::vector3(0.0, 0.0, 0.0595), math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), Y_AXIS, 12, M_PI_2, -2.05);
  manipulator.addJoint("joint3", "joint2", "joint4", math::vector3(0.024, 0.0, 0.128), math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), Y_AXIS, 13, 1.53, -M_PI_2);
  manipulator.addJoint("joint4", "joint3", "gripper", math::vector3(0.124, 0.0, 0.0), math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), Y_AXIS, 14, 2.0, -1.8);
  manipulator.addTool("gripper", "joint4", math::vector3(0.126, 0.0, 0.0), math::convertRPYToRotationMatrix(0.0, 0.0, 0.0), 15, 0.010, -0.010, -0.015);

//  kinematics_ = new kinematics::SolverCustomizedforOMChain();
  kinematics_ = new kinematics::SolverUsingCRAndSRPositionOnlyJacobian();
  manipulator.addKinematics(kinematics_);

//  trajectory_.addCustomTrajectory(CUSTOM_TRAJECTORY_LINE, new custom_trajectory::Line());

  joint = new dynamixel::JointDynamixelProfileControl(0.010);
  STRING dxl_comm_arg[2] = {"COM6", "1000000"};
  void *p_dxl_comm_arg = &dxl_comm_arg;

  std::vector<uint8_t> jointDxlId;
  jointDxlId.push_back(11);
  jointDxlId.push_back(12);
  jointDxlId.push_back(13);
  jointDxlId.push_back(14);
  manipulator.addJointActuator(JOINT_DYNAMIXEL, joint, jointDxlId, p_dxl_comm_arg);

  tool = new dynamixel::GripperDynamixel();
  
  uint8_t gripperDxlId = 15;
  manipulator.addToolActuator(TOOL_DYNAMIXEL, tool, gripperDxlId, p_dxl_comm_arg);

  STRING gripper_dxl_mode_arg = "current_based_position_mode";
  void *p_gripper_dxl_mode_arg = &gripper_dxl_mode_arg;
  manipulator.setToolActuatorMode(TOOL_DYNAMIXEL, p_gripper_dxl_mode_arg);

  STRING gripper_dxl_opt_arg[2];
  void *p_gripper_dxl_opt_arg = &gripper_dxl_opt_arg;
  gripper_dxl_opt_arg[0] = "Profile_Acceleration";
  gripper_dxl_opt_arg[1] = "20";
  manipulator.setToolActuatorMode(TOOL_DYNAMIXEL, p_gripper_dxl_opt_arg);

  gripper_dxl_opt_arg[0] = "Profile_Velocity";
  gripper_dxl_opt_arg[1] = "200";
  manipulator.setToolActuatorMode(TOOL_DYNAMIXEL, p_gripper_dxl_opt_arg);

  manipulator.enableAllActuator();
  position_meter = Eigen::Vector3d(0.1, 0.1, 0.1);
}

void loop() 
{ 
//  Serial.println("GOGOGO");
//  processOpenManipulator(millis()/1000.0);
//  manipulator.makeTaskTrajectoryFromPresentPose("gripper", position_meter, 1.0, manipulator.getAllActiveJointValue());
  readAngle(manipulator.getKinematicPose("gripper"));
}

void processOpenManipulator(double present_time)
{
  JointWaypoint goal_joint_value = manipulator.getJointGoalValueFromTrajectory(present_time);
  if(goal_joint_value.size() != 0) manipulator.sendAllJointActuatorValue(goal_joint_value);
  manipulator.solveForwardKinematics();
}

void readAngle(KinematicPose joint_states_vector)
{
  Serial.print("position");
  Serial.print(", ");
  Serial.print(joint_states_vector.position.x());
  Serial.print("\n");
}
