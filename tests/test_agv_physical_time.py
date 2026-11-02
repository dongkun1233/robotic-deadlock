import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rware.agv_physical_time import AGVPhysicalTimeConfig
class PhysicalTimeTests(unittest.TestCase):
 def test_translation_and_rotation_reference_values(self):
  c=AGVPhysicalTimeConfig();self.assertAlmostEqual(c.translation_time_s(1.5),1.875);self.assertAlmostEqual(c.translation_time_s(6),7.5);self.assertAlmostEqual(c.rotation_time_s(math.pi/2),2.0707963267948966);self.assertAlmostEqual(c.rotation_time_s(math.pi),3.641592653589793)
 def test_small_angle_and_profile_boundary(self):
  c=AGVPhysicalTimeConfig();self.assertAlmostEqual(c.rotation_time_s(.25),2*math.sqrt(.25/2));self.assertAlmostEqual(c.rotation_time_s(.5),1);self.assertAlmostEqual(c.rotation_time_s(-math.pi/2),c.rotation_time_s(math.pi/2));self.assertEqual(c.rotation_time_s(0),0)
 def test_visualization_period_does_not_quantize_motion(self):
  a=AGVPhysicalTimeConfig(visualization_dt_s=.1);b=AGVPhysicalTimeConfig(visualization_dt_s=.01);self.assertEqual(a.translation_time_s(1.5),b.translation_time_s(1.5));self.assertEqual(a.rotation_time_s(math.pi/2),b.rotation_time_s(math.pi/2))
 def test_invalid_physics_rejected(self):
  for values in [dict(speed_mps=1.2),dict(angular_acceleration_rad_s2=0),dict(safety_margin_m=-1),dict(max_speed_mps=float('nan'))]:
   with self.assertRaises(ValueError):AGVPhysicalTimeConfig(**values)
 def test_body_envelope(self):
  c=AGVPhysicalTimeConfig();self.assertAlmostEqual(c.body_radius_m,math.hypot(.825,.480)/2);self.assertLess(2*c.reserved_radius_m,c.single_lane_width_m);self.assertEqual(c.shared_zone_count,6)
if __name__=='__main__':unittest.main()
