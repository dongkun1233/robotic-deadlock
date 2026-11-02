"""Shared physical assumptions and exact action durations; not a traffic simulator.

Linear acceleration is omitted. Rotation starts/ends at rest, with symmetric
angular acceleration/deceleration. Safety margin is a research assumption.
"""
from dataclasses import asdict,dataclass
import math

@dataclass(frozen=True)
class AGVPhysicalTimeConfig:
    model:str='SHARP_XF_Type_A'
    length_m:float=.825
    width_m:float=.480
    height_m:float=.190
    max_speed_mps:float=1.
    speed_mps:float=.8
    max_angular_speed_rad_s:float=1.
    angular_acceleration_rad_s2:float=2.
    turn_extra_delay_s:float=0.
    cell_size_m:float=1.5
    single_lane_width_m:float=1.5
    two_lane_width_m:float=3.
    safety_margin_m:float=.10
    visualization_dt_s:float=.1
    layout_rows:int=2
    layout_cols:int=3
    shared_zone_capacity:int=2

    def __post_init__(self):
        for name in ('length_m','width_m','height_m','max_speed_mps','speed_mps','max_angular_speed_rad_s','angular_acceleration_rad_s2','cell_size_m','single_lane_width_m','two_lane_width_m','visualization_dt_s'):
            v=getattr(self,name)
            if not math.isfinite(v) or v<=0:raise ValueError(f'{name} must be finite and positive')
        for name in ('turn_extra_delay_s','safety_margin_m'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:raise ValueError(f'{name} must be finite and nonnegative')
        if self.speed_mps>self.max_speed_mps:raise ValueError('Operating speed exceeds model maximum')
        if self.layout_rows<1 or self.layout_cols<1 or self.shared_zone_capacity<1:raise ValueError('Invalid layout/capacity')

    @property
    def body_radius_m(self):return math.hypot(self.length_m,self.width_m)/2
    @property
    def reserved_radius_m(self):return self.body_radius_m+self.safety_margin_m
    @property
    def shared_zone_count(self):return self.layout_rows*self.layout_cols

    def translation_time_s(self,distance_m):
        if not math.isfinite(distance_m) or distance_m<0:raise ValueError('Invalid distance')
        return distance_m/self.speed_mps

    def rotation_time_s(self,angle_rad):
        if not math.isfinite(angle_rad):raise ValueError('Invalid angle')
        angle=abs(angle_rad)
        if angle==0:return 0.
        w=self.max_angular_speed_rad_s;a=self.angular_acceleration_rad_s2
        duration=2*math.sqrt(angle/a) if angle<=w*w/a else angle/w+w/a
        return duration+self.turn_extra_delay_s

    def as_research_config(self):
        return {**asdict(self),'shared_zone_count':self.shared_zone_count,'turn_mode':'spin_turn','rotation_center_assumption':'geometric_body_center','linear_acceleration_mode':'omitted','angular_profile':'rest_to_rest_symmetric_accel_decel','body_only':True,'traffic_side':'right','simulation_time_mode':'exact_event_timestamps_planned','controller_period_s':None,'primary_objective':'makespan_seconds','secondary_metrics':['mean_wait_seconds','p95_wait_seconds','max_wait_seconds','spawn_delay_seconds'],'enable_path_conflict_check':True,'enable_cell_reservation':True,'safety_margin_source':'proposed research assumption, not manufacturer safety certification','implementation_status':'config_and_duration_primitives_only; event traffic engine not implemented by this module'}
