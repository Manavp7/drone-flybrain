"""Prevent software execution and planner counters becoming false flight passes."""
import unittest
from experiments.mantis_studio_checks import score


def receipt():
    return dict(status='completed',summary=dict(failure=None,statistics=dict(
        contacts=0,wrong_person_observations=0,evaluated_selected_observations=20,
        actual_yolo_calls=20,actual_flyvis_observations=20,detours_completed=1,
        selected_actor_reference='blue',
        minimum_obstacle_hull_clearance_m=.25,detour_completion_events=[dict(
            sim_s=5.,position=[1.7,-.4,1.1],selected_track=2,command_sequence=7,source_capture_time_s=4.8)],
        motion_gap_resets=0,motion_valid_speed_reductions=10,positive_forward_ticks=20)))


def following_rows():
    return [dict(frame=dict(capture_time_s=t,sequence=7+i),completed_time_s=t+.15,
                 navigation=dict(reason='following'),safety=dict(forward_speed=.1),
                 selection=dict(track_id=2,sequence=7+i,capture_time_s=t,held=False,reason='observed'),
                 guidance=dict(valid=True,track_id=2,capture_time_s=t),
                 evaluation=dict(wrong_person=False,actor_id='blue'),
                 position=[x,-.4,1.1]) for i,(t,x) in enumerate([(4.8,1.7),(5.1,1.8),(5.5,1.95),(6.,2.2)])]


class DetourScoreTests(unittest.TestCase):
    def check(self,r=None,rows=None):
        return score(r or receipt(),following_rows() if rows is None else rows,dict(scenario='detour'))

    def test_requires_observed_passage_and_following_not_counter(self):
        self.assertTrue(self.check()['passed'])
        r=receipt();r['summary']['statistics']['detour_completion_events']=[]
        self.assertFalse(self.check(r)['passed'])
        rows=following_rows()
        for row in rows:row['position'][0]=1.8
        self.assertFalse(self.check(rows=rows)['passed'])

    def test_contacts_margin_wrong_person_or_unobserved_authority_fail(self):
        for key,value in [('contacts',1),('wrong_person_observations',1),
                          ('minimum_obstacle_hull_clearance_m',None),
                          ('minimum_obstacle_hull_clearance_m',.19)]:
            r=receipt();r['summary']['statistics'][key]=value
            self.assertFalse(self.check(r)['passed'],key)
        r=receipt();r['summary']['statistics']['detour_completion_events'][0]['source_capture_time_s']=3.
        self.assertFalse(self.check(r)['passed'])

    def test_stale_candidate_does_not_prove_resumed_following(self):
        rows=following_rows()
        for row in rows:row['completed_time_s']=row['frame']['capture_time_s']+1.
        self.assertFalse(self.check(rows=rows)['passed'])

    def test_event_needs_a_recorded_matching_selected_observation(self):
        r=receipt();r['summary']['statistics']['detour_completion_events'][0]['selected_track']=None
        self.assertFalse(self.check(r)['passed'])
        self.assertFalse(self.check(rows=following_rows()[1:])['passed'])
        rows=following_rows();rows[0]['selection']['track_id']=3
        self.assertFalse(self.check(rows=rows)['passed'])

    def test_passage_before_completion_does_not_prove_the_detour(self):
        rows=following_rows();rows[0]['position'][0]=2.2
        for row in rows[1:]:row['position'][0]=2.
        self.assertFalse(self.check(rows=rows)['passed'])

    def test_held_selection_mismatched_guidance_and_duplicate_rows_fail(self):
        for field,value in [('track_id',99),('capture_time_s',0.)]:
            rows=following_rows();rows[0]['guidance'][field]=value
            self.assertFalse(self.check(rows=rows)['passed'])
        rows=following_rows()
        for row in rows:row['selection'].update(held=True,reason='selected_person_lost')
        self.assertFalse(self.check(rows=rows)['passed'])
        rows=following_rows()
        self.assertFalse(self.check(rows=[rows[0],rows[-1],rows[-1],rows[-1]])['passed'])

    def test_ended_or_partial_run_does_not_imply_success(self):
        for status in ['error','interrupted','budget-exhausted']:
            r=receipt();r['status']=status
            self.assertFalse(self.check(r)['passed'])


class MotionScoreTests(unittest.TestCase):
    def rows(self):
        return [dict(frame=dict(sequence=i,capture_time_s=i*.2),
            motion=dict(valid=i>=3,capture_time_s=i*.2,response_time_s=i*.2+.02,
            available_time_s=i*.2+.1,neural=dict(rms_decoder_magnitude=.03)),
            completed_time_s=i*.2+.1,inference_wall_s=.1) for i in range(20)]

    def test_warmed_fresh_sensor_and_speed_effect_are_separate(self):
        r=receipt();rows=self.rows();config=dict(scenario='stationary',motion_mode='brake')
        r['summary']['statistics']['motion_brake_pairs']=[dict(sim_s=1.15,
            motion_capture_time_s=1.,motion_available_time_s=1.1,
            command_capture_time_s=1.,depth_capture_time_s=1.1,
            depth_approved_unscaled_speed=.3,actual_forward_speed=.2,scale=2/3)]
        self.assertTrue(score(r,rows,config)['passed'])
        r['summary']['statistics']['motion_valid_speed_reductions']=0
        self.assertFalse(score(r,rows,config)['passed'])
        self.assertTrue(score(r,rows,dict(config,motion_mode='observe'))['passed'])

    def test_reduction_counter_without_a_same_tick_depth_pair_fails(self):
        self.assertFalse(score(receipt(),self.rows(),dict(scenario='stationary',motion_mode='brake'))['passed'])

    def test_warming_stale_or_future_neural_output_cannot_pass(self):
        for mode in ['warming','stale','future']:
            rows=self.rows()
            for row in rows:
                if mode=='warming':row['motion']['valid']=False
                elif mode=='stale':row['completed_time_s']+=1.
                else:row['motion']['available_time_s']+=1.
            self.assertFalse(score(receipt(),rows,dict(scenario='stationary',motion_mode='observe'))['passed'],mode)


if __name__=='__main__':
    unittest.main()
