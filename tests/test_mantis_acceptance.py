"""Negative causal-proof fixtures; no models, renderer or scene truth needed."""
from copy import deepcopy
import unittest

from experiments.mantis_acceptance import depth_causality,CAUSAL_DEPTH_LIMITS


def observation(sequence,capture,track_id=1):
    issued=capture+.05
    return dict(sequence=sequence,capture_time_s=capture,completed_time_s=issued,
        discarded_after_episode=False,
        observation=dict(valid=True,reason='observed',track_id=track_id,
                         sequence=sequence,capture_time_s=capture),
        detections=dict(detections=[dict(track_id=track_id,class_id=0,label='person')]),
        candidate=dict(valid=True,sequence=sequence,capture_time_s=capture,
            issued_at_s=issued,valid_until_s=capture+.9,forward_speed=.2,
            estimate=dict(valid=True,track_id=track_id,capture_time_s=capture)))


def fixture():
    spec=dict(causal_depth_test=True,kind='stop',event_s=1.,duration_s=3.)
    observations=[observation(0,.5),observation(1,1.),observation(2,1.2),observation(3,1.4)]
    ticks=[]
    for i in range(20):
        time=1.45+i*.005
        meta=dict(requested_speed_clamped_m_s=.2,depth_age_s=.02,considered_speed_m_s=.2,
                  total_reaction_s=.17,state_age_s=0.,level_velocity_m_s=[.1,0.,0.])
        actual=dict(reason='blocked_stopping_distance',valid_clearance=True,
                    forward_speed=0.,stopping_distance_m=.24,metadata=meta)
        pair=deepcopy(actual);pair.update(reason='clear',forward_speed=.2)
        ticks.append(dict(index=290+i,time_s=time,applied_command_sequence=3,
            request=dict(forward_speed=.2,sequence=3),guardian=actual,
            counterfactual_guardian=pair,actual_depth_capture_time_s=time-.02,
            counterfactual_depth_capture_time_s=time-.02))
    return spec,ticks,observations


class CausalDepthTests(unittest.TestCase):
    def score(self,transform=None):
        spec,ticks,observations=fixture()
        if transform:
            transform(spec,ticks,observations)
        return depth_causality(spec,ticks,observations)

    def test_sustained_fresh_paired_braking_passes(self):
        result=self.score()
        self.assertTrue(all(result['gates'].values()),result)
        self.assertEqual(result['metrics']['causal_depth_qualifying_ticks'],20)
        self.assertAlmostEqual(result['metrics']['causal_depth_longest_sustained_s'],.1)
        self.assertEqual(result['metrics']['causal_depth_valid_postevent_person_captures'],3)
        self.assertEqual(result['metrics']['causal_depth_postevent_target_fraction'],1.)
        self.assertEqual(CAUSAL_DEPTH_LIMITS['minimum_consecutive_override_ticks'],10)

    def test_noncausal_cases_add_no_new_gates(self):
        spec,ticks,observations=fixture();spec.pop('causal_depth_test')
        self.assertEqual(depth_causality(spec,ticks,observations),dict(gates={},metrics={}))

    def test_last_pre_event_command_never_counts(self):
        def change(spec,ticks,rows):
            for tick in ticks:
                tick['applied_command_sequence']=0;tick['request']['sequence']=0
        result=self.score(change)
        self.assertFalse(result['gates']['causal_depth_sustained_fresh_paired_braking'])
        self.assertEqual(result['metrics']['causal_depth_qualifying_ticks'],0)

    def test_missing_unobserved_or_different_person_invalidates_authority(self):
        changes=[lambda row:row['observation'].update(valid=False),
                 lambda row:row['observation'].update(reason='predicted'),
                 lambda row:row['observation'].update(track_id=2),
                 lambda row:row['detections']['detections'][0].update(label='bird',class_id=14),
                 lambda row:row.update(discarded_after_episode=True)]
        for change in changes:
            with self.subTest(change=change):
                result=self.score(lambda spec,ticks,rows:change(rows[-1]))
                self.assertFalse(result['gates']['causal_depth_sustained_fresh_paired_braking'])
                self.assertFalse(result['gates']['causal_depth_postevent_target_fraction'])

    def test_unreleased_or_expired_or_inconsistent_candidate_rejected(self):
        changes=[dict(issued_at_s=1.8),dict(valid_until_s=1.44),dict(valid=False),
                 dict(sequence=2),dict(capture_time_s=1.39),dict(forward_speed=.1)]
        for change in changes:
            with self.subTest(change=change):
                result=self.score(lambda spec,ticks,rows:rows[-1]['candidate'].update(change))
                self.assertFalse(result['gates']['causal_depth_sustained_fresh_paired_braking'])

    def test_fresh_observation_cannot_be_aliased_to_other_sequence(self):
        def change(spec,ticks,rows):
            for tick in ticks:tick['request']['sequence']=2
        result=self.score(change)
        self.assertFalse(result['gates']['causal_depth_sustained_fresh_paired_braking'])

    def test_visual_loss_stop_is_not_depth_causality(self):
        for kind in ('zero_request','unknown_depth','counterfactual_blocks'):
            def change(spec,ticks,rows):
                for tick in ticks:
                    if kind=='zero_request':tick['request']['forward_speed']=0.
                    elif kind=='unknown_depth':tick['guardian']['reason']='missing_depth'
                    else:tick['counterfactual_guardian'].update(reason='blocked_stopping_distance',forward_speed=0.)
            with self.subTest(kind=kind):
                result=self.score(change)
                self.assertFalse(result['gates']['causal_depth_sustained_fresh_paired_braking'])

    def test_depth_pair_must_be_present_synchronized_fresh_and_nonfuture(self):
        for value in (None,float('nan'),1.2,1.7,'bad'):
            with self.subTest(value=value):
                def change(spec,ticks,rows):
                    for tick in ticks:tick['counterfactual_depth_capture_time_s']=value
                result=self.score(change)
                self.assertFalse(result['gates']['causal_depth_sustained_fresh_paired_braking'])
        def old_pair(spec,ticks,rows):
            for tick in ticks:
                tick['actual_depth_capture_time_s']=tick['counterfactual_depth_capture_time_s']=tick['time_s']-.11
        self.assertFalse(self.score(old_pair)['gates']['causal_depth_sustained_fresh_paired_braking'])

    def test_different_counterfactual_state_or_request_is_rejected(self):
        for field,value in [('requested_speed_clamped_m_s',.3),('considered_speed_m_s',.3),
                             ('total_reaction_s',.2),('state_age_s',.05),
                             ('depth_age_s',.04),('level_velocity_m_s',[.2,0,0])]:
            with self.subTest(field=field):
                def change(spec,ticks,rows):
                    for tick in ticks:tick['counterfactual_guardian']['metadata'][field]=value
                self.assertFalse(self.score(change)['gates']['causal_depth_sustained_fresh_paired_braking'])

    def test_matching_pair_must_allow_the_exact_positive_request(self):
        for values in (dict(forward_speed=.1),dict(stopping_distance_m=.3),dict(valid_clearance=False)):
            def change(spec,ticks,rows):
                for tick in ticks:tick['counterfactual_guardian'].update(values)
            self.assertFalse(self.score(change)['gates']['causal_depth_sustained_fresh_paired_braking'])

    def test_duration_requires_consecutive_ten_physics_intervals(self):
        spec,ticks,rows=fixture()
        self.assertFalse(depth_causality(spec,ticks[:9],rows)['gates']['causal_depth_sustained_fresh_paired_braking'])
        self.assertTrue(depth_causality(spec,ticks[:10],rows)['gates']['causal_depth_sustained_fresh_paired_braking'])
        # Ten separated qualifying intervals are not .05 seconds sustained.
        for i,tick in enumerate(ticks):
            if i%2:tick['request']['forward_speed']=0.
        result=depth_causality(spec,ticks,rows)
        self.assertEqual(result['metrics']['causal_depth_qualifying_ticks'],10)
        self.assertFalse(result['gates']['causal_depth_sustained_fresh_paired_braking'])

    def test_three_distinct_postevent_captures_and_eighty_percent_required(self):
        spec,ticks,rows=fixture()
        result=depth_causality(spec,ticks,[rows[0],rows[2],rows[3]])
        self.assertFalse(result['gates']['causal_depth_postevent_person_captures'])
        rows.append(observation(4,1.7));rows[-1]['observation']['valid']=False
        result=depth_causality(spec,ticks,rows)
        self.assertEqual(result['metrics']['causal_depth_postevent_target_fraction'],.75)
        self.assertFalse(result['gates']['causal_depth_postevent_target_fraction'])

    def test_duplicate_observations_and_noncontinuous_ticks_fail_trace_gate(self):
        spec,ticks,rows=fixture();rows.append(deepcopy(rows[-1]))
        self.assertFalse(depth_causality(spec,ticks,rows)['gates']['causal_depth_trace_well_formed'])
        spec,ticks,rows=fixture();ticks[5]['time_s']+=.002
        self.assertFalse(depth_causality(spec,ticks,rows)['gates']['causal_depth_trace_well_formed'])

    def test_invalid_spec_or_unbound_initial_identity_fails_closed(self):
        spec,ticks,rows=fixture();spec['event_s']=float('nan')
        self.assertFalse(all(depth_causality(spec,ticks,rows)['gates'].values()))
        spec,ticks,rows=fixture()
        self.assertFalse(depth_causality(spec,ticks,rows[1:])['gates']['causal_depth_initial_person_identity'])

    def test_scoring_does_not_mutate_recorded_evidence(self):
        args=fixture();before=deepcopy(args);depth_causality(*args)
        self.assertEqual(args,before)


if __name__=='__main__':
    unittest.main()
