"""Read-only independent reconciliation of every V3 campaign result and trace.

Imports no simulator physics, collision or scorer routines. Scenario generation
is used only to compare saved inputs with the predeclared trial protocol.
"""
from pathlib import Path
from collections import Counter
import argparse, csv, gzip, hashlib, json, math, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataclasses import asdict


def suite_inputs(suite):
    if suite == 'hard':
        from stress.course import make_trial, APERTURES
        return make_trial, lambda seed: APERTURES
    if suite == 'holdout':
        from validation.holdout import make_trial, apertures_for_seed
        return make_trial, apertures_for_seed
    raise ValueError(f'Unsupported campaign suite: {suite}')


def source_hash(package,root=ROOT):
    digest = hashlib.sha256()
    for path in sorted((Path(root) / package).glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def norm(v): return math.sqrt(sum(x*x for x in v))
def dist(a,b): return norm([x-y for x,y in zip(a,b)])
def clamp(v, maximum):
    magnitude=norm(v)
    return [x*maximum/magnitude for x in v] if magnitude>maximum else list(v)


def point_distance(p, box):
    return math.sqrt(sum(max(box['low'][i]-p[i],0,p[i]-box['high'][i])**2 for i in range(3)))


def segment_hit(a,b,box,radius=.45):
    lo,hi=0.,1.
    for i in range(3):
        delta=b[i]-a[i]
        lower,upper=box['low'][i]-radius,box['high'][i]+radius
        if abs(delta)<1e-12:
            if not lower<=a[i]<=upper: return False
        else:
            one,two=sorted(((lower-a[i])/delta,(upper-a[i])/delta))
            lo=max(lo,one);hi=min(hi,two)
            if lo>hi: return False
    return True


def scene_at(scenario,t):
    result=[]
    for box in scenario['obstacles']:
        if not any(box['velocity']): result.append(box);continue
        phase=((scenario['seed']*2654435761)&0xffffffff)/4294967296*2*math.pi
        speed=norm(box['velocity'])
        frequency=speed/4.5
        delta=4.5*math.sin(frequency*t+phase)
        low=[box['low'][i]+box['velocity'][i]/speed*delta for i in range(3)]
        high=[box['high'][i]+box['velocity'][i]/speed*delta for i in range(3)]
        velocity=[v*math.cos(frequency*t+phase) for v in box['velocity']]
        result.append({**box,'low':low,'high':high,'velocity':velocity})
    return result


def sensor_draws(seed,tick):
    """Independent implementation of the declared stateless sensor schedule."""
    word=((seed & 0xffffffff)^((tick+1)*0x9e3779b9)) & 0xffffffff
    word^=word>>16;word=(word*0x85ebca6b)&0xffffffff;word^=word>>13
    values=[]
    for _ in range(7):
        word=(1664525*word+1013904223)&0xffffffff
        values.append(word/4294967296.)
    return values


def boxes_equal(actual,expected,tolerance=1.1e-6):
    if len(actual)!=len(expected): return False
    expected_by_id={box['id']:box for box in expected}
    if len({box['id'] for box in actual})!=len(actual):return False
    for box in actual:
        other=expected_by_id.get(box['id'])
        if other is None:return False
        if any(abs(a-b)>tolerance for key in ('low','high','velocity') for a,b in zip(box[key],other[key])):return False
    return True


def verify_observation(scenario,trace,n,static_ids,require,trial):
    sample=trace[n];t=sample['t'];observation=sample['observation']
    metadata=observation['visibility_and_fault_metadata']
    in_window=scenario['fault_start']<=t<scenario['fault_start']+scenario['fault_duration']
    capture_index=max(0,n-scenario['latency_steps']) if in_window and scenario['latency_steps'] else n
    capture=trace[capture_index];actual_capture=capture['t']
    draws=sensor_draws(scenario['seed'],capture_index)
    amplitude=scenario['sensor_noise']*math.sqrt(3.)
    expected_position=[capture['p'][i]+amplitude*(2*draws[i]-1) for i in range(3)]
    expected_velocity=[capture['v'][i]+.5*amplitude*(2*draws[i+3]-1) for i in range(3)]
    require(dist(expected_position,observation['position'])<2e-6,'independent noisy/delayed sensor position',trial)
    require(dist(expected_velocity,observation['velocity'])<2e-6,'independent noisy/delayed sensor velocity',trial)
    require(abs(capture['battery_wh']-observation['battery_wh'])<1.1e-6,'sensor capture-time battery',trial)
    require(abs(metadata['capture_truth_time']-actual_capture)<1e-6,'sensor actual capture timestamp',trial)
    require(metadata['in_fault_window']==in_window,'independent fault-window schedule',trial)
    clock_offset=12. if in_window and scenario['category'] in ('clock_reset','compound') else 0.
    require(abs(observation['capture_time']-(actual_capture-clock_offset))<1e-6,'independent sensor clock offset',trial)
    expected_valid=not (in_window and (sensor_draws(scenario['seed'],n)[6]<scenario['dropout_probability'] or scenario['category']=='compute_stall'))
    require(observation['valid']==expected_valid,'independent invalid-observation schedule',trial)
    require(abs(observation['receive_time']-t)<1e-6,'sensor receive timestamp',trial)
    require(set(observation['observed_static_ids'])==static_ids,'declared static prior',trial)
    boxes=scene_at(scenario,actual_capture)
    static=[b for b in boxes if b['id'] in static_ids]
    expected_dynamic=[]
    for box in boxes:
        if box['id'] in static_ids:continue
        center=[(a+b)/2 for a,b in zip(box['low'],box['high'])]
        if point_distance(capture['p'],box)<=6. and not any(segment_hit(capture['p'],center,b,0.) for b in static):
            expected_dynamic.append(box)
    require(boxes_equal(observation['observed_dynamic_obstacles'],expected_dynamic),'independent dynamic sensor range/occlusion/capture geometry',trial)


def physics_prediction(sample,scenario):
    dt=scenario['dt']; v=sample['v']; p=sample['p']
    command=clamp(sample['command']['velocity'],5.)
    gust=.75+.25*math.sin(sample['t']*.83+(scenario['seed']%101)*.071)
    acc=clamp([(command[i]-v[i])/.55+scenario['wind'][i]*.45*gust for i in range(3)],3.)
    next_v=clamp([v[i]+acc[i]*dt for i in range(3)],5.)
    next_p=[p[i]+(v[i]+next_v[i])*.5*dt for i in range(3)]
    energy=(180+7*norm(next_v)**2+16*max(next_v[2],0)+4*norm(acc))*dt/3600
    return next_p,next_v,energy


def audit(directory,source_root=None,reconciled=False):
    started=time.perf_counter(); errors=[]; warnings=[]; counts=Counter(); maxima=Counter()
    directory=Path(directory).resolve()
    audited_source=Path(source_root).resolve() if source_root else ROOT
    sys.path.insert(0,str(audited_source))
    data_directory=directory/'DERIVED_RECONCILIATION' if reconciled else directory
    def require(condition,label,trial=None):
        if not condition: errors.append({'check':label,'trial_id':trial})
    protocol=json.loads((directory/'protocol.json').read_text())
    suite=protocol.get('suite','hard')
    make_trial,apertures_for_seed=suite_inputs(suite)
    complete=json.loads((directory/'COMPLETE.json').read_text()) if (directory/'COMPLETE.json').exists() else None
    rows=[json.loads(line) for line in (data_directory/'all_trials.jsonl').read_text().splitlines()]
    csv_rows=list(csv.DictReader((data_directory/'all_trials.csv').open()))
    journal_path=directory/'completion_journal.jsonl'
    journal=[json.loads(line) for line in journal_path.read_text().splitlines()] if journal_path.exists() else []
    require(len(rows)==len(csv_rows)==protocol['trials'],'primary row counts')
    if reconciled:
        provenance=json.loads((data_directory/'PROVENANCE.json').read_text())
        warnings.append({'code':'original_attempt_invalid_derived_tables_only','original_invalidation':provenance['original_invalidation'],
                         'missing_original_journal_trial_ids':provenance['missing_original_journal_trial_ids'],
                         'note':'All retained traces are independently checked; this does not certify the invalid original campaign.'})
    else:
        require(complete is not None and complete['rows']==len(rows),'complete campaign marker and count')
    require(len({r['seed'] for r in rows})==len(rows),'unique seeds')
    require(len({r['trial_id'] for r in rows})==len(rows),'unique trial IDs')
    require(len(list((directory/'traces').glob('*.json.gz')))==len(rows),'one trace file per trial')
    require(reconciled or not (directory/'CAMPAIGN_INVALID.json').exists(),'no campaign invalidation marker')
    require(source_hash('flybrain_sim',audited_source)==protocol['controller_hash'],'controller source freeze')
    require(source_hash('stress',audited_source)==protocol['harness_hash'],'harness source freeze')
    if complete:
        require(complete['controller_hash']==protocol['controller_hash'] and complete['harness_hash']==protocol['harness_hash'],'completion marker source hashes')
    baseline_protocol=json.loads((ROOT/'reference/hard_course_2000/protocol.json').read_text())
    require(source_hash('reference/v2/flybrain_sim')==baseline_protocol['controller_hash'],'preserved V2 reference hash')
    baseline_scenarios={}
    if suite=='hard' and not protocol.get('development',False):
        scenario_reference=json.loads((ROOT/'reference/hard_course_2000/scenario_hashes.json').read_text())
        baseline_scenarios={item['seed']:item for item in scenario_reference['rows']}
        require(len(baseline_scenarios)==scenario_reference['trials']==baseline_protocol['trials'],'complete original V2 input reference')
        require(all(row['seed'] in baseline_scenarios for row in rows),'all final hard-suite seeds have original V2 inputs')
    if suite=='hard':
        require(protocol['course_hash']==baseline_protocol['course_hash'],'original hard-course geometry hash')
    if suite=='holdout':
        require(source_hash('validation',audited_source)==protocol['validation_hash'],'holdout generator source freeze')
        if complete:require(complete['validation_hash']==protocol['validation_hash'],'completion marker holdout source hash')
    for name in ('physics.py','runner.py','contracts.py','geometry.py'):
        require((audited_source/'flybrain_sim'/name).read_bytes()==(ROOT/'reference/v2/flybrain_sim'/name).read_bytes(),f'V2 {name} scoring/dynamics contract retained')
    rows_by_id={r['trial_id']:r for r in rows}
    require(len({r['trial_id'] for r in journal})==len(journal),'unique auxiliary journal IDs')
    require(all(({**r,'trace_file':'../'+r['trace_file']} if reconciled else r)==rows_by_id.get(r['trial_id']) for r in journal),'present auxiliary journal rows match primary records')
    missing_journal_ids=sorted(set(rows_by_id)-{r['trial_id'] for r in journal})
    immutable_receipts=protocol.get('recording_policy')=='immutable_per_trial_receipts'
    if immutable_receipts:
        require(complete is not None and complete.get('recording_policy')=='immutable_per_trial_receipts' and complete.get('receipts')==len(rows),'completion marker immutable receipt policy/count')
        require(len(list((directory/'receipts').glob('*.json')))==len(rows),'one immutable row receipt per trial')
        for row in rows:
            receipt=directory/'receipts'/f"trial_{row['trial_id']:04d}.json"
            require(receipt.exists(),'row receipt exists',row['trial_id'])
            if receipt.exists():require(json.loads(receipt.read_text())==row,'immutable receipt matches primary row',row['trial_id'])
    elif missing_journal_ids and not reconciled:
        warnings.append({'code':'auxiliary_completion_journal_incomplete','journal_rows':len(journal),
                         'primary_rows':len(rows),'missing_trial_ids':missing_journal_ids,
                         'journal_sha256':hashlib.sha256(journal_path.read_bytes()).hexdigest() if journal_path.exists() else None,
                         'note':'Original journal preserved. All primary records and compressed traces are independently checked. No causal explanation for missing auxiliary lines is established.'})
    require([{'trial_id':r['trial_id'],'seed':r['seed'],'profile':r['profile']} for r in rows]==protocol['trial_plan'],'declared trial order and seeds')
    distinct_trajectories=set(); profile_counts=Counter(); gate_counts=Counter(); fault_counts=Counter()
    for row,csv_row in zip(rows,csv_rows):
        trial=row['trial_id']; profile_counts[row['profile']]+=1
        expected_csv_fields=set(row)-{'diagnostics'}
        expected_csv_fields|={f'diagnostics.{key}' for key in row['diagnostics']}
        require(set(csv_row)==expected_csv_fields,'all CSV columns retained',trial)
        for key,value in row.items():
            if key=='diagnostics':continue
            expected='' if value is None else str(value)
            require(expected==csv_row.get(key),f'CSV field {key}',trial)
        for key,value in row['diagnostics'].items():
            column=f'diagnostics.{key}'
            if isinstance(value,(dict,list)):
                try: actual=json.loads(csv_row[column])
                except (KeyError,ValueError):actual=None
                require(actual==value,f'CSV field {column}',trial)
            else:
                require(('' if value is None else str(value))==csv_row.get(column),f'CSV field {column}',trial)
        compressed=(data_directory/row['trace_file']).read_bytes()
        require(hashlib.sha256(compressed).hexdigest()==row['trace_sha256'],'compressed trace SHA256',trial)
        require(len(compressed)==row['trace_bytes'],'trace byte size',trial)
        saved=json.loads(gzip.decompress(compressed)); trace=saved['trace']; scenario=saved['scenario']; diag=saved['diagnostics']
        static_ids={b['id'] for b in scenario['obstacles'] if not any(b['velocity'])}
        apertures=apertures_for_seed(row['seed'])
        require(saved['seed']==row['seed'] and saved['trial_id']==trial and saved['profile']==row['profile'],'trace identity',trial)
        require(saved['controller_hash']==protocol['controller_hash'],'trace controller hash',trial)
        require(scenario==json.loads(json.dumps(asdict(make_trial(row['seed'],row['profile'])))),'saved scenario vs declared generator',trial)
        if baseline_scenarios:
            original=baseline_scenarios.get(row['seed'],{})
            digest=hashlib.sha256(json.dumps(scenario,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            require(original.get('profile')==row['profile'] and original.get('scenario_sha256')==digest,'exact scenario/profile match to original V2 full trace',trial)
            counts['original_V2_inputs_matched']+=int(original.get('scenario_sha256')==digest)
        require(all(row[k]==v for k,v in saved['result'].items()),'trace result vs row',trial)
        require(diag==row['diagnostics'],'trace diagnostics vs row',trial)
        require(len(trace)==diag['trace_samples']==diag['integration_steps']+1,'all integration samples retained',trial)
        require(sum(s['sample_kind']=='control' for s in trace)==diag['command_updates'],'control update count',trial)
        # Scenario inputs retain full precision; logged positions are rounded
        # independently per coordinate. Non-decimal holdout homes must use the
        # declared half-micro-metre rounding bound, not exact-coordinate equality.
        require(trace[0]['t']==0 and all(abs(a-b)<=0.50001e-6 for a,b in zip(trace[0]['p'],scenario['home'])),'initial home state with recorded-coordinate rounding bound',trial)
        require(abs(trace[-1]['t']-row['simulated_seconds'])<1e-6,'terminal duration',trial)
        require(trace[-1]['terminal_outcome']==row['outcome'],'terminal sample outcome',trial)
        terminals=[e for e in saved['events'] if e['type']=='terminal']
        require(len(terminals)==1 and terminals[0]['outcome']==row['outcome'] and abs(terminals[0]['t']-row['simulated_seconds'])<1e-6,'single explicit terminal event',trial)
        require(bool(saved['contacts'])==row['collision'],'contacts accompany every collision',trial)
        require(row['mission_complete']==(row['outcome']=='mission_complete'),'success label',trial)
        require(abs(trace[-1]['energy_used_wh']-row['energy_wh'])<=.51e-6,'terminal energy',trial)
        require(abs(trace[-1]['battery_wh']-row['final_battery_wh'])<=.51e-6,'terminal battery',trial)
        require(abs(scenario['initial_battery_wh']-row['energy_wh']-row['final_battery_wh'])<2e-9 or row['final_battery_wh']==0,'energy accounting',trial)
        dt=scenario['dt']; path=0.; energy=0.; minimum=math.inf; max_speed=0.; max_acc=0.
        any_collision=False; possible_collision=False; violation=False
        dwell=[None]*len(scenario['waypoints']); inspected=set(); inspections={}
        mode_seconds=Counter(); injected_seconds=Counter(); rejected_seconds=0
        exposure_count=visible_count=0
        for n,sample in enumerate(trace):
            t=sample['t']; p=sample['p']; v=sample['v']; boxes=scene_at(scenario,t)
            require(boxes_equal(sample['dynamic_obstacles'],[b for b in boxes if b['id'] not in static_ids]),'independent moving-obstacle trace geometry',trial)
            require(abs(t-n*dt)<1e-6,'uniform simulation timestep',trial)
            this_clearance=min(point_distance(p,b) for b in boxes)-.45
            minimum=min(minimum,this_clearance); max_speed=max(max_speed,norm(v))
            require(abs(this_clearance-sample['clearance_m'])<2e-6,'sample clearance',trial)
            inside=all(.45<=p[i]<=scenario['bounds'][i]-.45 for i in range(3))
            violation|=not inside
            require(inside==sample['inside_bounds'],'sample body bounds',trial)
            if sample['sample_kind']=='control':
                for i,waypoint in enumerate(scenario['waypoints']):
                    if dist(p,waypoint)<=1. and norm(v)<=1.2:
                        if dwell[i] is None: dwell[i]=t
                        if t-dwell[i]>=.6-1e-8 and i not in inspected: inspected.add(i);inspections[str(i)]=t
                    else: dwell[i]=None
                require(set(sample['truth_inspected_waypoints'])==inspected,'independent elapsed-time inspection score',trial)
                observation=sample['observation'];meta=observation['visibility_and_fault_metadata']
                exposure_count+=bool(meta['in_fault_window']);visible_count+=bool(observation['observed_dynamic_obstacles'])
                verify_observation(scenario,trace,n,static_ids,require,trial)
                require(abs(t-observation['capture_time']-observation['reported_age_s'])<2e-6,'reported observation age',trial)
            if n==len(trace)-1: continue
            successor=trace[n+1]; next_boxes=scene_at(scenario,successor['t'])
            require(sample['sample_kind']=='control','integration has preceding control sample',trial)
            path+=dist(p,successor['p']); max_acc=max(max_acc,dist(v,successor['v'])/dt)
            predicted_p,predicted_v,increment=physics_prediction(sample,scenario);energy+=increment
            p_error=dist(predicted_p,successor['p']);v_error=dist(predicted_v,successor['v'])
            maxima['position_step_error_m']=max(maxima['position_step_error_m'],p_error)
            maxima['velocity_step_error_m_s']=max(maxima['velocity_step_error_m_s'],v_error)
            require(p_error<2e-6 and v_error<2e-6,'independent dynamics recurrence',trial)
            mode_seconds[sample['mode']]+=dt
            for fault in sample['observation']['visibility_and_fault_metadata']['active_faults']: injected_seconds[fault]+=dt
            if not sample['observation']['accepted']: rejected_seconds+=dt
            for first,second in zip(boxes,next_boxes):
                moving=first['low']!=second['low'] or first['high']!=second['high']; padding=.01 if moving else 0
                envelope={'low':[min(x,y)-padding for x,y in zip(first['low'],second['low'])],
                          'high':[max(x,y)+padding for x,y in zip(first['high'],second['high'])]}
                hit=segment_hit(p,successor['p'],envelope,.45-2e-6)
                any_collision|=hit
                possible_collision|=segment_hit(p,successor['p'],envelope,.45+2e-6)
                if hit: require(n==len(trace)-2 and row['collision'],'collision stops simulation',trial)
            for gate in apertures:
                axis=1 if gate.get('partition_axis')=='y' else 0
                transverse=1-axis
                center=gate.get('center',(gate['x'],gate['y'],gate['z']))
                plane=center[axis]
                center_y=center[transverse]
                if (p[axis]-plane)*(successor['p'][axis]-plane)<0:
                    ratio=(plane-p[axis])/(successor['p'][axis]-p[axis])
                    crossing=[p[i]+ratio*(successor['p'][i]-p[i]) for i in range(3)]
                    gate_counts[f"{gate['id']}_{'positive_axis' if successor['p'][axis]>p[axis] else 'negative_axis'}"]+=1
                    clear=(abs(crossing[transverse]-center_y)<=gate['width']/2-.45+2e-6 and abs(crossing[2]-center[2])<=gate['height']/2-.45+2e-6)
                    require(clear or row['collision'],'gate aperture body clearance',trial)
        require(not any_collision or row['collision'],'independent swept collision detection',trial)
        require(not row['collision'] or possible_collision,'reported collision independently possible',trial)
        require(violation==row['geofence_violation'],'independent geofence result',trial)
        require(len(inspected)==row['waypoints_completed'],'independent final inspection count',trial)
        require(set(inspections)==set(diag['truth_waypoint_completion_times_s']) and all(abs(t-diag['truth_waypoint_completion_times_s'][i])<1e-6 for i,t in inspections.items()),'independent dwell completion times',trial)
        home=dist(trace[-1]['p'],scenario['home'])<=1 and norm(trace[-1]['v'])<=.6
        require(row['returned_home']==(home and not row['collision'] and not violation),'independent actual home check',trial)
        require(not row['mission_complete'] or (len(inspected)==len(scenario['waypoints']) and home),'independent full success condition',trial)
        require(abs(path-row['distance_m'])<=len(trace)*math.sqrt(3)*1e-6,'integrated path length with trace rounding bound',trial)
        require(abs(energy-row['energy_wh'])<len(trace)*2e-8,'independently recomputed energy surrogate',trial)
        require(abs(minimum-row['minimum_clearance_m'])<2e-6,'independent minimum clearance',trial)
        require(abs(max_speed-row['max_speed_mps'])<2e-6 and abs(max_acc-row['max_acceleration_mps2'])<1e-5,'peak speed and acceleration',trial)
        require(dict(mode_seconds)==diag['mode_seconds'],'mode durations',trial)
        require(dict(injected_seconds)==diag['fault_seconds'],'injected fault durations',trial)
        require(rejected_seconds==diag['rejected_observation_seconds'],'rejected observation duration',trial)
        require(exposure_count==row['fault_window_observation_samples'] and visible_count==row['dynamic_observation_samples'],'fault and dynamic observation exposure',trial)
        require(bool(injected_seconds)==row['timed_fault_exposed'],'actual timed fault exposure flag',trial)
        fault_counts.update({fault:1 for fault in injected_seconds})
        counts.update({'trials':1,'trace_samples':len(trace),'integration_steps':len(trace)-1,'collision_trials':int(row['collision']),'verified_home_trials':int(row['returned_home']),'full_success_trials':int(row['mission_complete'])})
        distinct_trajectories.add(hashlib.sha256(json.dumps([(s['p'],s['v']) for s in trace],separators=(',',':')).encode()).hexdigest())
        if trial%200==0: print(json.dumps({'audited':trial,'errors':len(errors),'elapsed_s':round(time.perf_counter()-started,2)}),flush=True)
    # Distinct trajectories are descriptive, not a validity gate: an honest
    # deterministic controller can legitimately produce identical trajectories.
    summary=json.loads((data_directory/'summary.json').read_text())
    for key,target in [('runs',len(rows)),('successes',counts['full_success_trials']),('collisions',counts['collision_trials']),('returned_home',counts['verified_home_trials']),('trace_samples',counts['trace_samples'])]:require(summary[key]==target,f'summary {key}')
    require(summary['outcomes']==dict(Counter(r['outcome'] for r in rows)),'summary outcomes')
    by_profile=json.loads((data_directory/'by_profile.json').read_text())
    require({p['profile'] for p in by_profile}==set(profile_counts),'all profile summaries present')
    for profile in by_profile:
        selected=[row for row in rows if row['profile']==profile['profile']]
        for key,field in [('successes','mission_complete'),('collisions','collision'),('geofence','geofence_violation'),('returned_home','returned_home'),('timed_fault_exposed','timed_fault_exposed')]:
            require(profile[key]==sum(row[field] for row in selected),f'profile {profile["profile"]}: {key}')
        require(profile['runs']==len(selected),f'profile {profile["profile"]}: run count')
        require(profile['outcomes']==dict(Counter(row['outcome'] for row in selected)),f'profile {profile["profile"]}: outcomes')
    return {'audit_passed':not errors and not warnings,'primary_results_verified':not errors,
            'all_artifacts_complete':not errors and not warnings,
            'audit_status':('failed' if errors else 'retained_traces_verified_original_campaign_invalid' if reconciled
                            else 'verified_with_auxiliary_warning' if warnings else 'verified'),
            'errors':errors,'warnings':warnings,'counts':dict(counts),'profile_counts':dict(profile_counts),
            'distinct_state_trajectories':len(distinct_trajectories),'gate_crossing_counts':dict(gate_counts),
            'actual_fault_exposure_trial_counts':dict(fault_counts),'max_observed_rounding_reconstruction_errors':dict(maxima),
            'suite':suite,'source_root':str(audited_source),'reconciled_invalid_attempt':reconciled,
            'immutable_receipts_verified':len(rows) if immutable_receipts else 0,
            'controller_hash':source_hash('flybrain_sim',audited_source),'harness_hash':source_hash('stress',audited_source),
            'audit_wall_seconds':time.perf_counter()-started,
            'scope':'Every compressed trace SHA256 and row; independent trace physics, energy, collision, geofence, dwell, home, sensor noise/delay/validity/visibility and timing reconstruction. Physics/scoring contract is checked byte-for-byte against preserved V2 source. Trace rounding requires explicitly bounded numerical tolerances. No real flight or untested research-model claims.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory');parser.add_argument('output',nargs='?')
    parser.add_argument('--source-root');parser.add_argument('--reconciled',action='store_true')
    args=parser.parse_args()
    report=audit(args.directory,args.source_root,args.reconciled)
    if args.output: Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['primary_results_verified']: raise SystemExit(1)
