"""
Functions to run Random Forest analyses (classification of training quartile)
Dec2020
@author: Ines
@editors: Ines
"""
import pandas as pd
import numpy as np
import datajoint as dj
from ibl_pipeline import subject, action, acquisition, behavior
from ibl_pipeline.analyses import behavior as behavioral_analyses

# Some constants
CUTOFF_DATE = '2020-03-23'  # Date after which sessions are excluded, previously 30th Nov

"""
QUERY DATA
"""


def query_subjects(as_dataframe=False, criterion='trained'):
    """
    Query all mice for analysis of behavioral data
    Parameters
    ----------
    as_dataframe:    boolean if true returns a pandas dataframe (default is False)
    criterion:       what criterion by the 30th of November - trained (a and b), biased, ephys
                     (includes ready4ephysrig, ready4delay and ready4recording).  If None,
                     all mice that completed a training session are returned, with date_trained
                     being the date of their first training session.
    """
    from ibl_pipeline import subject, acquisition, reference
    from ibl_pipeline.analyses import behavior as behavior_analysis

    # Query all subjects with project ibl_neuropixel_brainwide_01 and get the date at which
    # they reached a given training status
    all_subjects = (subject.Subject * subject.SubjectLab * reference.Lab * subject.SubjectProject &
                    'subject_project = "ibl_neuropixel_brainwide_01"')
    sessions = acquisition.Session * behavior_analysis.SessionTrainingStatus()
    fields = ('subject_nickname', 'sex', 'subject_birth_date', 'institution_short')

    if criterion is None:
        # Find first session of all mice; date_trained = date of first training session
        subj_query = all_subjects.aggr(
            sessions, * fields, date_trained='min(date(session_start_time))')
    else:  # date_trained = date of first session when criterion was reached
        if criterion == 'trained':
            restriction = 'training_status="trained_1a" OR training_status="trained_1b"'
        elif criterion == 'biased':
            restriction = 'task_protocol LIKE "%biased%"'
        elif criterion == 'ephys':
            restriction = 'training_status LIKE "ready%"'
        else:
            raise ValueError('criterion must be "trained", "biased" or "ephys"')
        subj_query = all_subjects.aggr(
            sessions & restriction, * fields, date_trained='min(date(session_start_time))')

    # Select subjects that reached criterion before cutoff date
    subjects = (subj_query & 'date_trained <= "%s"' % CUTOFF_DATE)
    if as_dataframe is True:
        subjects = subjects.fetch(format='frame')
        subjects = subjects.sort_values(by=['lab_name']).reset_index()

    return subjects


def training_time(subjects):

    """
    Days until 'trained' criterion

    Parameters
    subjects:           DJ table of subjects of interest

    """
    # -- Query data
    sessions = (acquisition.Session * subjects * behavioral_analyses.SessionTrainingStatus).proj(
        'subject_uuid', 'training_status', 'session_start_time')
    sessions = pd.DataFrame.from_dict(sessions.fetch(as_dict=True))

    df = pd.DataFrame(columns=['subject_uuid', 'training_time'], index=range(len(
        sessions['subject_uuid'].unique())))

    for i, mouse in enumerate(sessions['subject_uuid'].unique()):
        subj_sess = sessions.loc[sessions['subject_uuid'] == mouse].copy()
        df['subject_uuid'][i] = mouse
        if (np.sum(subj_sess['training_status'] == "trained_1a") + np.sum(
                subj_sess['training_status'] == "trained_1b")) > 0:
            df['training_time'][i] = np.sum(subj_sess['training_status'] ==
                                            "in_training") + np.sum(subj_sess['training_status'] ==
                                                                    "untrainable")
        else:
            df['training_time'][i] = np.nan

    return df


def training_trials(subjects):

    """
    Query all training trials for animals that got trained

    Parameters
    ----------
    subjects:         Subjects to query trials of

    """

    # --Find subjects that got trained
    trained = (acquisition.Session * subjects * behavioral_analyses.SessionTrainingStatus &
               'training_status in ("trained_1a", "trained_1b")').proj('training_status')
    trained = pd.DataFrame.from_dict(trained.fetch(as_dict=True))
    trained_subjects = trained['subject_uuid'].unique()

    # --Query training trials from animals that got trained

    training_sessions = (acquisition.Session * subject.Subject *
                         behavioral_analyses.SessionTrainingStatus &
                         [{'subject_uuid': eid} for eid in trained_subjects] &
                         'training_status in ("in_training", "untrainable")').proj(
        'session_uuid', 'subject_nickname', 'task_protocol', 'training_status',
        'session_start_time', 'session_lab', session_date='DATE(session_start_time)')
    trials = (training_sessions * behavior.TrialSet.Trial *
              behavioral_analyses.BehavioralSummaryByDate & 'training_day <= 5')

    # -- Warning, this step takes up a lot of memory and some time
    trials_df = pd.DataFrame.from_dict(trials.fetch(as_dict=True))

    return trials_df


def filt_ses(trials):

    # -- Eliminates trials from sessions with less than 90% of RT info

    mice = trials['subject_uuid'].unique()

    for i, m in enumerate(mice):
        mouse_trials = trials.loc[trials['subject_uuid'] == m]
        ses = mouse_trials['session_uuid'].unique()

        for ii, s in enumerate(ses):
            ses_trials = mouse_trials.loc[mouse_trials['session_uuid'] == s]
            tot_trials = len(ses_trials)
            last_trial = np.max(ses_trials.trial_id)
            complete = np.sum(np.abs(ses_trials['RT']) > 0)
            frac = complete / tot_trials
            frac_2 = tot_trials / last_trial

            if frac_2 < 0.7:
                print(mouse_trials.subject_nickname.unique(), frac_2, 'total trials')
                trials.loc[trials['session_uuid'] == s, 'RT'] = np.nan
            elif frac < 0.7:
                print(mouse_trials.subject_nickname.unique(), frac, 'RTs')
                trials.loc[trials['session_uuid'] == s, 'RT'] = np.nan

    return trials


def design_matrix(trials, subjects, session):

    """
    Build design matrix

    Parameters
    session:            Last session to be included

    """

    # -- Start design matrix with training time
    matrix = training_time(subjects)
    matrix = matrix.dropna()  # Keep only mice that got trained

    # Compute quantiles here, before loosing mice due to lacking data!
    matrix['training_time'] = matrix['training_time'].astype(int)
    crit = matrix['training_time']
    quantiles = crit.quantile([.25, .5, .75])
    matrix['quantile'] = matrix['training_time']
    matrix.loc[matrix['training_time'] <= quantiles[0.25], 'quantile'] = 1
    matrix.loc[(matrix['training_time'] > quantiles[0.25]) & (matrix['training_time'] <=
                                                              quantiles[0.5]), 'quantile'] = 2
    matrix.loc[(matrix['training_time'] > quantiles[0.5]) & (matrix['training_time'] <=
                                                             quantiles[0.75]), 'quantile'] = 3
    matrix.loc[matrix['training_time'] > quantiles[0.75], 'quantile'] = 4

    # -- Merge performance metrics
    perf = performance_metrics(trials, session)
    matrix = matrix.merge(perf, on=['subject_uuid'], how='outer')

    # -- Merge wheel data
    wheel = wheel_metrics(trials, session)
    matrix = matrix.merge(wheel, on=['subject_uuid'], how='outer')

    # -- Merge mouse id metrics
    id = subject_id(subjects)
    matrix = matrix.merge(id, on=['subject_uuid'], how='outer')
    matrix.loc[matrix.sex == 'F', 'sex'] = 1
    matrix.loc[matrix.sex == 'M', 'sex'] = 0

    # -- Merge water restriction metrics
    wr = water_and_weight(trials, subjects)
    matrix = matrix.merge(wr, on=['subject_uuid'], how='outer')

    # -- Merge rig metrics
    rig = ambient_metrics(trials)
    rig = rig.reset_index(level=[0])
    matrix = matrix.merge(rig, on=['subject_uuid'], how='outer')

    # -- Merge welfare metrics
    wf = welfare(trials)
    wf = wf.drop(['session_lab'], axis=1)
    matrix = matrix.merge(wf, on=['subject_uuid'], how='outer')

    return matrix


"""
DESIGN MATRIX COMPONENTS
"""


def subject_id(subjects):

    """
    Age at start of training and sex

    """
    # Query data

    # Initialize output df

    df = pd.DataFrame(columns=['subject_uuid', 'sex', 'age_start'], index=range(len(subjects)))

    sessions = subjects * acquisition.Session
    sessions = pd.DataFrame.from_dict(sessions.fetch(as_dict=True))

    for i, mouse in enumerate(sessions['subject_uuid'].unique()):
        subj_sess = sessions[sessions['subject_uuid'] == mouse].copy()
        subj_sess.reset_index()
        df['subject_uuid'][i] = mouse
        if np.invert(pd.isnull(list(subj_sess.subject_birth_date)[0])):
            age = min(pd.to_datetime(subj_sess['session_start_time']).dt.date) - list(
                subj_sess.subject_birth_date)[0]
            df['age_start'][i] = age.days
        df['sex'][i] = list(subj_sess.sex)[0]

    return df


def water_and_weight(trials, subjects):

    """
    Water and weight logs
    """
    # --Get water and weight information from DATAJOINT
    mice = trials.subject_uuid.unique()

    water = (subjects * action.WaterAdministration).proj('watertype_name', 'water_administered',
                                                         'adlib', 'subject_uuid',
                                                         session_date='DATE(administration_time)')
    water = pd.DataFrame.from_dict(water.fetch(as_dict=True))
    weight = (subjects * action.Weighing).proj('weight', 'subject_uuid',
                                               session_date='DATE(weighing_time)')
    weight = pd.DataFrame.from_dict(weight.fetch(as_dict=True))
    water_restriction = (subjects * action.WaterRestriction).proj(
        'reference_weight', 'subject_uuid', session_date='DATE(restriction_start_time)')
    water_restriction = pd.DataFrame.from_dict(water_restriction.fetch(as_dict=True))

    # -- Merge water administrations and weight logs
    water_weight = weight.merge(water, on=['subject_uuid', 'session_date', 'subject_project'],
                                how='outer')

    # -- Merge water and weight with training days info
    training_days = trials[['subject_uuid', 'session_date', 'training_day']]
    water_weight_training = pd.merge(left=water_weight, right=training_days,
                                     how='left', left_on=['subject_uuid', 'session_date'],
                                     right_on=['subject_uuid', 'session_date'])

    # -- Start Matrix
    df = pd.DataFrame(columns=['subject_uuid', 'weekend_water', 'weight_start',
                               'weight_loss', 'water_weight'], index=range(len(mice)))

    df['subject_uuid'] = trials.subject_uuid.unique()

    for m, mouse in enumerate(mice):

        # --Weekend water
        watertype = list(water.loc[water['subject_uuid'] == mouse, 'watertype_name'])
        if (('Water 2% Citric Acid' in watertype) or ('Citric Acid Water 2%' in watertype) or
           ('Citric Acid Water 3%' in watertype)):
            df.loc[df['subject_uuid'] == mouse, 'weekend_water'] = 1  # Citric acid
        else:
            df.loc[df['subject_uuid'] == mouse, 'weekend_water'] = 0  # Measured water

        # --Starting weight
        weight_start = water_weight_training.loc[(water_weight_training['subject_uuid'] == mouse) &
                                                 (water_weight_training['training_day'] == 1),
                                                 'weight']
        if len(weight_start) > 0:
            df.loc[df['subject_uuid'] == mouse, 'weight_start'] = list(weight_start)[0]
        else:
            print('Mouse', mouse, 'missing starting weight')

        # -- Weight loss
        try:
            starting_date = list(training_days.loc[(training_days['subject_uuid'] == mouse) &
                                                   (training_days['training_day'] == 1),
                                                   'session_date'])[0]
            restriction = water_restriction.loc[(water_restriction['subject_uuid'] == mouse) &
                                                (water_restriction['session_date'] <
                                                starting_date)]
            reference_weight = list(restriction['reference_weight'])[-1]

            weight_frac = water_weight_training.loc[(water_weight_training['subject_uuid'] == mouse
                                                     ) & (water_weight_training['training_day'] ==
                                                          1), 'weight'] / reference_weight
            weight_frac = weight_frac.drop_duplicates()
            df.loc[df['subject_uuid'] == mouse, 'weight_loss'] = weight_frac.mean()

        except IndexError:
            print('Mouse', mouse, 'missing data')

        # -- Water per body weight
        sessions = water_weight_training.loc[(water_weight_training['subject_uuid'] == mouse) &
                                             (water_weight_training['training_day'] <= 5)]
        consumed = pd.DataFrame(sessions.groupby(['training_day'])['water_administered'].sum())
        weight = pd.DataFrame(sessions.groupby(['training_day'])['weight'].mean())

        # df.loc[df['subject_uuid'] == mouse, 'water_weight'] = np.nanmean(
        #     water['water_administered'] / weight['weight'])
        df.loc[df['subject_uuid'] == mouse, 'water_weight'] = np.nanmean(
            consumed['water_administered'] / weight['weight'])

    return df


def ambient_metrics(trials):

    # Takes ambient sensor metrics for all trials of the sessions
    # of interest and averages across subjects
    sessions = trials.session_uuid.unique()
    ambientSensor = behavior.AmbientSensorData * behavior.Settings & [{'session_uuid': eid} for
                                                                      eid in sessions]
    ambientSensor = acquisition.Session.aggr(ambientSensor, 'session_start_time',
                                             temperature_c="avg(temperature_c)",
                                             air_pressure_mb='avg(air_pressure_mb)',
                                             relative_humidity='avg(relative_humidity)')
    rig_data = pd.DataFrame.from_dict(ambientSensor.fetch(as_dict=True))
    avg_metrics = rig_data.groupby(['subject_uuid']).agg({'temperature_c': 'median',
                                                          'air_pressure_mb': 'median',
                                                          'relative_humidity': 'median'})

    return avg_metrics


def performance_metrics(trials, session):

    """
    Build design matrix with performance metrics

    Parameters
    trials:             All training trials for the mice of interest

    """

    mice = trials['subject_uuid'].unique()
    d_matrix = pd.DataFrame(columns=['subject_uuid', 'perf_init', 'RT_init', 'trials_init',
                                     'delta_variance', 'trials_sum', 'perf_delta1', 'RT_delta1',
                                     'trials_delta1'], index=range(len(mice)))

    # --Pre-processing
    trials['contrast'] = trials['trial_stim_contrast_left'] + trials['trial_stim_contrast_right']
    trials['RT'] = trials['trial_response_time'] - trials['trial_stim_on_time']

    for m, mouse in enumerate(mice):

        mouse_data = trials.loc[trials.subject_uuid == mouse]

        d_matrix['subject_uuid'][m] = mouse

        # Task performance on the first session
        d_matrix['perf_init'][m] = np.nanmean(mouse_data.loc[mouse_data['training_day'] == 1,
                                                             'performance_easy'])
        d_matrix['RT_init'][m] = np.nanmedian(list(mouse_data.
                                              loc[(mouse_data['training_day'] == 1) &
                                                  (mouse_data['contrast'] >= 0.5), 'RT']))
        d_matrix['trials_init'][m] = np.max(mouse_data.loc[mouse_data['training_day'] == 1,
                                                           'trial_id'])

        # Change in task performance across the first sessions
        d_matrix['perf_delta1'][m] = np.nanmean(mouse_data.loc[mouse_data['training_day'] ==
                                                session, 'performance_easy']) - np.nanmean(
                                                    mouse_data.loc[mouse_data['training_day'] == 1,
                                                                   'performance_easy'])
        d_matrix['RT_delta1'][m] = np.nanmedian(list(mouse_data.loc[(mouse_data['training_day'] ==
                                                session) & (mouse_data['contrast'] >= 0.5),
                                                'RT'])) - np.nanmedian(list(mouse_data.loc[(
                                                    mouse_data['training_day'] == 1) &
                                                    (mouse_data['contrast'] >= 0.5), 'RT']))
        d_matrix['trials_delta1'][m] = np.max(mouse_data.loc[mouse_data['training_day'] ==
                                              session, 'trial_id']) - np.max(mouse_data.loc[
                                                  mouse_data['training_day'] == 1, 'trial_id'])

        d_matrix['trials_sum'][m] = np.max(mouse_data.loc[mouse_data['training_day'] == 1,
                                                          'trial_id']) + \
            np.max(mouse_data.loc[mouse_data['training_day'] == 2, 'trial_id']) + \
            np.max(mouse_data.loc[mouse_data['training_day'] == 3, 'trial_id']) + \
            np.max(mouse_data.loc[mouse_data['training_day'] == 4, 'trial_id']) + \
            np.max(mouse_data.loc[mouse_data['training_day'] == 5, 'trial_id'])

        mouse_perf = pd.DataFrame(mouse_data.groupby(['subject_uuid', 'training_day'])
                                  ['performance_easy'].mean())
        mouse_perf = mouse_perf.reset_index(level=[0, 1])
        delta = np.sign(np.diff(mouse_perf['performance_easy']))
        d_matrix['delta_variance'][m] = np.sum(delta) / len(delta)

    return d_matrix


def wheel_metrics(trials, session):

    session_duration = pd.DataFrame(trials.groupby(
        ['subject_uuid', 'session_start_time'])['trial_start_time'].max())
    session_duration = session_duration.reset_index(level=[0, 1])
    session_duration = session_duration.rename(columns={'trial_start_time': "session_duration"})

    # Data with added column for session duration
    data2 = trials.merge(session_duration, on=['subject_uuid', 'session_start_time'])

    # --Wheel data
    dj_wheel = dj.create_virtual_module('wheel_moves', 'group_shared_wheel')

    movements_summary = pd.DataFrame.from_dict(dj_wheel.WheelMoveSet().fetch(as_dict=True))
    movements_summary = movements_summary.merge(data2, on=['subject_uuid', 'session_start_time'])
    movements_summary['disp_norm'] = np.abs(movements_summary['total_displacement'] /
                                            movements_summary['total_distance'])
    movements_summary['moves_time'] = (movements_summary['total_distance'] /
                                       movements_summary['session_duration'])

    # -- df
    # data should have same mice as data2, but movements_summary might have less
    mice = movements_summary['subject_uuid'].unique()
    df = pd.DataFrame(columns=['subject_uuid', 'disp_norm', 'moves_time'], index=range(len(mice)))
    for m, mouse in enumerate(mice):

        mouse_data = movements_summary.loc[movements_summary.subject_uuid == mouse]

        df['subject_uuid'][m] = mouse
        df['disp_norm'][m] = np.nanmean(mouse_data.loc[mouse_data['training_day'] <= session,
                                                       'disp_norm'])
        df['moves_time'][m] = np.nanmean(mouse_data.loc[mouse_data['training_day'] <= session,
                                                        'moves_time'])

    return df


def welfare(data):

    df = data[['subject_uuid', 'subject_nickname', 'session_lab']]
    df = df.drop_duplicates()

    food_map = {'cortexlab': 18, 'hoferlab': 16, 'mrsicflogellab': 16, 'wittenlab': 20,
                'mainenlab': 20, 'zadorlab': 20, 'churchlandlab': 20, 'danlab': 20,
                'angelakilab': 19, 'steinmetzlab': 20}
    df['food'] = df.session_lab.map(food_map)

    # Inverted cycle = 1; non-inverted = 0 'hoferlab':1, 'mrsicflogellab':1,
    light_map = {'cortexlab': 0, 'wittenlab': 1, 'mainenlab': 0, 'zadorlab': 0, 'churchlandlab': 1,
                 'danlab': 0, 'angelakilab': 1, 'steinmetzlab': 0}
    df['light'] = df.session_lab.map(light_map)

    # Mice in SWC were assigned to diferent light cycles
    light_map_extra = {'SWC_001': 1, 'SWC_002': 1, 'SWC_003': 1, 'SWC_004': 1, 'SWC_005': 1,
                       'SWC_006': 1, 'SWC_007': 1, 'SWC_008': 1, 'SWC_009': 1, 'SWC_010': 1,
                       'SWC_011': 1, 'SWC_012': 1, 'SWC_013': 1, 'SWC_014': 1, 'SWC_015': 1,
                       'SWC_016': 1, 'SWC_017': 1, 'SWC_018': 1, 'SWC_019': 1, 'SWC_020': 1,
                       'SWC_021': 1, 'SWC_022': 1, 'SWC_023': 1, 'SWC_024': 1, 'SWC_025': 1,
                       'SWC_026': 1, 'SWC_027': 1, 'SWC_028': 1, 'SWC_029': 1, 'SWC_030': 1,
                       'SWC_031': 1, 'SWC_032': 1, 'SWC_033': 1, 'SWC_034': 1, 'SWC_035': 1,
                       'SWC_036': 1, 'SWC_044': 1, 'SWC_045': 1, 'SWC_046': 1, 'SWC_050': 1,
                       'SWC_051': 1, 'SWC_052': 1, 'SWC_056': 1, 'SWC_057': 1, 'SWC_058': 1,
                       'SWC_037': 0, 'SWC_038': 0, 'SWC_039': 0, 'SWC_040': 0, 'SWC_041': 0,
                       'SWC_042': 0, 'SWC_043': 0, 'SWC_047': 0, 'SWC_048': 0, 'SWC_049': 0,
                       'SWC_053': 0, 'SWC_054': 0, 'SWC_055': 0, 'SWC_059': 0, 'SWC_060': 0,
                       'SWC_061': 0}
    df.loc[df['session_lab'].isin(['hoferlab', 'mrsicflogellab']),
           'light'] = df.loc[df['session_lab'].isin(['hoferlab',
                                                     'mrsicflogellab']),
                             'subject_nickname'].map(light_map_extra)

    return df
