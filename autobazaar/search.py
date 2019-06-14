# -*- coding: utf-8 -*-

"""AutoBazaar Search Module.


This module contains the PipelineSearcher, which is the class that
contains the main logic of the Auto Machine Learning process.
"""

import itertools
import json
import logging
import os
import signal
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
from btb import HyperParameter
from btb.tuning import GP, GPEi, Uniform
from mit_d3m.loaders import get_loader
from mlblocks.mlpipeline import MLPipeline
from sklearn.model_selection import KFold, StratifiedKFold

from autobazaar.pipeline import ABPipeline
from autobazaar.utils import ensure_dir, make_dumpable, remove_dots, restore_dots

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')


TRIVIAL_PIPELINE_METHOD = {
    'classification': 'mode',
    'regression': 'median',
    'collaborativeFiltering': 'median',
    'graphMatching': 'mode',
}

TUNERS = {
    'gp': GP,
    'gpei': GPEi,
    'uniform': Uniform
}

PARAM_TYPES = {
    'str': 'string',
}


class StopSearch(Exception):
    pass


class PipelineSearcher(object):
    """PipelineSearcher class.

    This class is responsible for searching the best pipeline to solve a
    given dataset and problem.
    """

    def __init__(self, pipelines_dir, db=None, test_id=None, tuner_type='gp',
                 cv_splits=5, random_state=0):

        self._db = db

        self._pipelines_dir = pipelines_dir
        ensure_dir(self._pipelines_dir)

        self._cv_splits = cv_splits
        self._random_state = random_state

        self._tuner_type = tuner_type
        self._tuner_class = TUNERS[tuner_type]

        self._test_id = test_id

    def _dump(self, pipeline):
        if not pipeline.dumped:
            pipeline.fit(self.data_params)

            mlpipeline = pipeline.pipeline

            LOGGER.info("Dumping pipeline %s: %s", pipeline.id, pipeline.pipeline)
            LOGGER.info("Hyperparameters: %s", mlpipeline.get_hyperparameters())
            pipeline.dump(self._pipelines_dir)

        else:
            LOGGER.info("Skipping already dumped pipeline %s", pipeline.id)

        self.pipelines.append({
            'elapsed': (datetime.utcnow() - self.start_time).total_seconds(),
            'iterations': len(self.solutions) - 1,
            'cv_score': self.best_pipeline.score,
            'rank': self.best_pipeline.rank,
            'pipeline': self.best_pipeline.id,
            'load_time': self.load_time,
            'trivial_time': self.trivial_time,
            'fit_time': self.fit_time,
            'cv_time': self.cv_time
        })

    def _save_pipeline(self, pipeline):
        solution = pipeline.to_dict(True)
        solution['_id'] = pipeline.id
        solution['ts'] = datetime.utcnow()

        self.solutions.append(solution)

        if self._db:
            insertable = remove_dots(solution)
            insertable.pop('problem_doc')
            insertable['dataset'] = self.dataset_id
            insertable['tuner_type'] = self._tuner_type
            insertable['test_id'] = self._test_id
            self._db.solutions.insert_one(insertable)

    def _build_trivial_pipeline(self):
        LOGGER.info("Building the Trivial pipeline")
        try:
            method = TRIVIAL_PIPELINE_METHOD.get(self.task_type)
            pipeline_dict = {
                'name': 'trivial.{}'.format(method),
                'primitives': ['mlprimitives.trivial.TrivialPredictor'],
                'init_params': {
                    'mlprimitives.trivial.TrivialPredictor': {
                        'method': method
                    }
                }
            }
            pipeline = ABPipeline(pipeline_dict, self.loader, self.metric, self.problem_doc)
            pipeline.cv_score(self.data_params.X, self.data_params.y,
                              self.data_params.context, cv=self.kf)

            self._save_pipeline(pipeline)

            return pipeline

        except Exception:
            # if the Trivial pipeline crashes we can do nothing,
            # so we just log the error and move on.
            LOGGER.exception("The Trivial pipeline crashed.")

    def _load_template_json(self, template_name):
        if template_name.endswith('.json'):
            template_filename = template_name
            name = template_name[:-5]
        else:
            name = template_name
            template_name = template_name.replace('/', '.') + '.json'
            template_filename = os.path.join(TEMPLATES_DIR, template_name)

        if os.path.exists(template_filename):
            with open(template_filename, 'r') as template_file:
                template_dict = json.load(template_file)
                template_dict['name'] = name

            return template_dict

    def _find_template(self, template_name):
        match = {
            'metadata.name': template_name
        }
        cursor = self._db.pipelines.find(match)
        templates = list(cursor.sort('metadata.insert_ts', -1).limit(1))
        if templates:
            template = templates[0]
            template['name'] = template.pop('metadata')['name']
            template['template'] = str(template.pop('_id'))
            return restore_dots(template)

    def _load_template(self, template_name):
        if self._db:
            return self._find_template(template_name)

        return self._load_template_json(template_name)

    def _get_template(self, template_name=None):
        if template_name:
            template = self._load_template(template_name)
            if not template:
                raise ValueError("Template {} not found".format(template_name))

            return template

        else:
            problem_type = [
                self.data_modality,
                self.task_type,
                self.task_subtype
            ]

            for levels in reversed(range(1, 4)):
                # Try the following options:
                # modality/task/subtask/default
                # modality/task/default
                # modality/default
                template_name = '/'.join(problem_type[:levels] + ['default'])
                template = self._load_template(template_name)
                if template:
                    return template

            # Nothing has been found for this modality/task/subtask combination
            raise StopSearch()

    def _build_default_pipeline(self, template_dict):
        LOGGER.info("Building the default pipeline")
        pipeline = ABPipeline(template_dict, self.loader, self.metric, self.problem_doc)

        X = self.data_params.X
        y = self.data_params.y
        context = self.data_params.context

        X = pipeline.preprocess(X, y, context)

        try:
            pipeline.cv_score(X, y, context, cv=self.kf)

            LOGGER.info("Saving the default pipeline %s", pipeline.id)

            self._save_pipeline(pipeline)
        except Exception:
            # if the Default pipeline crashes we can do nothing,
            # so we just log the error and move on.
            LOGGER.exception("The Default pipeline crashed.")
            pipeline = None

        return pipeline, X, y, context

    def _get_tuner(self, pipeline, template_dict):
        # Build an MLPipeline to get the tunables and the default params
        mlpipeline = MLPipeline.from_dict(template_dict)

        tunables = []
        tunable_keys = []
        for block_name, params in mlpipeline.get_tunable_hyperparameters().items():
            for param_name, param_details in params.items():
                key = (block_name, param_name)
                param_type = param_details['type']
                param_type = PARAM_TYPES.get(param_type, param_type)
                if param_type == 'bool':
                    param_range = [True, False]
                else:
                    param_range = param_details.get('range') or param_details.get('values')

                value = HyperParameter(param_type, param_range)
                tunables.append((key, value))
                tunable_keys.append(key)

        # Create the tuner
        LOGGER.info('Creating %s tuner', self._tuner_class.__name__)
        tuner = self._tuner_class(tunables)

        if pipeline:
            # Add the default params and the score obtained by the default pipeline to the tuner.
            default_params = defaultdict(dict)
            for block_name, params in pipeline.pipeline.get_hyperparameters().items():
                for param, value in params.items():
                    key = (block_name, param)
                    if key in tunable_keys:
                        # default_params[key] = 'None' if value is None else value
                        default_params[key] = value

            tuner.add(default_params, 1 - pipeline.rank)

        return tuner

    def _set_checkpoint(self):
        next_checkpoint = self.checkpoints.pop(0)
        interval = next_checkpoint - self.current_checkpoint
        LOGGER.info("Setting %s seconds checkpoint in %s seconds", next_checkpoint, interval)
        signal.alarm(interval)
        self.current_checkpoint = next_checkpoint

    def _checkpoint(self, signum=None, frame=None, final=False):
        signal.alarm(0)

        checkpoint_name = 'Final' if final else str(self.current_checkpoint) + ' seconds'

        LOGGER.info("%s checkpoint reached", checkpoint_name)

        set_checkpoint = (not final) and bool(self.checkpoints)
        if set_checkpoint:
            self._set_checkpoint()

        try:
            if self.best_pipeline:
                self._dump(self.best_pipeline)

        except Exception:
            LOGGER.exception("Checkpoint dump crashed")

        if not set_checkpoint:
            self.current_checkpoint = None
            raise StopSearch()

    def search(self, d3mds, template_name=None, budget=None, checkpoints=None):
        # Problem variables
        problem_id = d3mds.get_problem_id()
        self.task_type = d3mds.get_task_type()
        self.task_subtype = d3mds.problem.get_task_subtype()

        if self.task_type == 'classification':
            self.kf = StratifiedKFold(
                n_splits=self._cv_splits,
                shuffle=True,
                random_state=self._random_state
            )
        else:
            self.kf = KFold(
                n_splits=self._cv_splits,
                shuffle=True,
                random_state=self._random_state
            )

        self.problem_doc = d3mds.problem_doc

        # Dataset variables
        self.dataset_id = d3mds.dataset_id
        self.data_modality = d3mds.get_data_modality()

        self.metric = d3mds.get_metric()

        self.loader = get_loader(self.data_modality, self.task_type)

        self.best_pipeline = None

        self.solutions = []
        self.checkpoints = sorted(checkpoints or [])
        self.current_checkpoint = 0
        self.pipelines = []

        self.load_time = None
        self.trivial_time = None
        self.fit_time = None
        self.cv_times = []
        self.cv_time = None

        try:
            self.start_time = datetime.utcnow()

            LOGGER.info("Running TA2 Search")
            LOGGER.info("Problem Id: %s", problem_id)
            LOGGER.info("    Data Modality: %s", self.data_modality)
            LOGGER.info("    Task type: %s", self.task_type)
            LOGGER.info("    Task subtype: %s", self.task_subtype)
            LOGGER.info("    Metric: %s", self.metric)
            LOGGER.info("    Checkpoints: %s", self.checkpoints)
            LOGGER.info("    Budget: %s", budget)

            if self.checkpoints:
                signal.signal(signal.SIGALRM, self._checkpoint)
                self._set_checkpoint()

            load_start = datetime.utcnow()
            self.data_params = self.loader.load(d3mds)
            load_end = datetime.utcnow()
            self.load_time = (load_end - load_start).total_seconds()
            import ipdb; ipdb.set_trace()

            # Build the trivial pipeline
            trivial_start = datetime.utcnow()
            self.best_pipeline = self._build_trivial_pipeline()
            trivial_end = datetime.utcnow()
            self.trivial_time = (trivial_end - trivial_start).total_seconds()

            template_dict = self._get_template(template_name)

            # Do not continue if there is no budget or no fit data
            if budget == 0 or not len(self.data_params.X):
                raise StopSearch()

            # Build the default pipeline
            fit_start = datetime.utcnow()
            default_pipeline, X, y, context = self._build_default_pipeline(template_dict)
            if default_pipeline:
                self.best_pipeline = default_pipeline

            fit_end = datetime.utcnow()
            self.fit_time = (fit_end - fit_start).total_seconds()

            if budget is not None:
                budget -= 1
            if budget == 0:
                raise StopSearch()

            # Build the tuner
            tuner = self._get_tuner(default_pipeline, template_dict)

            LOGGER.info("Starting the tuning loop")

            if budget is not None:
                iterator = range(budget)
            else:
                iterator = itertools.count()   # infinite range

            for iteration in iterator:
                proposed_params = tuner.propose()
                params = make_dumpable(proposed_params)

                cv_start = datetime.utcnow()
                pipeline_dict = template_dict.copy()
                pipeline_dict['hyperparameters'] = params
                pipeline = ABPipeline(pipeline_dict, self.loader,
                                      self.metric, self.problem_doc)
                try:
                    pipeline.cv_score(X, y, context, cv=self.kf)
                except StopSearch:
                    raise

                except Exception:
                    LOGGER.exception("Crash during cross validation")
                    tuner.add(proposed_params, -1000000)

                if pipeline.rank is not None:
                    tuner.add(proposed_params, 1 - pipeline.rank)

                    cv_end = datetime.utcnow()
                    cv_time = (cv_end - cv_start).total_seconds()
                    self.cv_times.append(cv_time)
                    self.cv_time = np.mean(self.cv_times)

                    LOGGER.info("Saving pipeline %s: %s", iteration + 1, pipeline.id)
                    self._save_pipeline(pipeline)

                    if not self.best_pipeline or (pipeline.rank < self.best_pipeline.rank):
                        self.best_pipeline = pipeline
                        LOGGER.info('Best pipeline so far: %s; rank: %s, score: %s',
                                    self.best_pipeline, self.best_pipeline.rank,
                                    self.best_pipeline.score)

        except (StopSearch, KeyboardInterrupt):
            pass

        finally:
            signal.alarm(0)

        if self.current_checkpoint:
            self._checkpoint(final=True)
        elif self.best_pipeline and not checkpoints:
            self._dump(self.best_pipeline)

        if self.best_pipeline:
            LOGGER.info('Best pipeline for problem %s found: %s; rank: %s, score: %s',
                        problem_id, self.best_pipeline,
                        self.best_pipeline.rank, self.best_pipeline.score)

        else:
            LOGGER.info('No pipeline could be found for problem %s', problem_id)

        return self.pipelines
