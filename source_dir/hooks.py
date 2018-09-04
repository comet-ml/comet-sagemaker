from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from comet_ml import Experiment

import time

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.util.tf_export import tf_export


class _HookTimer(object):
    """Base timer for determining when Hooks should trigger.
    Should not be instantiated directly.
    """

    def __init__(self):
        pass

    def reset(self):
        """Resets the timer."""
        pass

    def should_trigger_for_step(self, step):
        """Return true if the timer should trigger for the specified step."""
        raise NotImplementedError

    def update_last_triggered_step(self, step):
        """Update the last triggered time and step number.
        Args:
          step: The current step.
        Returns:
          A pair `(elapsed_time, elapsed_steps)`, where `elapsed_time` is the number
          of seconds between the current trigger and the last one (a float), and
          `elapsed_steps` is the number of steps between the current trigger and
          the last one. Both values will be set to `None` on the first trigger.
        """
        raise NotImplementedError

    def last_triggered_step(self):
        """Returns the last triggered time step or None if never triggered."""
        raise NotImplementedError


@tf_export("train.SecondOrStepTimer")
class SecondOrStepTimer(_HookTimer):
    """Timer that triggers at most once every N seconds or once every N steps.
    """

    def __init__(self, every_secs=None, every_steps=None):
        self.reset()
        self._every_secs = every_secs
        self._every_steps = every_steps

        if self._every_secs is None and self._every_steps is None:
            raise ValueError(
                "Either every_secs or every_steps should be provided.")
        if (self._every_secs is not None) and (self._every_steps is not None):
            raise ValueError(
                "Can not provide both every_secs and every_steps.")

        super(SecondOrStepTimer, self).__init__()

    def reset(self):
        self._last_triggered_step = None
        self._last_triggered_time = None

    def should_trigger_for_step(self, step):
        """Return true if the timer should trigger for the specified step.
        Args:
          step: Training step to trigger on.
        Returns:
          True if the difference between the current time and the time of the last
          trigger exceeds `every_secs`, or if the difference between the current
          step and the last triggered step exceeds `every_steps`. False otherwise.
        """
        if self._last_triggered_step is None:
            return True

        if self._last_triggered_step == step:
            return False

        if self._every_secs is not None:
            if time.time() >= self._last_triggered_time + self._every_secs:
                return True

        if self._every_steps is not None:
            if step >= self._last_triggered_step + self._every_steps:
                return True

        return False

    def update_last_triggered_step(self, step):
        current_time = time.time()
        if self._last_triggered_time is None:
            elapsed_secs = None
            elapsed_steps = None
        else:
            elapsed_secs = current_time - self._last_triggered_time
            elapsed_steps = step - self._last_triggered_step

        self._last_triggered_time = current_time
        self._last_triggered_step = step
        return (elapsed_secs, elapsed_steps)

    def last_triggered_step(self):
        return self._last_triggered_step


class NeverTriggerTimer(_HookTimer):
    """Timer that never triggers."""

    def should_trigger_for_step(self, step):
        _ = step
        return False

    def update_last_triggered_step(self, step):
        _ = step
        return (None, None)

    def last_triggered_step(self):
        return None


''' Adapted tf.train.LogTrainerHook to work with Comet
'''


class CometSessionHook(tf.train.SessionRunHook):

    def __init__(self,
                 tensors,
                 parameters,
                 every_n_iter=1,
                 every_n_secs=None,
                 at_end=False,
                 formatter=None):
        self.parameters = parameters

        only_log_at_end = (
            at_end and (every_n_iter is None) and (every_n_secs is None))

        if (not only_log_at_end and
                (every_n_iter is None) == (every_n_secs is None)):
            raise ValueError(
                "either at_end and/or exactly one of every_n_iter and every_n_secs "
                "must be provided.")

        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)

        if not isinstance(tensors, dict):
            self._tag_order = tensors
            tensors = {item: item for item in tensors}

        else:
            self._tag_order = sorted(tensors.keys())
            self._tensors = tensors
            self._formatter = formatter
            self._timer = (
                NeverTriggerTimer() if only_log_at_end else
                SecondOrStepTimer(every_secs=every_n_secs,
                                  every_steps=every_n_iter))
            self._log_at_end = at_end

    def begin(self):
        self.experiment = Experiment(
            api_key="<COMET API KEY>",
            project_name="comet-sagemaker",
            workspace="<workspace>")
        self.experiment.log_multiple_params(self.parameters)

        self._timer.reset()
        self._iter_count = 0

        # Convert names to tensors if given
        self._current_tensors = {tag: _as_graph_element(tensor)
                                 for (tag, tensor) in self._tensors.items()}

    def after_create_session(self, session, coord):
        self.experiment.set_model_graph(session.graph)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)
        if self._should_trigger:
            # set up feed dict for sess run
            return SessionRunArgs(self._current_tensors)

        else:
            return None

    def _log_tensors(self, tensor_values):
        self.experiment.log_multiple_metrics(tensor_values)

        # log comet metrics
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(
            self._iter_count)
        if self._formatter:
            logging.info(self._formatter(tensor_values))
        else:
            stats = []
            comet_stats = {}
        for tag in self._tag_order:
            stats.append("%s = %s" % (tag, tensor_values[tag]))
            comet_stats[tag] = tensor_values[tag]

        if elapsed_secs is not None:
            logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
        else:
            logging.info("%s", ", ".join(stats))
            np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        _ = run_context
        if self._should_trigger:
            self._log_tensors(run_values.results)

        self._iter_count += 1

    def end(self, session):
        if self._log_at_end:
            values = session.run(self._current_tensors)
            self._log_tensors(values)


def _as_graph_element(obj):
    """Retrieves Graph element."""
    graph = ops.get_default_graph()
    if not isinstance(obj, six.string_types):
        if not hasattr(obj, "graph") or obj.graph != graph:
            raise ValueError("Passed %s should have graph attribute that is equal "
                             "to current graph %s." % (obj, graph))
        return obj
    if ":" in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(obj + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, "
                             "as this `Operation` has multiple outputs "
                             "(at least 2)." % obj)
    return element
