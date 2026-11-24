"""Pipelines: mnm's own multi-step runs over instances. A template is typed
stages; a run keeps each item's step in its run record, stops at review points, resumes after
a failure or a restart, and redoes one item alone."""
