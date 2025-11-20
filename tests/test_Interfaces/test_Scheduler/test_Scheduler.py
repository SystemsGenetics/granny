from Granny.Interfaces.Scheduler.Scheduler import Scheduler
import pytest
from unittest.mock import MagicMock

# your test functions follow...


def test_add_analysis_without_dependencies():
    scheduler = Scheduler()
    analysis = MagicMock()
    scheduler.add_analysis(analysis, [])
    assert id(analysis) in scheduler.analyses

def test_add_analysis_with_dependencies():
    scheduler = Scheduler()
    parent = MagicMock()
    child = MagicMock()
    scheduler.add_analysis(parent, [])
    scheduler.add_analysis(child, [parent])
    assert id(child) in scheduler.graph[id(parent)]

def test_schedule_order():
    scheduler = Scheduler()
    a = MagicMock()
    b = MagicMock()
    scheduler.add_analysis(a, [])
    scheduler.add_analysis(b, [a])
    order = scheduler.schedule()
    assert order == [id(a), id(b)]

def test_schedule_cycle_detection():
    scheduler = Scheduler()
    a = MagicMock()
    b = MagicMock()
    scheduler.add_analysis(a, [b])
    scheduler.add_analysis(b, [a])
    with pytest.raises(ValueError):
        scheduler.schedule()

def test_run_calls_performAnalysis():
    scheduler = Scheduler()
    a = MagicMock()
    scheduler.add_analysis(a, [])
    scheduler.run()
    a.performAnalysis.assert_called_once()
